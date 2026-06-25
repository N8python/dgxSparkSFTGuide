#!/usr/bin/env python3
"""Evaluate GSM8K with SGLang offline inference.

The prompt is the GSM8K question verbatim. The prediction is extracted from the
last \boxed{...} in the model output; if no boxed answer exists, the prediction
falls back to the last number in the output.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


TOKEN_COUNT_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens")
DEFAULT_OUTPUT_LOG = "eval_outputs/sglang_gsm8k_eval.jsonl"
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")
FRAC_RE = re.compile(r"\\frac\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}")


def install_resource_tracker_child_patch() -> None:
    """Patch resource_tracker subprocesses without editing the conda env."""
    patch_dir = Path(tempfile.mkdtemp(prefix="gsm8k_eval_rt_patch_"))
    sitecustomize = patch_dir / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "try:",
                "    import inspect",
                "    import multiprocessing.resource_tracker as rt",
                "    src = inspect.getsource(rt.main)",
                "    src = src.replace('cache[rtype].remove(name)', 'cache[rtype].discard(name)')",
                "    exec(src, rt.__dict__)",
                "except Exception:",
                "    pass",
                "",
            ]
        ),
        encoding="utf-8",
    )
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(patch_dir) if not existing else f"{patch_dir}{os.pathsep}{existing}"


def disable_mp_semaphore_resource_tracker() -> None:
    """Avoid noisy Python resource_tracker crashes on SGLang shutdown."""
    import multiprocessing.resource_tracker as resource_tracker

    original_register = resource_tracker.register
    original_unregister = resource_tracker.unregister

    def is_mp_semaphore(name: Any, rtype: str) -> bool:
        name_s = name.decode("ascii", errors="ignore") if isinstance(name, bytes) else str(name)
        return rtype == "semaphore" and name_s.startswith("/mp-")

    def register(name: Any, rtype: str) -> None:
        if is_mp_semaphore(name, rtype):
            return
        original_register(name, rtype)

    def unregister(name: Any, rtype: str) -> None:
        if is_mp_semaphore(name, rtype):
            return
        original_unregister(name, rtype)

    resource_tracker.register = register
    resource_tracker.unregister = unregister


def sglang_supports_top_k() -> bool:
    try:
        from sglang.srt.sampling.sampling_params import SamplingParams

        return "top_k" in inspect.signature(SamplingParams).parameters
    except Exception:
        return False


def sglang_supports_repetition_penalty() -> bool:
    try:
        from sglang.srt.sampling.sampling_params import SamplingParams

        return "repetition_penalty" in inspect.signature(SamplingParams).parameters
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--eval-path", default="data/gsm8k-test.jsonl")
    ap.add_argument("--output-log", default=DEFAULT_OUTPUT_LOG)
    ap.add_argument("--summary-path", default=None, help="Defaults to output-log with .summary.json suffix.")
    ap.add_argument("--dataset-id", default="openai/gsm8k")
    ap.add_argument("--dataset-config", default="main")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--force-download", action="store_true", help="Re-download GSM8K even if --eval-path exists.")
    ap.add_argument("--lora", action="store_true", default=False, help="Evaluate with the LoRA adapter at --lora-path. Off by default.")
    ap.add_argument("--lora-path", default="checkpoints/qwen3_4b_pii_lora_r32_1ep")
    ap.add_argument("--lora-name", default="gsm8k", help="Name used to register the LoRA adapter with SGLang.")
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    ap.add_argument("--batch-size", type=int, default=0, help="Prompt chunk size. 0 submits all pending prompts at once and lets SGLang schedule batching internally.")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 disables top-k filtering, matching SGLang's native default.")
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help="1.0 disables repetition penalty, matching SGLang's native default.")
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem-fraction-static", type=float, default=0.40)
    ap.add_argument("--max-running-requests", type=int, default=128)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--lora-target-modules", nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="LoRA target module names for SGLang LoRA initialization.")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true", help="Skip indices already present in output-log and include them in the final metrics.")
    return ap.parse_args()


def slugify_lora_source(lora_path: str) -> str:
    source = Path(lora_path).name or "lora"
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in source)


def default_output_log_for_args(args: argparse.Namespace) -> Path:
    output_log = Path(args.output_log)
    if args.lora and args.output_log == DEFAULT_OUTPUT_LOG:
        suffix = "".join(output_log.suffixes) or output_log.suffix
        stem = output_log.name[: -len(suffix)] if suffix else output_log.name
        output_log = output_log.with_name(f"{stem}-{slugify_lora_source(args.lora_path)}{suffix}")
    return output_log


def download_gsm8k_jsonl(path: Path, dataset_id: str, config: str, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    offset = 0
    page_size = 100
    with path.open("w", encoding="utf-8") as handle:
        while True:
            query = urllib.parse.urlencode(
                {
                    "dataset": dataset_id,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": page_size,
                }
            )
            url = f"https://datasets-server.huggingface.co/rows?{query}"
            request = urllib.request.Request(url, headers={"User-Agent": "sglang-gsm8k-eval/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = payload.get("rows") or []
            if not rows:
                break
            for item in rows:
                row = item.get("row", {})
                question = row.get("question")
                answer = row.get("answer")
                if not isinstance(question, str) or not isinstance(answer, str):
                    raise ValueError(f"Unexpected GSM8K row shape at offset {offset}: {row!r}")
                handle.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_written += 1
            offset += len(rows)
            total = payload.get("num_rows_total")
            if total is not None and offset >= int(total):
                break
    if rows_written == 0:
        raise RuntimeError(f"No rows downloaded from {dataset_id}/{config}/{split}")


def ensure_eval_data(path: Path, dataset_id: str, config: str, split: str, force_download: bool) -> None:
    if force_download or not path.exists():
        download_gsm8k_jsonl(path, dataset_id, config, split)


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit and len(rows) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(question, str):
                raise ValueError(f"{path}:{index + 1} has no string-valued question field")
            if not isinstance(answer, str):
                raise ValueError(f"{path}:{index + 1} has no string-valued answer field")
            gold_answer, gold_method = extract_gold_answer(answer)
            rows.append(
                {
                    "index": index,
                    "question": question,
                    "gold_answer_raw": answer,
                    "gold_answer": gold_answer,
                    "gold_extraction_method": gold_method,
                }
            )
    return rows


def clean_number(value: str) -> str:
    return value.replace(",", "").strip().strip("$ ")


def decimal_from_number(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(clean_number(value))
    except InvalidOperation:
        return None


def extract_latex_fraction(text: str) -> str | None:
    matches = list(FRAC_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    numerator = decimal_from_number(match.group(1))
    denominator = decimal_from_number(match.group(2))
    if numerator is None or denominator in (None, Decimal(0)):
        return None
    return str(numerator / denominator)


def extract_last_number(text: str) -> str | None:
    fraction = extract_latex_fraction(text)
    if fraction is not None:
        return fraction
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return None
    return clean_number(matches[-1].group(0))


def iter_boxed_contents(text: str):
    starts: list[int] = []
    starts.extend(match.start() for match in re.finditer(r"\\boxed\s*\{", text))
    starts.extend(match.start() for match in re.finditer(r"(?<!\\)boxed\s*\{", text))
    for start in sorted(set(starts)):
        brace_start = text.find("{", start)
        if brace_start < 0:
            continue
        depth = 0
        escaped = False
        for pos in range(brace_start, len(text)):
            char = text[pos]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[brace_start + 1 : pos]
                    break


def extract_prediction(raw_output: str) -> tuple[str | None, str]:
    boxed_values = list(iter_boxed_contents(raw_output))
    for content in reversed(boxed_values):
        value = extract_last_number(content)
        if value is not None:
            return value, "boxed"
    value = extract_last_number(raw_output)
    if value is not None:
        return value, "last_number"
    return None, "none"


def extract_gold_answer(gold_output: str) -> tuple[str | None, str]:
    if "####" in gold_output:
        value = extract_last_number(gold_output.rsplit("####", 1)[-1])
        if value is not None:
            return value, "hash_answer"
    value = extract_last_number(gold_output)
    if value is not None:
        return value, "last_number"
    return None, "none"


def answers_match(predicted: str | None, gold: str | None) -> bool:
    predicted_decimal = decimal_from_number(predicted)
    gold_decimal = decimal_from_number(gold)
    return predicted_decimal is not None and gold_decimal is not None and predicted_decimal == gold_decimal


def get_generated_text(output: Any) -> str:
    if isinstance(output, dict):
        value = output.get("text")
        if isinstance(value, str):
            return value
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if isinstance(first.get("text"), str):
                    return first["text"]
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
    return str(output)


def get_meta_info(output: Any) -> dict[str, Any]:
    if isinstance(output, dict) and isinstance(output.get("meta_info"), dict):
        return output["meta_info"]
    return {}


def extract_token_counts(meta_info: dict[str, Any] | None) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    meta_info = meta_info or {}
    for field in TOKEN_COUNT_FIELDS:
        value = meta_info.get(field)
        counts[field] = int(value) if value is not None else None
    return counts


def empty_totals() -> dict[str, int]:
    totals = {"rows": 0, "extracted": 0, "correct": 0}
    for field in TOKEN_COUNT_FIELDS:
        totals[field] = 0
    return totals


def load_existing_metrics(output_log: Path) -> tuple[set[int], dict[str, int]]:
    seen: set[int] = set()
    totals = empty_totals()
    if not output_log.exists():
        return seen, totals
    with output_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["index"])
            seen.add(index)
            totals["rows"] += 1
            totals["extracted"] += int(record.get("predicted_answer") is not None)
            totals["correct"] += int(bool(record.get("correct")))
            for field in TOKEN_COUNT_FIELDS:
                totals[field] += int(record.get(field) or 0)
    return seen, totals


def score_and_write_record(
    *,
    row: dict[str, Any],
    raw_output: str,
    meta_info: dict[str, Any] | None,
    handle: Any,
    totals: dict[str, int],
) -> dict[str, Any]:
    predicted_answer, prediction_method = extract_prediction(raw_output)
    correct = answers_match(predicted_answer, row["gold_answer"])
    token_counts = extract_token_counts(meta_info)
    totals["rows"] += 1
    totals["extracted"] += int(predicted_answer is not None)
    totals["correct"] += int(correct)
    for field, value in token_counts.items():
        totals[field] += value or 0
    record = {
        "index": row["index"],
        "question": row["question"],
        "gold_answer_raw": row["gold_answer_raw"],
        "gold_answer": row["gold_answer"],
        "gold_extraction_method": row["gold_extraction_method"],
        "raw_output": raw_output,
        "predicted_answer": predicted_answer,
        "prediction_extraction_method": prediction_method,
        "correct": correct,
    }
    record.update(token_counts)
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return record


def stream_chunks_with_fallback_indices(stream_item: Any):
    if isinstance(stream_item, list):
        for fallback_index, chunk in enumerate(stream_item):
            yield fallback_index, chunk
    else:
        yield None, stream_item


def get_chunk_index(chunk: Any, fallback_index: int | None, batch_size: int) -> int:
    if isinstance(chunk, dict):
        for key in ("index", "request_index"):
            value = chunk.get(key)
            if value is not None:
                return int(value)
        meta_info = chunk.get("meta_info")
        if isinstance(meta_info, dict):
            for key in ("index", "request_index"):
                value = meta_info.get(key)
                if value is not None:
                    return int(value)
    if fallback_index is not None:
        return int(fallback_index)
    if batch_size == 1:
        return 0
    raise RuntimeError(f"Could not determine streamed chunk index for batch_size={batch_size}: {chunk!r}")


def is_finished_chunk(chunk: Any) -> bool:
    if not isinstance(chunk, dict):
        return False
    meta_info = chunk.get("meta_info")
    if not isinstance(meta_info, dict):
        return False
    return meta_info.get("finish_reason") is not None


def progress_postfix(totals: dict[str, int]) -> dict[str, str]:
    accuracy = totals["correct"] / totals["rows"] if totals["rows"] else 0.0
    extracted = totals["extracted"] / totals["rows"] if totals["rows"] else 0.0
    return {"acc": f"{accuracy:.3f}", "extracted": f"{extracted:.1%}", "tok_out": str(totals["completion_tokens"])}


def main() -> None:
    install_resource_tracker_child_patch()
    disable_mp_semaphore_resource_tracker()

    args = parse_args()
    eval_path = Path(args.eval_path)
    output_log = default_output_log_for_args(args)
    summary_path = Path(args.summary_path) if args.summary_path else output_log.with_suffix(".summary.json")
    output_log.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_eval_data(eval_path, args.dataset_id, args.dataset_config, args.dataset_split, args.force_download)

    if args.lora and not Path(args.lora_path).exists():
        raise FileNotFoundError(f"LoRA path does not exist: {args.lora_path}. Omit --lora to evaluate the base model.")

    rows = load_rows(eval_path, args.limit)
    seen, totals = load_existing_metrics(output_log) if args.resume else (set(), empty_totals())
    pending = [row for row in rows if row["index"] not in seen]

    import sglang as sgl

    engine_kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "context_length": args.context_length,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.lora:
        engine_kwargs.update(
            {
                "enable_lora": True,
                "max_lora_rank": args.max_lora_rank,
                "lora_target_modules": args.lora_target_modules,
                "lora_paths": [f"{args.lora_name}={args.lora_path}"],
            }
        )

    llm = sgl.Engine(**engine_kwargs)
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.top_k < -1:
        raise ValueError("--top-k must be >= -1. Use -1 to disable top-k filtering.")
    if args.top_k != -1:
        if not sglang_supports_top_k():
            raise RuntimeError("This SGLang install does not support top_k sampling.")
        sampling_params["top_k"] = args.top_k
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be > 0. Use 1.0 to disable repetition penalty.")
    if args.repetition_penalty != 1.0:
        if not sglang_supports_repetition_penalty():
            raise RuntimeError("This SGLang install does not support repetition_penalty sampling.")
        sampling_params["repetition_penalty"] = args.repetition_penalty

    start = time.perf_counter()
    try:
        mode = "a" if args.resume else "w"
        with output_log.open(mode, encoding="utf-8") as handle:
            if args.batch_size and args.batch_size > 0:
                row_groups = [pending[start_index : start_index + args.batch_size] for start_index in range(0, len(pending), args.batch_size)]
            else:
                row_groups = [pending]
            for batch_rows in tqdm(row_groups, desc="SGLang GSM8K submissions", unit="submission"):
                if not batch_rows:
                    continue
                batch_prompts = [row["question"] for row in batch_rows]
                final_outputs: list[str | None] = [None] * len(batch_rows)
                final_meta_infos: list[dict[str, Any] | None] = [None] * len(batch_rows)
                completed = [False] * len(batch_rows)
                generate_kwargs: dict[str, Any] = {
                    "prompt": batch_prompts,
                    "sampling_params": sampling_params,
                    "stream": True,
                    "rid": [f"gsm8k-{row['index']}" for row in batch_rows],
                }
                if args.lora:
                    generate_kwargs["lora_path"] = [args.lora_name] * len(batch_rows)
                stream = llm.generate(**generate_kwargs)
                with tqdm(total=len(batch_rows), desc="Completed GSM8K examples", unit="example", leave=False) as completed_bar:
                    for stream_item in stream:
                        for fallback_index, chunk in stream_chunks_with_fallback_indices(stream_item):
                            chunk_index = get_chunk_index(chunk, fallback_index, len(batch_rows))
                            if not 0 <= chunk_index < len(batch_rows):
                                raise RuntimeError(f"Streamed chunk index out of range: {chunk_index} for batch size {len(batch_rows)}")
                            final_outputs[chunk_index] = get_generated_text(chunk)
                            final_meta_infos[chunk_index] = get_meta_info(chunk)
                            if is_finished_chunk(chunk) and not completed[chunk_index]:
                                row = batch_rows[chunk_index]
                                score_and_write_record(
                                    row=row,
                                    raw_output=final_outputs[chunk_index] or "",
                                    meta_info=final_meta_infos[chunk_index],
                                    handle=handle,
                                    totals=totals,
                                )
                                completed[chunk_index] = True
                                completed_bar.update(1)
                                completed_bar.set_postfix(progress_postfix(totals))
                                handle.flush()
                    for chunk_index, row in enumerate(batch_rows):
                        if completed[chunk_index]:
                            continue
                        if final_outputs[chunk_index] is None:
                            raise RuntimeError(f"No streamed output received for GSM8K index {row['index']}")
                        score_and_write_record(
                            row=row,
                            raw_output=final_outputs[chunk_index] or "",
                            meta_info=final_meta_infos[chunk_index],
                            handle=handle,
                            totals=totals,
                        )
                        completed[chunk_index] = True
                        completed_bar.update(1)
                        completed_bar.set_postfix(progress_postfix(totals))
                    handle.flush()
    finally:
        llm.shutdown()

    elapsed = time.perf_counter() - start
    accuracy = totals["correct"] / totals["rows"] if totals["rows"] else 0.0
    extraction_rate = totals["extracted"] / totals["rows"] if totals["rows"] else 0.0
    prompt_tokens_per_second = totals["prompt_tokens"] / max(elapsed, 1e-9)
    tokens_generated_per_second = totals["completion_tokens"] / max(elapsed, 1e-9)
    summary = {
        "eval_path": str(eval_path),
        "output_log": str(output_log),
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "model_path": args.model_path,
        "lora_path": args.lora_path if args.lora else None,
        "rows": totals["rows"],
        "extracted": totals["extracted"],
        "extraction_rate": extraction_rate,
        "correct": totals["correct"],
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "examples_per_second": totals["rows"] / max(elapsed, 1e-9),
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "cached_tokens": totals["cached_tokens"],
        "prompt_tokens_per_second": prompt_tokens_per_second,
        "completion_tokens_per_second": tokens_generated_per_second,
        "tokens_generated_per_second": tokens_generated_per_second,
        "sampling_params": sampling_params,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"gsm8k_accuracy: {accuracy:.6f}")
    print(f"correct: {totals['correct']}/{totals['rows']}")
    print(f"fraction_extracted: {extraction_rate:.6f}")
    print(f"prompt_tokens_per_second: {prompt_tokens_per_second:.2f}")
    print(f"tokens_generated_per_second: {tokens_generated_per_second:.2f}")


if __name__ == "__main__":
    main()
