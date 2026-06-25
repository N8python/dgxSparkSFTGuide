#!/usr/bin/env python3
"""Evaluate a PII-extraction LoRA with SGLang offline inference.

Reads the compact eval JSONL format:
    {"text": ..., "spans": [{"text": ..., "label": ...}, ...]}

Builds prompts of the form:
    Extract the PII as JSON:\n{text}

Writes one JSON object per evaluated row and a summary JSON containing parse
rate and entity-level micro-F1. Entity matching is exact multiset matching on
(label, text), so duplicate mentions count correctly.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


TOKEN_COUNT_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens")
DEFAULT_OUTPUT_LOG = "eval_outputs/sglang_pii_eval.jsonl"


def disable_mp_semaphore_resource_tracker() -> None:
    """Avoid noisy Python resource_tracker crashes on SGLang shutdown.

    SGLang tears down a subprocess tree at shutdown. On Python 3.12 this can
    leave multiprocessing.resource_tracker with stale /mp-* semaphore
    bookkeeping and produce a harmless-but-scary KeyError traceback after the
    eval has already completed. The semaphore finalizers still unlink their own
    semaphores; this only prevents the separate tracker process from double
    tracking this specific class of multiprocessing semaphore.
    """
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


def install_resource_tracker_child_patch() -> None:
    """Patch resource_tracker subprocesses without editing the conda env.

    Python starts multiprocessing.resource_tracker as a fresh interpreter, so
    in-process monkeypatches do not reach it. Put a tiny sitecustomize module on
    PYTHONPATH so child Python processes use set.discard for unregisters that
    may arrive after SGLang has already torn down related subprocesses.
    """
    import os
    import tempfile

    patch_dir = Path(tempfile.mkdtemp(prefix="pii_eval_rt_patch_"))
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--eval-path", default="test.jsonl")
    ap.add_argument("--output-log", default=DEFAULT_OUTPUT_LOG)
    ap.add_argument("--summary-path", default=None, help="Defaults to output-log with .summary.json suffix.")
    ap.add_argument("--lora", action="store_true", default=False, help="Evaluate with the LoRA adapter at --lora-path. Off by default.")
    ap.add_argument("--lora-path", default="checkpoints/qwen3_4b_pii_lora_r32_1ep")
    ap.add_argument("--lora-name", default="pii", help="Name used to register the LoRA adapter with SGLang.")
    ap.add_argument("--no-lora", action="store_false", dest="lora", help=argparse.SUPPRESS)
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    ap.add_argument("--batch-size", type=int, default=0, help="Prompt chunk size. 0 submits all pending prompts at once and lets SGLang schedule batching internally.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
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


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit and len(rows) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            spans = row.get("spans", [])
            if not isinstance(text, str):
                raise ValueError(f"{path}:{index + 1} has no string-valued text field")
            if isinstance(spans, str):
                spans = ast.literal_eval(spans)
            if not isinstance(spans, list):
                raise ValueError(f"{path}:{index + 1} has no list-valued spans field")
            rows.append({"index": index, "text": text, "gold_spans": spans})
    return rows


def apply_chat_template(tokenizer: Any, text: str) -> str:
    messages = [{"role": "user", "content": f"Extract the PII as JSON:\n{text}"}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def extract_balanced_json(text: str) -> str | None:
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        stack: list[str] = []
        in_string = False
        escaped = False
        for pos in range(start, len(text)):
            current = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current in "[{":
                stack.append(current)
            elif current in "]}":
                if not stack:
                    break
                opener = stack.pop()
                if (opener, current) not in (("[", "]"), ("{", "}")):
                    break
                if not stack:
                    return text[start : pos + 1]
    return None


def parse_json_answer(raw_output: str) -> tuple[bool, Any, str | None]:
    stripped = raw_output.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        fenced = stripped.strip("`").strip()
        if fenced.lower().startswith("json"):
            fenced = fenced[4:].strip()
        candidates.append(fenced)
    extracted = extract_balanced_json(stripped)
    if extracted is not None:
        candidates.append(extracted)
    seen: set[str] = set()
    last_error: str | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return True, json.loads(candidate), None
        except Exception as exc:  # noqa: BLE001 - report parse failure compactly.
            last_error = f"{type(exc).__name__}: {exc}"
    return False, None, last_error


def span_items(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("spans", "pii", "entities", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return [parsed]
    return []


def normalize_span(span: Any) -> tuple[str, str] | None:
    if not isinstance(span, dict):
        return None
    label = span.get("label", span.get("type", span.get("category")))
    text = span.get("text", span.get("value", span.get("span")))
    if label is None or text is None:
        return None
    label_s = str(label).strip()
    text_s = str(text).strip()
    if not label_s or not text_s:
        return None
    return label_s, text_s


def span_counter(spans: Any) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for item in span_items(spans):
        normalized = normalize_span(item)
        if normalized is not None:
            counter[normalized] += 1
    return counter


def score_prediction(gold_spans: list[dict[str, Any]], parsed_answer: Any) -> tuple[int, int, int, list[dict[str, str]]]:
    gold = span_counter(gold_spans)
    pred = span_counter(parsed_answer)
    tp = sum((gold & pred).values())
    fp = sum((pred - gold).values())
    fn = sum((gold - pred).values())
    pred_spans = [
        {"label": label, "text": text}
        for (label, text), count in pred.items()
        for _ in range(count)
    ]
    return tp, fp, fn, pred_spans


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


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


def empty_totals() -> dict[str, int]:
    totals = {"rows": 0, "parsed": 0, "tp": 0, "fp": 0, "fn": 0, "parsed_tp": 0, "parsed_fp": 0, "parsed_fn": 0}
    for field in TOKEN_COUNT_FIELDS:
        totals[field] = 0
    return totals


def extract_token_counts(meta_info: dict[str, Any] | None) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    meta_info = meta_info or {}
    for field in TOKEN_COUNT_FIELDS:
        value = meta_info.get(field)
        counts[field] = int(value) if value is not None else None
    return counts


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
            totals["parsed"] += int(bool(record.get("parse_ok")))
            totals["tp"] += int(record.get("tp", 0))
            totals["fp"] += int(record.get("fp", 0))
            totals["fn"] += int(record.get("fn", 0))
            if record.get("parse_ok"):
                totals["parsed_tp"] += int(record.get("tp", 0))
                totals["parsed_fp"] += int(record.get("fp", 0))
                totals["parsed_fn"] += int(record.get("fn", 0))
            for field in TOKEN_COUNT_FIELDS:
                totals[field] += int(record.get(field) or 0)
    return seen, totals



def score_and_write_record(
    *,
    row: dict[str, Any],
    prompt: str,
    raw_output: str,
    meta_info: dict[str, Any] | None,
    handle: Any,
    totals: dict[str, int],
) -> dict[str, Any]:
    parse_ok, parsed_answer, parse_error = parse_json_answer(raw_output)
    tp, fp, fn, predicted_spans = score_prediction(row["gold_spans"], parsed_answer if parse_ok else [])
    token_counts = extract_token_counts(meta_info)
    totals["rows"] += 1
    totals["parsed"] += int(parse_ok)
    totals["tp"] += tp
    totals["fp"] += fp
    totals["fn"] += fn
    if parse_ok:
        totals["parsed_tp"] += tp
        totals["parsed_fp"] += fp
        totals["parsed_fn"] += fn
    for field, value in token_counts.items():
        totals[field] += value or 0
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    record = {
        "index": row["index"],
        "prompt": prompt,
        "gold_spans": row["gold_spans"],
        "raw_output": raw_output,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "parsed_answer": parsed_answer if parse_ok else None,
        "predicted_spans": predicted_spans,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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
    precision, recall, micro_f1 = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    parse_rate = totals["parsed"] / totals["rows"] if totals["rows"] else 0.0
    return {"parse": f"{parse_rate:.1%}", "micro_f1": f"{micro_f1:.3f}", "tok_out": str(totals["completion_tokens"]), "p": f"{precision:.3f}", "r": f"{recall:.3f}"}


def main() -> None:
    install_resource_tracker_child_patch()
    disable_mp_semaphore_resource_tracker()

    args = parse_args()
    eval_path = Path(args.eval_path)
    output_log = default_output_log_for_args(args)
    summary_path = Path(args.summary_path) if args.summary_path else output_log.with_suffix(".summary.json")
    output_log.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.lora and not Path(args.lora_path).exists():
        raise FileNotFoundError(f"LoRA path does not exist: {args.lora_path}. Omit --lora to evaluate the base model.")

    rows = load_rows(eval_path, args.limit)
    seen, totals = load_existing_metrics(output_log) if args.resume else (set(), empty_totals())
    pending = [row for row in rows if row["index"] not in seen]

    from transformers import AutoTokenizer
    import sglang as sgl

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    prompts = {row["index"]: apply_chat_template(tokenizer, row["text"]) for row in pending}

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
                row_groups = [pending[start : start + args.batch_size] for start in range(0, len(pending), args.batch_size)]
            else:
                row_groups = [pending]
            for batch_rows in tqdm(row_groups, desc="SGLang eval submissions", unit="submission"):
                if not batch_rows:
                    continue
                batch_prompts = [prompts[row["index"]] for row in batch_rows]
                final_outputs: list[str | None] = [None] * len(batch_rows)
                final_meta_infos: list[dict[str, Any] | None] = [None] * len(batch_rows)
                completed = [False] * len(batch_rows)
                generate_kwargs: dict[str, Any] = {
                    "prompt": batch_prompts,
                    "sampling_params": sampling_params,
                    "stream": True,
                    "rid": [f"eval-{row['index']}" for row in batch_rows],
                }
                if args.lora:
                    generate_kwargs["lora_path"] = [args.lora_name] * len(batch_rows)
                stream = llm.generate(**generate_kwargs)
                with tqdm(total=len(batch_rows), desc="Completed eval examples", unit="example", leave=False) as completed_bar:
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
                                    prompt=prompts[row["index"]],
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
                            raise RuntimeError(f"No streamed output received for eval index {row['index']}")
                        score_and_write_record(
                            row=row,
                            prompt=prompts[row["index"]],
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

    precision, recall, micro_f1 = precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])
    parsed_precision, parsed_recall, parsed_micro_f1 = precision_recall_f1(totals["parsed_tp"], totals["parsed_fp"], totals["parsed_fn"])
    elapsed = time.perf_counter() - start
    parse_rate = totals["parsed"] / totals["rows"] if totals["rows"] else 0.0
    eval_score = parsed_micro_f1 * parse_rate
    prompt_tokens_per_second = totals["prompt_tokens"] / max(elapsed, 1e-9)
    tokens_generated_per_second = totals["completion_tokens"] / max(elapsed, 1e-9)
    summary = {
        "eval_path": str(eval_path),
        "output_log": str(output_log),
        "model_path": args.model_path,
        "lora_path": args.lora_path if args.lora else None,
        "rows": totals["rows"],
        "parse_ok": totals["parsed"],
        "parse_rate": parse_rate,
        "eval_score": eval_score,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "parsed_tp": totals["parsed_tp"],
        "parsed_fp": totals["parsed_fp"],
        "parsed_fn": totals["parsed_fn"],
        "parsed_micro_precision": parsed_precision,
        "parsed_micro_recall": parsed_recall,
        "parsed_micro_f1": parsed_micro_f1,
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
    print(f"eval_score: {eval_score:.6f}")
    print(f"parsed_micro_f1: {parsed_micro_f1:.6f}")
    print(f"fraction_parsed: {parse_rate:.6f}")
    print(f"prompt_tokens_per_second: {prompt_tokens_per_second:.2f}")
    print(f"tokens_generated_per_second: {tokens_generated_per_second:.2f}")


if __name__ == "__main__":
    main()
