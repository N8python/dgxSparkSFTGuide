"""Unsloth LoRA throughput benchmark.

Sibling to unsloth_sft_bench.py, but trains a PEFT LoRA adapter instead of the
full Qwen3 backbone. Defaults are meant to probe the "tiny gradient payload"
case: rank-32 LoRA, lr=1e-4, and all non-embedding/non-lm-head linear modules.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import unsloth  # noqa: F401  (must be imported before transformers)
from unsloth import FastLanguageModel
import torch

import utils as base


def discover_nonembedding_linear_targets(model: torch.nn.Module) -> list[str]:
    """Return unique leaf names for trainable transformer Linear modules."""
    excluded_fragments = (
        "embed",
        "embedding",
        "lm_head",
        "score",
        "classifier",
    )
    targets: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered = name.lower()
        if any(fragment in lowered for fragment in excluded_fragments):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf:
            targets.add(leaf)
    return sorted(targets)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--allow-model-downloads", action="store_true", help="Allow downloading model files if they are missing from the local Hugging Face cache.")
    ap.add_argument("--train-path", default="train-messages.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=56)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", nargs="*", default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--pad-to-multiple", type=int, default=8)
    ap.add_argument("--peak-flops", type=float, default=100e12)
    ap.add_argument("--mem-fraction", type=float, default=0.65)
    ap.add_argument("--log-every", type=int, default=8)
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        use_gradient_checkpointing=args.gradient_checkpointing,
        local_files_only=not args.allow_model_downloads,
        use_exact_model_name=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    target_modules = args.target_modules or discover_nonembedding_linear_targets(model)
    if not target_modules:
        raise RuntimeError("No LoRA target modules discovered; pass --target-modules explicitly.")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.gradient_checkpointing,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    load_start = time.perf_counter()
    rows = base.load_jsonl_messages(Path(args.train_path))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    examples = [
        enc
        for messages in rows
        if (enc := base.encode_messages(tokenizer, messages, max_seq_len=args.max_seq_len)) is not None
    ]
    batches = base.build_batches(
        examples,
        per_device_batch_size=args.per_device_batch_size,
        pad_to_multiple=args.pad_to_multiple,
        shuffle_seed=0,
        bucket_by_length=True,
    )
    preprocess_seconds = time.perf_counter() - load_start

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )

    total_param_count, trainable_param_count = count_parameters(model)
    base_param_count = sum(p.numel() for name, p in model.named_parameters() if "lora_" not in name)
    print(json.dumps({
        "event": "setup",
        "library": "unsloth_lora",
        "unsloth_version": unsloth.__version__,
        "torch": torch.__version__,
        "total_param_count": total_param_count,
        "base_param_count_for_flops": base_param_count,
        "trainable_param_count": trainable_param_count,
        "trainable_fraction": trainable_param_count / total_param_count if total_param_count else 0.0,
        "target_modules": target_modules,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "num_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "encoded_examples": len(examples),
        "num_batches": len(batches),
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "max_seq_len": args.max_seq_len,
        "preprocess_seconds": preprocess_seconds,
    }, sort_keys=True), flush=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    measured_start = None
    measured = {"input": 0, "label": 0, "padded": 0, "s2": 0, "microsteps": 0}
    total_input = 0
    last_loss = math.nan

    optimizer.zero_grad(set_to_none=True)
    step = 0
    batch_index = 0
    while step < args.max_steps:
        if step >= args.warmup_steps and measured_start is None:
            torch.cuda.synchronize()
            measured_start = time.perf_counter()
        batch_examples = batches[batch_index % len(batches)]
        batch_index += 1
        batch, stats = base.collate_batch(
            batch_examples,
            pad_token_id=pad_token_id,
            pad_to_multiple=args.pad_to_multiple,
            device=device,
        )
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        last_loss = float(loss.detach().cpu())
        total_input += stats.input_tokens
        if measured_start is not None:
            measured["input"] += stats.input_tokens
            measured["label"] += stats.label_tokens
            measured["padded"] += stats.padded_input_tokens
            measured["s2"] += stats.attention_s2
            measured["microsteps"] += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(json.dumps({
                "event": "progress",
                "optimizer_step": step,
                "elapsed_seconds": elapsed,
                "input_tokens": total_input,
                "last_loss": last_loss,
                "last_batch_padded_shape": list(batch["input_ids"].shape),
            }, sort_keys=True), flush=True)

    torch.cuda.synchronize()
    end = time.perf_counter()
    measured_elapsed = end - (measured_start if measured_start is not None else start)
    useful = base.flops_for_window(
        param_count=base_param_count,
        num_layers=int(model.config.num_hidden_layers),
        hidden_size=int(model.config.hidden_size),
        padded_input_tokens=measured["padded"],
        attention_s2=measured["s2"],
        include_checkpoint_recompute=False,
    )
    print(json.dumps({
        "event": "summary",
        "library": "unsloth_lora",
        "optimizer_steps": step,
        "measured_microsteps": measured["microsteps"],
        "measured_elapsed_seconds": measured_elapsed,
        "measured_input_tokens": measured["input"],
        "measured_label_tokens": measured["label"],
        "measured_input_tokens_per_second": measured["input"] / max(measured_elapsed, 1e-9),
        "measured_label_tokens_per_second": measured["label"] / max(measured_elapsed, 1e-9),
        "padding_efficiency": measured["input"] / measured["padded"] if measured["padded"] else 0.0,
        "last_loss": last_loss,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "useful_tflops_per_second": useful["total_flops"] / max(measured_elapsed, 1e-9) / 1e12,
        "useful_mfu": useful["total_flops"] / max(measured_elapsed, 1e-9) / args.peak_flops,
        "peak_flops_denominator": args.peak_flops,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
