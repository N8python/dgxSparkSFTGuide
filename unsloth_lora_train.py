"""Real LoRA SFT training run: unsloth PEFT adapter training for Qwen3.

Sibling to unsloth_sft_train.py, but trains and saves a LoRA adapter instead
of full-finetuning the Qwen3 backbone. Defaults follow the validated rank-32
bf16 LoRA setup used by the throughput benchmark.
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
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm

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


def scaled_adam_beta2(*, reference_beta2: float, reference_batch_size: float, batch_size: float) -> float:
    """Scale Adam beta2 by preserving second-moment half-life in tokens.

    Following arXiv:2507.07101, if beta2 is known for a reference batch
    size B, then the beta2 for a new batch size B* is beta2 ** (B* / B).
    This assumes the sequence length/token budget per example is comparable.
    """
    if not 0.0 < reference_beta2 < 1.0:
        raise ValueError(f"reference_beta2 must be in (0, 1), got {reference_beta2}")
    if reference_batch_size <= 0:
        raise ValueError(f"reference_batch_size must be positive, got {reference_batch_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return reference_beta2 ** (batch_size / reference_batch_size)


def download_tulu_reference_jsonl(
    *,
    output_path: Path,
    dataset_id: str,
    split: str,
    max_samples: int,
) -> None:
    from datasets import load_dataset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            out = {
                "id": row.get("id"),
                "source": row.get("source"),
                "messages": messages,
            }
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_written += 1
            if max_samples > 0 and rows_written >= max_samples:
                break
    if rows_written == 0:
        raise RuntimeError(f"No reference rows downloaded from {dataset_id} split={split}")
    with (output_path.with_suffix(output_path.suffix + ".meta.json")).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_id": dataset_id,
                "split": split,
                "rows_written": rows_written,
                "max_samples": max_samples,
            },
            handle,
            indent=2,
        )


def ensure_reference_jsonl(args: argparse.Namespace) -> None:
    if args.ref_kl_coeff <= 0.0:
        return
    ref_path = Path(args.ref_train_path)
    if ref_path.exists():
        meta_path = ref_path.with_suffix(ref_path.suffix + ".meta.json")
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            cached_max_samples = int(meta.get("max_samples", 0) or 0)
            cached_rows = int(meta.get("rows_written", 0) or 0)
            requested_rows = int(args.ref_max_samples)
            cache_is_too_small = cached_max_samples > 0 and (
                requested_rows == 0 or cached_rows < requested_rows
            )
            if cache_is_too_small:
                if args.no_ref_download_if_missing:
                    raise RuntimeError(f"Reference cache is partial and too small for this run: {ref_path}")
                tqdm.write(
                    "ref_redownload "
                    f"existing_rows={cached_rows} requested_rows={requested_rows} output={ref_path}"
                )
                download_tulu_reference_jsonl(
                    output_path=ref_path,
                    dataset_id=args.ref_dataset_id,
                    split=args.ref_dataset_split,
                    max_samples=args.ref_max_samples,
                )
        return
    if args.no_ref_download_if_missing:
        raise FileNotFoundError(f"Reference training file is missing: {ref_path}")
    tqdm.write(
        "ref_download "
        f"dataset={args.ref_dataset_id} split={args.ref_dataset_split} "
        f"output={ref_path} max_samples={args.ref_max_samples}"
    )
    download_tulu_reference_jsonl(
        output_path=ref_path,
        dataset_id=args.ref_dataset_id,
        split=args.ref_dataset_split,
        max_samples=args.ref_max_samples,
    )


def prepare_reference_examples(
    *,
    args: argparse.Namespace,
    tokenizer,
) -> tuple[list[base.EncodedExample], int]:
    if args.ref_kl_coeff <= 0.0:
        return [], 0
    ensure_reference_jsonl(args)
    ref_rows = base.load_jsonl_messages(Path(args.ref_train_path))
    if args.ref_max_samples > 0:
        ref_rows = ref_rows[: args.ref_max_samples]
    ref_examples = [
        enc
        for messages in ref_rows
        if (enc := base.encode_messages(tokenizer, messages, max_seq_len=args.max_seq_len)) is not None
    ]
    if not ref_examples:
        raise RuntimeError("No trainable assistant tokens remained in the reference KL set after preprocessing.")
    return ref_examples, len(ref_rows)


def shuffled_reference_order(*, num_examples: int, epoch: int, seed: int) -> list[int]:
    order = list(range(num_examples))
    rng = random.Random(seed + epoch)
    rng.shuffle(order)
    return order


def next_reference_example(
    *,
    ref_examples: list[base.EncodedExample],
    ref_order: list[int],
    ref_cursor: int,
) -> tuple[base.EncodedExample, int]:
    if not ref_examples:
        raise RuntimeError("Reference examples are required for KL mode.")
    index = ref_order[ref_cursor % len(ref_order)]
    return ref_examples[index], ref_cursor + 1


def kl_base_to_lora_on_assistant_tokens(
    *,
    lora_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_labels = labels[:, 1:]
    assistant_mask = shift_labels != base.IGNORE_INDEX
    if not bool(assistant_mask.any()):
        raise RuntimeError("Reference KL batch has no shifted assistant-token labels.")
    lora_token_logits = lora_logits[:, :-1, :][assistant_mask]
    base_token_logits = base_logits[:, :-1, :][assistant_mask]
    base_log_probs = F.log_softmax(base_token_logits.float(), dim=-1)
    lora_log_probs = F.log_softmax(lora_token_logits.float(), dim=-1)
    base_probs = base_log_probs.exp()
    return (base_probs * (base_log_probs - lora_log_probs)).sum(dim=-1).mean()


def compute_reference_kl_loss(
    *,
    model,
    ref_batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    ref_inputs = {key: value for key, value in ref_batch.items() if key != "labels"}
    lora_outputs = model(**ref_inputs)
    with torch.no_grad(), model.disable_adapter():
        base_outputs = model(**ref_inputs)
    return kl_base_to_lora_on_assistant_tokens(
        lora_logits=lora_outputs.logits,
        base_logits=base_outputs.logits,
        labels=ref_batch["labels"],
    )


def save_lora_checkpoint(
    *,
    model,
    tokenizer,
    output_dir: Path,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    total_tokens: int,
    total_label_tokens: int,
    train_start: float,
    final: bool,
) -> Path:
    checkpoint_dir = output_dir if final else output_dir / f"checkpoint-step-{global_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    metrics = {
        "final": final,
        "epoch": epoch,
        "global_step": global_step,
        "total_tokens": total_tokens,
        "total_label_tokens": total_label_tokens,
        "elapsed_train_seconds": time.perf_counter() - train_start,
        "args": vars(args),
    }
    with (checkpoint_dir / "lora_train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return checkpoint_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--allow-model-downloads", action="store_true", help="Allow downloading model files if they are missing from the local Hugging Face cache.")
    ap.add_argument("--train-path", default="train-messages.jsonl")
    ap.add_argument("--output-dir", default="checkpoints/qwen3_4b_pii_lora_r32_1ep")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=None, help="Override computed beta2. Defaults to token-half-life scaling.")
    ap.add_argument("--adam-beta2-reference", type=float, default=0.95, help="Reference beta2 from the large-batch baseline.")
    ap.add_argument("--adam-beta2-reference-batch-size", type=float, default=512.0, help="Reference batch size for beta2 token-half-life scaling.")
    ap.add_argument("--lr-warmup-steps", type=int, default=100)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", nargs="*", default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--pad-to-multiple", type=int, default=8)
    ap.add_argument("--mem-fraction", type=float, default=0.65)
    ap.add_argument("--save-every", type=int, default=2500, help="Save an adapter checkpoint every N optimizer steps. 0 disables intermediate checkpoints.")
    ap.add_argument("--ref-kl-coeff", type=float, default=0.0, help="If > 0, mix in KL(base || lora) on a Tulu reference example each step.")
    ap.add_argument("--ref-train-path", default="data/tulu-3-sft-mixture-train.jsonl")
    ap.add_argument("--ref-dataset-id", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--ref-dataset-split", default="train")
    ap.add_argument("--ref-max-samples", type=int, default=10000, help="0 means all reference rows.")
    ap.add_argument("--ref-shuffle-seed", type=int, default=3407)
    ap.add_argument("--no-ref-download-if-missing", action="store_true")
    ap.add_argument("--wandb-project", default="pii_lora_sft")
    ap.add_argument("--wandb-run-name", default="qwen3_4b_pii_lora_r32_1ep_8192")
    args = ap.parse_args()
    if not 0.0 <= args.ref_kl_coeff <= 1.0:
        raise ValueError(f"ref_kl_coeff must be in [0, 1], got {args.ref_kl_coeff}")
    if args.ref_kl_coeff > 0.0 and args.per_device_batch_size != 1:
        raise ValueError("--ref-kl-coeff requires --per-device-batch-size 1.")

    run_start = time.perf_counter()
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

    rows = base.load_jsonl_messages(Path(args.train_path))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    examples = [
        enc
        for messages in rows
        if (enc := base.encode_messages(tokenizer, messages, max_seq_len=args.max_seq_len)) is not None
    ]
    if not examples:
        raise RuntimeError("No trainable assistant tokens remained after preprocessing.")
    ref_examples, ref_rows_count = prepare_reference_examples(args=args, tokenizer=tokenizer)
    ref_enabled = args.ref_kl_coeff > 0.0

    model.config.use_cache = False
    model.train()
    adam_beta2 = args.adam_beta2
    if adam_beta2 is None:
        adam_beta2 = scaled_adam_beta2(
            reference_beta2=args.adam_beta2_reference,
            reference_batch_size=args.adam_beta2_reference_batch_size,
            batch_size=args.per_device_batch_size,
        )
    if not 0.0 < adam_beta2 < 1.0:
        raise ValueError(f"adam_beta2 must be in (0, 1), got {adam_beta2}")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, adam_beta2),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )

    steps_per_epoch = math.ceil(len(examples) / args.per_device_batch_size)
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < args.lr_warmup_steps:
            return (step + 1) / args.lr_warmup_steps
        progress = (step - args.lr_warmup_steps) / max(total_steps - args.lr_warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    total_param_count, trainable_param_count = count_parameters(model)
    setup_seconds = time.perf_counter() - run_start

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["gb10", "unsloth", "fa2", "lora", "bf16"],
        config={
            **vars(args),
            "dataset_rows": len(rows),
            "encoded_examples": len(examples),
            "model": args.model_path,
            "library": "unsloth_lora",
            "attn": "flash_attention_2",
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "total_param_count": total_param_count,
            "trainable_param_count": trainable_param_count,
            "trainable_fraction": trainable_param_count / total_param_count if total_param_count else 0.0,
            "target_modules": target_modules,
            "optimizer": "AdamW(fused)",
            "ref_kl_enabled": ref_enabled,
            "ref_kl_coeff": args.ref_kl_coeff,
            "ref_kl_direction": "base||lora",
            "ref_dataset_id": args.ref_dataset_id,
            "ref_dataset_split": args.ref_dataset_split,
            "ref_train_path": args.ref_train_path,
            "ref_rows": ref_rows_count,
            "ref_encoded_examples": len(ref_examples),
            "ref_max_samples": args.ref_max_samples,
            "ref_shuffle_seed": args.ref_shuffle_seed,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_beta2_reference": args.adam_beta2_reference,
            "adam_beta2_reference_batch_size": args.adam_beta2_reference_batch_size,
            "adam_beta2_scaled_by_batch_size": args.adam_beta2 is None,
        },
    )
    tqdm.write(
        "train_setup "
        f"rows={len(rows)} encoded={len(examples)} epochs={args.epochs} "
        f"steps={total_steps} trainable={trainable_param_count}/{total_param_count} "
        f"ref_kl={args.ref_kl_coeff:.3f} ref_rows={ref_rows_count} ref_encoded={len(ref_examples)} "
        f"beta2={adam_beta2:.8f} setup_seconds={setup_seconds:.1f}"
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    global_step = 0
    total_tokens = 0
    total_label_tokens = 0
    total_ref_tokens = 0
    total_ref_label_tokens = 0

    progress = tqdm(total=total_steps, desc="LoRA SFT", unit="step", dynamic_ncols=True)
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            epoch_tokens = 0
            epoch_label_tokens = 0
            epoch_ref_tokens = 0
            epoch_ref_label_tokens = 0
            loss_sum = 0.0
            loss_count = 0
            ref_order = shuffled_reference_order(
                num_examples=len(ref_examples),
                epoch=epoch,
                seed=args.ref_shuffle_seed,
            ) if ref_enabled else []
            ref_cursor = 0
            batches = base.build_batches(
                examples,
                per_device_batch_size=args.per_device_batch_size,
                pad_to_multiple=args.pad_to_multiple,
                shuffle_seed=epoch,
                bucket_by_length=True,
            )
            for batch_index, batch_examples in enumerate(batches, start=1):
                batch, stats = base.collate_batch(
                    batch_examples,
                    pad_token_id=pad_token_id,
                    pad_to_multiple=args.pad_to_multiple,
                    device=device,
                )
                outputs = model(**batch)
                ce_loss = outputs.loss
                ref_kl_loss = None
                ref_stats = None
                if ref_enabled:
                    ref_example, ref_cursor = next_reference_example(
                        ref_examples=ref_examples,
                        ref_order=ref_order,
                        ref_cursor=ref_cursor,
                    )
                    ref_batch, ref_stats = base.collate_batch(
                        [ref_example],
                        pad_token_id=pad_token_id,
                        pad_to_multiple=args.pad_to_multiple,
                        device=device,
                    )
                    ref_kl_loss = compute_reference_kl_loss(model=model, ref_batch=ref_batch)
                    loss = (1.0 - args.ref_kl_coeff) * ce_loss + args.ref_kl_coeff * ref_kl_loss
                else:
                    loss = ce_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                epoch_tokens += stats.input_tokens
                epoch_label_tokens += stats.label_tokens
                total_tokens += stats.input_tokens
                total_label_tokens += stats.label_tokens
                if ref_stats is not None:
                    epoch_ref_tokens += ref_stats.input_tokens
                    epoch_ref_label_tokens += ref_stats.label_tokens
                    total_ref_tokens += ref_stats.input_tokens
                    total_ref_label_tokens += ref_stats.label_tokens
                current_loss = float(loss.detach().cpu())
                current_ce_loss = float(ce_loss.detach().cpu())
                current_ref_kl_loss = float(ref_kl_loss.detach().cpu()) if ref_kl_loss is not None else 0.0
                loss_sum += current_loss
                loss_count += 1
                elapsed = time.perf_counter() - train_start
                tokens_per_second = total_tokens / max(elapsed, 1e-9)
                label_tokens_per_second = total_label_tokens / max(elapsed, 1e-9)
                epoch_progress = (epoch - 1) + batch_index / max(len(batches), 1)

                train_log = {
                    "train/loss": current_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch_progress,
                    "train/tokens_per_second": tokens_per_second,
                    "train/label_tokens_per_second": label_tokens_per_second,
                    "train/total_tokens": total_tokens,
                    "train/total_label_tokens": total_label_tokens,
                }
                if ref_enabled:
                    train_log.update(
                        {
                            "train/ce_loss": current_ce_loss,
                            "train/ref_kl_loss": current_ref_kl_loss,
                            "train/ref_kl_coeff": args.ref_kl_coeff,
                            "train/ref_tokens": total_ref_tokens,
                            "train/ref_label_tokens": total_ref_label_tokens,
                        }
                    )
                wandb.log(train_log, step=global_step)
                progress.update(1)
                postfix = {
                    "epoch": f"{epoch_progress:.3f}",
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "tok/s": f"{tokens_per_second:.1f}",
                }
                if ref_enabled:
                    postfix["ce"] = f"{current_ce_loss:.4f}"
                    postfix["kl"] = f"{current_ref_kl_loss:.4f}"
                progress.set_postfix(postfix)

                if args.save_every > 0 and global_step % args.save_every == 0:
                    checkpoint_dir = save_lora_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=Path(args.output_dir),
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                        total_tokens=total_tokens,
                        total_label_tokens=total_label_tokens,
                        train_start=train_start,
                        final=False,
                    )
                    tqdm.write(f"checkpoint_saved epoch={epoch} step={global_step} output_dir={checkpoint_dir}")

            torch.cuda.synchronize()
            tqdm.write(
                f"epoch_summary epoch={epoch} "
                f"seconds={time.perf_counter() - epoch_start:.1f} "
                f"tokens={epoch_tokens} label_tokens={epoch_label_tokens} "
                f"ref_tokens={epoch_ref_tokens} ref_label_tokens={epoch_ref_label_tokens} "
                f"mean_loss={loss_sum / max(loss_count, 1):.4f}"
            )
            wandb.log(
                {
                    "epoch/mean_loss": loss_sum / max(loss_count, 1),
                    "epoch/seconds": time.perf_counter() - epoch_start,
                    "epoch/num": epoch,
                },
                step=global_step,
            )
    finally:
        progress.close()

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    save_start = time.perf_counter()
    out = Path(args.output_dir)
    save_lora_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=out,
        args=args,
        global_step=global_step,
        epoch=args.epochs,
        total_tokens=total_tokens,
        total_label_tokens=total_label_tokens,
        train_start=train_start,
        final=True,
    )
    save_seconds = time.perf_counter() - save_start

    wandb.summary.update(
        {
            "total_wall_seconds": time.perf_counter() - run_start,
            "train_seconds": train_seconds,
            "save_seconds": save_seconds,
            "tokens_per_second": total_tokens / max(train_seconds, 1e-9),
            "label_tokens_per_second": total_label_tokens / max(train_seconds, 1e-9),
            "ref_tokens": total_ref_tokens,
            "ref_label_tokens": total_ref_label_tokens,
            "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "output_dir": str(out),
        }
    )
    wandb.finish()
    tqdm.write(
        "train_summary "
        f"epochs={args.epochs} steps={global_step} total_tokens={total_tokens} "
        f"total_ref_tokens={total_ref_tokens} total_ref_label_tokens={total_ref_label_tokens} "
        f"train_seconds={train_seconds:.1f} save_seconds={save_seconds:.1f} "
        f"tokens_per_second={total_tokens / max(train_seconds, 1e-9):.1f} "
        f"peak_cuda_memory_gb={torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} "
        f"output_dir={out}"
    )


if __name__ == "__main__":
    main()
