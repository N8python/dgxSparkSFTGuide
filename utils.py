"""Shared data, batching, and throughput helpers for DGX Spark SFT scripts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


IGNORE_INDEX = -100


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]

    @property
    def input_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def label_tokens(self) -> int:
        return sum(label != IGNORE_INDEX for label in self.labels)


@dataclass
class BatchStats:
    input_tokens: int
    label_tokens: int
    padded_input_tokens: int
    attention_s2: int


def round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def load_jsonl_messages(path: Path) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number} has no list-valued messages field")
            rows.append(messages)
    return rows


def apply_chat(tokenizer: AutoTokenizer, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
            enable_thinking=False,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    pieces: list[str] = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        pieces.append("<|im_start|>assistant\n")
    return tokenizer("".join(pieces), add_special_tokens=False).input_ids


def encode_messages(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    *,
    max_seq_len: int,
) -> EncodedExample | None:
    input_ids = apply_chat(tokenizer, messages, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(input_ids)

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        span_start = len(apply_chat(tokenizer, messages[:index], add_generation_prompt=True))
        span_end = len(apply_chat(tokenizer, messages[: index + 1], add_generation_prompt=False))
        span_start = min(span_start, len(input_ids))
        span_end = min(span_end, len(input_ids))
        for pos in range(span_start, span_end):
            labels[pos] = input_ids[pos]

    if max_seq_len > 0 and len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return EncodedExample(input_ids=input_ids, labels=labels)


def build_batches(
    examples: list[EncodedExample],
    *,
    per_device_batch_size: int,
    pad_to_multiple: int,
    shuffle_seed: int,
    bucket_by_length: bool,
) -> list[list[EncodedExample]]:
    rng = random.Random(shuffle_seed)
    ordered = list(examples)
    if bucket_by_length:
        ordered.sort(key=lambda example: example.input_tokens)
        buckets = [
            ordered[start : start + max(per_device_batch_size * 32, per_device_batch_size)]
            for start in range(0, len(ordered), max(per_device_batch_size * 32, per_device_batch_size))
        ]
        for bucket in buckets:
            rng.shuffle(bucket)
        ordered = [example for bucket in buckets for example in bucket]
    else:
        rng.shuffle(ordered)

    batches = [
        ordered[start : start + per_device_batch_size]
        for start in range(0, len(ordered), per_device_batch_size)
    ]
    rng.shuffle(batches)
    return batches


def collate_batch(
    examples: list[EncodedExample],
    *,
    pad_token_id: int,
    pad_to_multiple: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], BatchStats]:
    max_len = round_up(max(example.input_tokens for example in examples), pad_to_multiple)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    input_tokens = 0
    label_tokens = 0
    for row, example in enumerate(examples):
        length = example.input_tokens
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
        attention_mask[row, :length] = 1
        input_tokens += length
        label_tokens += example.label_tokens

    stats = BatchStats(
        input_tokens=input_tokens,
        label_tokens=label_tokens,
        padded_input_tokens=batch_size * max_len,
        attention_s2=batch_size * max_len * max_len,
    )
    return {
        "input_ids": input_ids.to(device=device, non_blocking=True),
        "labels": labels.to(device=device, non_blocking=True),
        "attention_mask": attention_mask.to(device=device, non_blocking=True),
    }, stats


def flops_for_window(
    *,
    param_count: int,
    num_layers: int,
    hidden_size: int,
    padded_input_tokens: int,
    attention_s2: int,
    include_checkpoint_recompute: bool,
) -> dict[str, float]:
    dense_flops = 6.0 * param_count * padded_input_tokens
    attention_flops = 12.0 * num_layers * hidden_size * attention_s2
    if include_checkpoint_recompute:
        dense_flops *= 4.0 / 3.0
        attention_flops *= 4.0 / 3.0
    total_flops = dense_flops + attention_flops
    return {
        "dense_flops": dense_flops,
        "attention_flops": attention_flops,
        "total_flops": total_flops,
        "attention_fraction": attention_flops / total_flops if total_flops else 0.0,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
