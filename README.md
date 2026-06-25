# PII fine-tuning tutorial on DGX Spark

(AI Writing Disclosure: This README assisted by the wonderful GPT-5.5)

This folder is a small tutorial workspace for supervised fine-tuning on DGX Spark with Unsloth. The config here isn't necessarily the optimal one, but it worked for me, and runs a bf16 LoRA on a Qwen3-4B-Instruct-2507 at 2000 tokens/sec on a single DGX Spark.

This folder demonstrates a classic SFT setup - you have a model (Qwen3-4B-Instruct-2507), and you want to inject a capability (PII extraction) into it while leaving the base model intact. This tutorial shows how to do that locally w/ a LoRA adapter on a DGX Spark.

## 1. Create a clean conda training environment

```bash
conda create -n trainEnv python=3.12 pip -y
conda activate trainEnv
python -m pip install --upgrade pip
```

Use `python -m pip`, not bare `pip`, so installs definitely go into the conda environment.

## 2. Install the pinned stack

From this folder:

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file installs PyTorch from the CUDA 13 PyTorch index and installs FlashAttention from a wheel built specially for the Spark.

## 3. Install Unsloth without dependency resolution

```bash
python -m pip install --no-deps unsloth_zoo==2026.6.2 unsloth==2026.6.2
```

The `--no-deps` flag is intentional. The working DGX Spark stack uses `torch==2.12.0+cu130`, while current Unsloth package metadata may declare an older Torch upper bound. Installing Unsloth without dependencies preserves the working CUDA/PyTorch stack.


You can also simply run `./install.sh` to install the pinned stack and Unsloth in one step.


## 4. Train the LoRA adapter

Default LoRA SFT uses the PII chat-message training file and AdamW:

```bash
conda activate trainEnv
python unsloth_lora_train.py
```

It should train at roughly 2000 tokens/sec on a single DGX Spark, and log to WandB. This will teach Qwen3-4B-Instruct-2507 to extract PII from chat messages, but may degrade the base model's existing capabilities when the adapter is enabled.

To preserve capabilities, you can KL-regularize the PII training - meaning minimizing the KL divergence between the base model and the LoRA adapter on a reference dataset (Tulu 3 SFT mixture). This is roughly 3x slower, but as we will see in the benchmark results, it preserves base model capability.

For KL-regularized training against the Tulu 3 SFT mixture:

```bash
python unsloth_lora_train.py \
  --ref-kl-coeff 0.1 \
  --output-dir checkpoints/qwen3_4b_pii_lora_r32_kl_tulu10k_1ep \
  --wandb-run-name qwen3_4b_pii_lora_r32_kl_tulu10k_1ep
```

This uses:

```text
loss = 0.9 * PII_CE + 0.1 * KL(base || lora)
```

The KL teacher is the same model with the LoRA adapter disabled, and teacher logits are computed with gradients off.

The first KL run needs Hugging Face network access and caches 10,000 Tulu examples by default at:

```text
data/tulu-3-sft-mixture-train.jsonl
```

The cache also gets a metadata sidecar:

```text
data/tulu-3-sft-mixture-train.jsonl.meta.json
```

KL mode currently requires `--per-device-batch-size 1`, so each optimizer step pairs one PII CE example with one Tulu reference KL example.

## 6. Create a separate inference environment for SGLang evals

Keep inference separate from `trainEnv`. SGLang currently uses a different Torch and Transformers stack than the Unsloth training environment, so installing it into `trainEnv` can uninstall or downgrade training-critical packages.

Create a dedicated environment:

```bash
conda create -n inferenceEnv python=3.12 pip -y
conda activate inferenceEnv
python -m pip install --upgrade pip
python -m pip install "sglang[all]==0.5.13.post1"
```

Known working inference stack on DGX Spark:

```text
Python 3.12
sglang 0.5.13.post1
torch 2.11.0
transformers 5.8.1
sglang-kernel 0.4.3
flashinfer-python 0.6.12
flashinfer-cubin 0.6.12
```

The eval scripts default to `--mem-fraction-static 0.40`, `--max-running-requests 128`, and `--batch-size 0`. `--batch-size 0` submits all prompts at once and lets SGLang schedule batching internally, while `--max-running-requests` limits SGLang active request concurrency.

## 7. Run PII extraction eval

From this folder:

```bash
conda activate inferenceEnv
python sglang_pii_eval.py
```

By default this evaluates the base model with no LoRA adapter. To evaluate the trained PII LoRA adapter, pass `--lora`:

```bash
python sglang_pii_eval.py --lora
```

The default LoRA path is:

```text
checkpoints/qwen3_4b_pii_lora_r32_1ep
```

To evaluate a different adapter - for instance, the KL-regularized one:

```bash
python sglang_pii_eval.py --lora --lora-path checkpoints/qwen3_4b_pii_lora_r32_kl_tulu10k_1ep
```

Outputs:

```text
eval_outputs/sglang_pii_eval.jsonl
eval_outputs/sglang_pii_eval.summary.json
```

LoRA-enabled runs automatically suffix the default output filename with the LoRA directory name, for example:

```text
eval_outputs/sglang_pii_eval-qwen3_4b_pii_lora_r32_1ep.jsonl
```

The terminal prints a compact scoreboard:

```text
eval_score: parsed_micro_f1 * fraction_parsed
parsed_micro_f1: micro-F1 over examples whose outputs parsed as JSON
fraction_parsed: fraction of outputs that parsed as JSON
prompt_tokens_per_second: input token throughput
tokens_generated_per_second: generated token throughput
```

## 8. Run GSM8K eval

The GSM8K evaluator downloads the Hugging Face test split on first use and caches it at:

```text
data/gsm8k-test.jsonl
```

Run the base model:

```bash
conda activate inferenceEnv
python sglang_gsm8k_eval.py --repetition-penalty 1.1
```

Run with the default LoRA adapter:

```bash
python sglang_gsm8k_eval.py --lora --repetition-penalty 1.1
```

The prompt is the GSM8K question verbatim. The scorer extracts the model answer from the last `\boxed{...}` expression; if no boxed answer exists, it falls back to the last number in the output.

Outputs:

```text
eval_outputs/sglang_gsm8k_eval.jsonl
eval_outputs/sglang_gsm8k_eval.summary.json
```

Similar to above, you can evaluate distinct adapters via:

```bash
python sglang_gsm8k_eval.py --lora --lora-path checkpoints/qwen3_4b_pii_lora_r32_kl_tulu10k_1ep --repetition-penalty 1.1
```

The terminal prints:

```text
gsm8k_accuracy: exact numeric accuracy
correct: correct / total
fraction_extracted: fraction of outputs with an extracted numeric answer
prompt_tokens_per_second: input token throughput
tokens_generated_per_second: generated token throughput
```

## Benchmark Results: Base vs Normal LoRA vs KL-Regularized LoRA

These runs were completed w/ SGLang. All runs used `--temperature 0`; GSM8K additionally used `--repetition-penalty 1.1`.

| Model | PII micro-F1 | GSM8K accuracy |
| --- | ---: | ---: |
| base | `0.026830` | `0.918878` (`1212/1319`) |
| normal LoRA | `0.955160` | `0.887794` (`1171/1319`) |
| KL LoRA | `0.955119` | `0.916603` (`1209/1319`) |

As we see, the KL-regularized LoRA adapter preserves GSM8K accuracy while achieving nearly the same-ish PII micro-F1 as the normal LoRA adapter, at the cost of 3x slower training.

# Conclusion
This repo forms a basic recipe to finetune a small model on a single DGX Spark while preserving its base capability. It's meant as a template to be modified, or a reference for agents to use.