#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps unsloth_zoo==2026.6.2 unsloth==2026.6.2

python - <<'PY'
import torch
import transformers
import flash_attn
import unsloth
import unsloth_zoo

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("unsloth", unsloth.__version__)
print("unsloth_zoo", unsloth_zoo.__version__)
PY
