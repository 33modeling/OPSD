#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OPSD_SETUP_SCRIPT="${OPSD_SETUP_SCRIPT:-scripts/setup_env_a100.sh}"
OPSD_PROFILE_LABEL="${OPSD_PROFILE_LABEL:-a100}"
source "$OPSD_SETUP_SCRIPT" "${OPSD_NUM_GPUS:-}"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "[install_${OPSD_PROFILE_LABEL}] python3 not found" >&2
  exit 1
}

echo "[install_${OPSD_PROFILE_LABEL}] using $($PYTHON_BIN --version 2>&1) at $(command -v "$PYTHON_BIN")"

if [ ! -f "$OPSD_VENV/bin/activate" ]; then
  echo "[install_${OPSD_PROFILE_LABEL}] creating venv: $OPSD_VENV"
  "$PYTHON_BIN" -m venv "$OPSD_VENV"
else
  echo "[install_${OPSD_PROFILE_LABEL}] reusing venv: $OPSD_VENV"
fi

# shellcheck disable=SC1090
source "$OPSD_VENV/bin/activate"

export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export PYTHONPATH="$OPSD_REPO_ROOT:${PYTHONPATH:-}"

python -m pip install --upgrade pip wheel setuptools packaging ninja

# Keep the upstream versions explicit. Override OPSD_TORCH_SPEC if the node's
# NVIDIA driver cannot run the default torch wheel.
OPSD_TORCH_SPEC="${OPSD_TORCH_SPEC:-torch==2.8.0}"
echo "[install_${OPSD_PROFILE_LABEL}] installing OPSD stack into venv"
python -m pip install \
  "$OPSD_TORCH_SPEC" \
  accelerate==1.11.0 \
  transformers==4.57.1 \
  trl==0.26.0 \
  datasets==3.6.0 \
  deepspeed==0.18.2 \
  peft==0.17.1 \
  bitsandbytes==0.48.2 \
  wandb==0.22.3 \
  vllm==0.11.0 \
  xformers==0.0.32.post1 \
  triton==3.4.0 \
  einops==0.8.1 \
  safetensors==0.5.3 \
  sentencepiece==0.1.99 \
  tiktoken==0.9.0 \
  math-verify==0.8.0 \
  "huggingface_hub[cli]"

if [ "${OPSD_SKIP_FLASH_ATTN:-0}" != "1" ]; then
  echo "[install_${OPSD_PROFILE_LABEL}] installing flash-attn==2.8.3"
  python -m pip install flash-attn==2.8.3 --no-build-isolation
else
  echo "[install_${OPSD_PROFILE_LABEL}] skipping flash-attn because OPSD_SKIP_FLASH_ATTN=1"
fi

python - <<'PY'
import importlib
mods = ["torch", "accelerate", "transformers", "trl", "datasets", "vllm", "deepspeed", "peft"]
for mod in mods:
    m = importlib.import_module(mod)
    print(f"{mod}: {getattr(m, '__version__', 'unknown')}")
import torch
print("torch.cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
PY

echo "[install_${OPSD_PROFILE_LABEL}] done"
