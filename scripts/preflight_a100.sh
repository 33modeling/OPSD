#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OPSD_SETUP_SCRIPT="${OPSD_SETUP_SCRIPT:-scripts/setup_env_a100.sh}"
OPSD_PROFILE_LABEL="${OPSD_PROFILE_LABEL:-a100}"
source "$OPSD_SETUP_SCRIPT" "${OPSD_NUM_GPUS:-}"
opsd_activate

echo "[preflight_${OPSD_PROFILE_LABEL}] nvidia-smi"
nvidia-smi || true

python - <<'PY'
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())

model_path = os.environ["MODEL_PATH_QWEN3_17B"]
cfg = Path(model_path) / "config.json"
print("model_path", model_path, "config_exists", cfg.exists())
if cfg.exists():
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("tokenizer", type(tok).__name__, "pad", tok.pad_token is not None)

os.environ["HF_DATASETS_OFFLINE"] = "0"
ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train[:2]")
print("train_dataset_sample_rows", len(ds), "columns", ds.column_names)
PY

echo "[preflight_${OPSD_PROFILE_LABEL}] done"
