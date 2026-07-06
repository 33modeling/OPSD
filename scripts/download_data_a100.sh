#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OPSD_SETUP_SCRIPT="${OPSD_SETUP_SCRIPT:-scripts/setup_env_a100.sh}"
OPSD_PROFILE_LABEL="${OPSD_PROFILE_LABEL:-a100}"
source "$OPSD_SETUP_SCRIPT" "${OPSD_NUM_GPUS:-}"
opsd_activate

export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0

if [ "$#" -eq 0 ]; then
  set -- train
fi

python - "$@" <<'PY'
import sys
from datasets import load_dataset

tasks = set(sys.argv[1:])

def show(name, split, **kwargs):
    ds = load_dataset(name, split=split, **kwargs)
    print(f"{name} [{split}] -> {len(ds)} rows")

if "train" in tasks:
    show("siyanzhao/Openthoughts_math_30k_opsd", "train")

if "eval" in tasks:
    show("HuggingFaceH4/aime_2024", "train")
    show("yentinglin/aime_2025", "train", trust_remote_code=True)
    show("MathArena/hmmt_feb_2025", "train", trust_remote_code=True)
    show("HuggingFaceH4/MATH-500", "test")
    show("math-ai/amc23", "test")
    show("math-ai/minervamath", "test")
    show("meituan-longcat/AMO-Bench", "test")
PY

echo "[download_data_${OPSD_PROFILE_LABEL}] HF cache: $HF_HOME"
