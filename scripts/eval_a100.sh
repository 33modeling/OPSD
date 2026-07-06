#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OPSD_SETUP_SCRIPT="${OPSD_SETUP_SCRIPT:-scripts/setup_env_a100.sh}"
OPSD_PROFILE_LABEL="${OPSD_PROFILE_LABEL:-a100}"
source "$OPSD_SETUP_SCRIPT" "${OPSD_NUM_GPUS:-}"
opsd_activate

MODEL_SIZE="${1:-1.7b}"
DATASET="${2:-aime24}"
CHECKPOINT_DIR="${3:-}"

case "$MODEL_SIZE" in
  1.7b|1b|17b) BASE_MODEL="$MODEL_PATH_QWEN3_17B" ;;
  4b) BASE_MODEL="$MODEL_PATH_QWEN3_4B" ;;
  8b) BASE_MODEL="$MODEL_PATH_QWEN3_8B" ;;
  *)
    echo "Usage: $0 {1.7b|4b|8b} {aime24|aime25|hmmt25|math500|amc23|minerva|amo-bench} [checkpoint_dir]" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$OPSD_CUDA_VISIBLE_DEVICES}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ARGS=(
  --base_model "$BASE_MODEL"
  --dataset "$DATASET"
  --val_n "${OPSD_EVAL_VAL_N:-12}"
  --temperature "${OPSD_EVAL_TEMPERATURE:-1.0}"
  --tensor_parallel_size "${OPSD_EVAL_TP_SIZE:-$OPSD_NUM_GPUS}"
)

if [ -n "$CHECKPOINT_DIR" ]; then
  ARGS+=(--checkpoint_dir "$CHECKPOINT_DIR")
fi

if [ "${OPSD_EVAL_NO_THINKING:-0}" = "1" ]; then
  ARGS+=(--no_thinking)
fi

python eval/evaluate_math.py "${ARGS[@]}"
