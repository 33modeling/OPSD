#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OPSD_SETUP_SCRIPT="${OPSD_SETUP_SCRIPT:-scripts/setup_env_a100.sh}"
OPSD_PROFILE_LABEL="${OPSD_PROFILE_LABEL:-a100}"
source "$OPSD_SETUP_SCRIPT" "${OPSD_NUM_GPUS:-}"
opsd_activate

MODEL_SIZE="${1:-1.7b}"
shift || true

case "$MODEL_SIZE" in
  1.7b|1b|17b)
    MODEL_PATH="$MODEL_PATH_QWEN3_17B"
    DEFAULT_BS=4
    DEFAULT_GA=2
    DEFAULT_CLIP=0.05
    MODEL_TAG=qwen3_1_7b
    ;;
  4b)
    MODEL_PATH="$MODEL_PATH_QWEN3_4B"
    DEFAULT_BS=2
    DEFAULT_GA=2
    DEFAULT_CLIP=0.05
    MODEL_TAG=qwen3_4b
    ;;
  8b)
    MODEL_PATH="$MODEL_PATH_QWEN3_8B"
    DEFAULT_BS=1
    DEFAULT_GA=4
    DEFAULT_CLIP=0.06
    MODEL_TAG=qwen3_8b
    ;;
  *)
    echo "Usage: $0 {1.7b|4b|8b} [extra opsd_train.py args...]" >&2
    exit 1
    ;;
esac

PER_DEVICE_BS="${OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_BS}"
GRAD_ACCUM="${OPSD_GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GA}"
JSD_CLIP="${OPSD_JSD_TOKEN_CLIP:-$DEFAULT_CLIP}"
MAX_COMPLETION="${OPSD_MAX_COMPLETION_LENGTH:-1024}"
MAX_LENGTH="${OPSD_MAX_LENGTH:-20000}"
NUM_EPOCHS="${OPSD_NUM_TRAIN_EPOCHS:-30}"
SAVE_STEPS="${OPSD_SAVE_STEPS:-25}"
LOGGING_STEPS="${OPSD_LOGGING_STEPS:-2}"
MASTER_PORT="${MASTER_PORT:-12949}"
RUN_CONFIG="${OPSD_RUN_CONFIG:-${MODEL_TAG}_${OPSD_PROFILE_LABEL}_g${OPSD_NUM_GPUS}_gen${MAX_COMPLETION}_opsd}"

EXTRA_ARGS=()
if [ -n "${OPSD_MAX_STEPS:-}" ]; then
  EXTRA_ARGS+=(--max_steps "$OPSD_MAX_STEPS")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$OPSD_CUDA_VISIBLE_DEVICES}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[run_opsd_${OPSD_PROFILE_LABEL}] model=$MODEL_PATH"
echo "[run_opsd_${OPSD_PROFILE_LABEL}] gpus=$CUDA_VISIBLE_DEVICES processes=$OPSD_NUM_GPUS bs=$PER_DEVICE_BS grad_accum=$GRAD_ACCUM"
echo "[run_opsd_${OPSD_PROFILE_LABEL}] output=$OPSD_OUTPUT_ROOT run_config=$RUN_CONFIG"

accelerate launch \
  --config_file "$OPSD_ACCELERATE_CONFIG" \
  --num_processes "$OPSD_NUM_GPUS" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --main_process_port "$MASTER_PORT" \
  opsd_train.py \
  --model_name_or_path "$MODEL_PATH" \
  --learning_rate "${OPSD_LEARNING_RATE:-5e-6}" \
  --max_grad_norm "${OPSD_MAX_GRAD_NORM:-0.1}" \
  --per_device_train_batch_size "$PER_DEVICE_BS" \
  --gradient_checkpointing \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --output_dir "$OPSD_OUTPUT_ROOT" \
  --run_config "$RUN_CONFIG" \
  --num_train_epochs "$NUM_EPOCHS" \
  --max_completion_length "$MAX_COMPLETION" \
  --save_steps "$SAVE_STEPS" \
  --logging_steps "$LOGGING_STEPS" \
  --attn_implementation "${OPSD_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --torch_dtype bfloat16 \
  --max_length "$MAX_LENGTH" \
  --beta "${OPSD_BETA:-0}" \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.6}" \
  --vllm_tensor_parallel_size "${OPSD_VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --use_peft \
  --lora_r "${OPSD_LORA_R:-64}" \
  --lora_alpha "${OPSD_LORA_ALPHA:-128}" \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature "${OPSD_TEMPERATURE:-1.1}" \
  --top_p "${OPSD_TOP_P:-0.95}" \
  --top_k "${OPSD_TOP_K:-20}" \
  --lmbda "${OPSD_LMBDA:-1}" \
  --fixed_teacher \
  --jsd_token_clip "$JSD_CLIP" \
  --wandb_project "${WANDB_PROJECT:-OPSD}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
