#!/usr/bin/env bash
# Source this file once per shell before running OPSD on an H100 server:
#   source scripts/setup_env_h100.sh 4
#
# H100 profile:
# - code checkout stays under /home/kms/dev
# - heavy artifacts live under /group-volume by default
# - one plain venv under group-volume; no uv, no conda requirement
# - supports 1, 2, 3, or 4 visible H100s
# - setup warns about missing assets but never aborts the shell

if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _opsd_setup_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  _opsd_setup_dir="$(pwd)"
fi

export OPSD_REPO_ROOT="${OPSD_REPO_ROOT:-$_opsd_setup_dir}"
export OPSD_PROFILE_LABEL="${OPSD_PROFILE_LABEL:-h100}"

export GROUP_VOLUME="${GROUP_VOLUME:-/group-volume}"
export OPSD_USER="${OPSD_USER:-${USER:-$(whoami)}}"

if [ -z "${OPSD_WORK:-}" ]; then
  export OPSD_WORK="${GROUP_VOLUME%/}/${OPSD_USER}/opsd-h100"
fi

export OPSD_OUTPUT_ROOT="${OPSD_OUTPUT_ROOT:-$OPSD_WORK/outputs}"
export OPSD_LOG_ROOT="${OPSD_LOG_ROOT:-$OPSD_WORK/logs}"
export OPSD_DATA_ROOT="${OPSD_DATA_ROOT:-$OPSD_WORK/data}"
export OPSD_VENV="${OPSD_VENV:-$OPSD_WORK/venvs/opsd}"
export OPSD_ACCELERATE_CONFIG="${OPSD_ACCELERATE_CONFIG:-$OPSD_REPO_ROOT/accelerate.h100.yaml}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$OPSD_WORK/cache/pip}"

export HF_HOME="${HF_HOME:-$OPSD_WORK/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"

export OPSD_TRAIN_DATASET="${OPSD_TRAIN_DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"

# Shared model mirror. download_models_h100.sh writes here unless overridden.
export OPSD_MODELS_DIR="${OPSD_MODELS_DIR:-${GROUP_VOLUME%/}/nait-models}"
export HFID_QWEN3_17B="${HFID_QWEN3_17B:-Qwen/Qwen3-1.7B}"
export HFID_QWEN3_4B="${HFID_QWEN3_4B:-Qwen/Qwen3-4B}"
export HFID_QWEN3_8B="${HFID_QWEN3_8B:-Qwen/Qwen3-8B}"

_opsd_first_existing() {
  for p in "$@"; do
    if [ -e "$p" ]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

export MODEL_PATH_QWEN3_17B="${MODEL_PATH_QWEN3_17B:-$(_opsd_first_existing \
  "$OPSD_MODELS_DIR/Qwen3-1.7B" \
  "$OPSD_MODELS_DIR/qwen3-1.7b" \
  "$OPSD_MODELS_DIR/Qwen/Qwen3-1.7B" \
  2>/dev/null || printf '%s' "$OPSD_MODELS_DIR/Qwen3-1.7B")}"

export MODEL_PATH_QWEN3_4B="${MODEL_PATH_QWEN3_4B:-$(_opsd_first_existing \
  "$OPSD_MODELS_DIR/Qwen3-4B" \
  "$OPSD_MODELS_DIR/qwen3-4b" \
  "$OPSD_MODELS_DIR/Qwen/Qwen3-4B" \
  2>/dev/null || printf '%s' "$OPSD_MODELS_DIR/Qwen3-4B")}"

export MODEL_PATH_QWEN3_8B="${MODEL_PATH_QWEN3_8B:-$(_opsd_first_existing \
  "$OPSD_MODELS_DIR/Qwen3-8B" \
  "$OPSD_MODELS_DIR/qwen3-8b" \
  "$OPSD_MODELS_DIR/Qwen/Qwen3-8B" \
  2>/dev/null || printf '%s' "$OPSD_MODELS_DIR/Qwen3-8B")}"

_opsd_count_cuda_visible() {
  python3 - "$1" <<'PY'
import sys
value = sys.argv[1].strip()
if not value or value in {"-1", "NoDevFiles"}:
    print(0)
else:
    print(len([part for part in value.split(",") if part.strip()]))
PY
}

_opsd_requested_gpus="${1:-${OPSD_NUM_GPUS:-}}"
if [ -n "$_opsd_requested_gpus" ]; then
  export OPSD_NUM_GPUS="$_opsd_requested_gpus"
else
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _opsd_detected_gpus="$(_opsd_count_cuda_visible "$CUDA_VISIBLE_DEVICES")"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    _opsd_detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    _opsd_detected_gpus=1
  fi
  if [ -z "$_opsd_detected_gpus" ] || [ "$_opsd_detected_gpus" = "0" ]; then
    _opsd_detected_gpus=1
  fi
  if [ "$_opsd_detected_gpus" -gt 4 ]; then
    _opsd_detected_gpus=4
  fi
  export OPSD_NUM_GPUS="$_opsd_detected_gpus"
fi

case "$OPSD_NUM_GPUS" in
  1|2|3|4) ;;
  *) echo "[setup_env_h100] WARN: unusual OPSD_NUM_GPUS=$OPSD_NUM_GPUS; expected 1,2,3,4" >&2 ;;
esac

_opsd_gpu_list() {
  python3 - "$1" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n)))
PY
}

if [ -z "${OPSD_CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ -z "$_opsd_requested_gpus" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export OPSD_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  else
    export OPSD_CUDA_VISIBLE_DEVICES="$(_opsd_gpu_list "$OPSD_NUM_GPUS")"
  fi
fi

if [ -z "${CUDA_HOME:-}" ]; then
  for _c in /usr/local/cuda /usr/local/cuda-12.[0-9] /usr/local/cuda-12.[0-9][0-9]; do
    if [ -x "$_c/bin/nvcc" ]; then
      export CUDA_HOME="$_c"
      break
    fi
  done
  if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
fi
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# Download scripts flip these to 0 in their subprocess. Training should use
# the local model directories and group-volume HF cache.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Keep huge core files off the small user volume.
ulimit -c 0 2>/dev/null || true

mkdir -p "$OPSD_WORK" "$OPSD_OUTPUT_ROOT" "$OPSD_LOG_ROOT" "$OPSD_DATA_ROOT" \
  "$OPSD_MODELS_DIR" "$PIP_CACHE_DIR" \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  2>/dev/null || true

opsd_activate() {
  if [ ! -f "$OPSD_VENV/bin/activate" ]; then
    echo "[opsd_activate] venv not found at $OPSD_VENV" >&2
    echo "[opsd_activate] create it: bash scripts/install_h100.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$OPSD_VENV/bin/activate"
  export PYTHONPATH="$OPSD_REPO_ROOT:${PYTHONPATH:-}"
  echo "[opsd_activate] active: $(command -v python)"
}
if [ -n "${BASH_VERSION:-}" ]; then export -f opsd_activate; fi

opsd_print_env() {
  echo "OPSD_REPO_ROOT=$OPSD_REPO_ROOT"
  echo "OPSD_WORK=$OPSD_WORK"
  echo "OPSD_VENV=$OPSD_VENV"
  echo "OPSD_MODELS_DIR=$OPSD_MODELS_DIR"
  echo "OPSD_OUTPUT_ROOT=$OPSD_OUTPUT_ROOT"
  echo "HF_HOME=$HF_HOME"
  echo "OPSD_NUM_GPUS=$OPSD_NUM_GPUS"
  echo "OPSD_CUDA_VISIBLE_DEVICES=$OPSD_CUDA_VISIBLE_DEVICES"
  echo "MODEL_PATH_QWEN3_17B=$MODEL_PATH_QWEN3_17B"
  echo "MODEL_PATH_QWEN3_4B=$MODEL_PATH_QWEN3_4B"
  echo "MODEL_PATH_QWEN3_8B=$MODEL_PATH_QWEN3_8B"
}

_opsd_missing=0
_opsd_warn() {
  local var="$1" path="$2" fix="$3"
  if [ ! -e "$path" ]; then
    if [ "$_opsd_missing" = "0" ]; then
      echo ""
      echo "------------------------------------------------------------------"
      echo "[setup_env_h100] WARNINGS: the following paths do not exist yet."
      echo "------------------------------------------------------------------"
    fi
    printf "  [missing] %-24s %s\n" "$var" "$path"
    printf "            fix:  %s\n" "$fix"
    _opsd_missing=$((_opsd_missing + 1))
  fi
}

_opsd_warn GROUP_VOLUME "$GROUP_VOLUME" "mount group-volume, or export GROUP_VOLUME=/your/large/disk"
_opsd_warn OPSD_VENV "$OPSD_VENV" "bash scripts/install_h100.sh"
_opsd_warn MODEL_PATH_QWEN3_17B "$MODEL_PATH_QWEN3_17B" "bash scripts/download_models_h100.sh 1.7b"

if [ "$_opsd_missing" -gt 0 ]; then
  echo "------------------------------------------------------------------"
  echo "[setup_env_h100] $_opsd_missing path(s) missing; env vars are still exported."
  echo "------------------------------------------------------------------"
else
  echo "[setup_env_h100] All required paths verified"
fi

echo ""
echo "OPSD H100 env loaded."
opsd_print_env

unset -f _opsd_warn
unset _opsd_missing _opsd_requested_gpus _opsd_detected_gpus _opsd_default_visible
