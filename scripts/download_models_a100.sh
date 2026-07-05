#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env_a100.sh "${OPSD_NUM_GPUS:-}"
opsd_activate

WHAT="${1:-1.7b}"

export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0

_fetch() {
  local repo="$1" dest="$2"
  if [ -f "$dest/config.json" ]; then
    echo "[skip] $repo already present at $dest"
    return 0
  fi
  echo "[fetch] $repo -> $dest"
  mkdir -p "$dest"
  huggingface-cli download "$repo" --local-dir "$dest" \
    --exclude "*.msgpack" "original/*" || \
    hf download "$repo" --local-dir "$dest" --exclude "*.msgpack" --exclude "original/*"
  test -f "$dest/config.json"
  echo "[done] $repo"
}

case "$WHAT" in
  1.7b|1b|17b) _fetch "$HFID_QWEN3_17B" "$MODEL_PATH_QWEN3_17B" ;;
  4b)          _fetch "$HFID_QWEN3_4B"  "$MODEL_PATH_QWEN3_4B" ;;
  8b)          _fetch "$HFID_QWEN3_8B"  "$MODEL_PATH_QWEN3_8B" ;;
  all)
    _fetch "$HFID_QWEN3_17B" "$MODEL_PATH_QWEN3_17B"
    _fetch "$HFID_QWEN3_4B"  "$MODEL_PATH_QWEN3_4B"
    _fetch "$HFID_QWEN3_8B"  "$MODEL_PATH_QWEN3_8B"
    ;;
  *) echo "usage: bash scripts/download_models_a100.sh {1.7b|4b|8b|all}" >&2; exit 2 ;;
esac

echo "[download_models_a100] done: $WHAT"
