# OPSD H100 setup

This profile adds H100-oriented wrappers for running OPSD on 1-4 local H100
GPUs. The training logic is shared with the A100 wrappers, but H100 uses its own
work directory, venv, cache, logs, and accelerate config by default.

Upstream: https://github.com/siyan-zhao/OPSD

## Defaults

- `OPSD_WORK`: `$GROUP_VOLUME/<user>/opsd-h100`
- `OPSD_VENV`: `$OPSD_WORK/venvs/opsd`
- `OPSD_ACCELERATE_CONFIG`: `accelerate.h100.yaml`
- `OPSD_MODELS_DIR`: `$GROUP_VOLUME/nait-models`
- supported GPU counts: `1`, `2`, `3`, `4`

If no GPU count is passed, `setup_env_h100.sh` respects an existing
`CUDA_VISIBLE_DEVICES`; otherwise it uses `nvidia-smi` and caps the detected
count at 4.

## One-time setup

Run this on an H100 node, preferably inside `tmux`.

```bash
cd /home/kms/dev/OPSD

# Optional. Defaults to /group-volume.
export GROUP_VOLUME=/group-volume
source scripts/setup_env_h100.sh 4

bash scripts/install_h100.sh
bash scripts/download_models_h100.sh 1.7b   # or: 4b, 8b, all
bash scripts/download_data_h100.sh train eval
bash scripts/preflight_h100.sh
```

## Smoke run

```bash
cd /home/kms/dev/OPSD
source scripts/setup_env_h100.sh 1

OPSD_MAX_STEPS=2 OPSD_SAVE_STEPS=2 bash scripts/run_opsd_h100.sh 1.7b
```

## Training

```bash
bash scripts/run_opsd_h100.sh 1.7b
bash scripts/run_opsd_h100.sh 4b
bash scripts/run_opsd_h100.sh 8b
```

Choose a specific local GPU slice:

```bash
CUDA_VISIBLE_DEVICES=0,1 OPSD_NUM_GPUS=2 bash scripts/run_opsd_h100.sh 4b
```

The default per-model batch settings match the A100 profile. Override them with:

```bash
OPSD_PER_DEVICE_TRAIN_BATCH_SIZE=2 \
OPSD_GRADIENT_ACCUMULATION_STEPS=2 \
bash scripts/run_opsd_h100.sh 8b
```

## Evaluation

```bash
cd /home/kms/dev/OPSD
source scripts/setup_env_h100.sh 4

bash scripts/eval_h100.sh 1.7b aime24
bash scripts/eval_h100.sh 1.7b aime25 /path/to/checkpoint-100
bash scripts/eval_h100.sh 1.7b hmmt25 /path/to/checkpoint-100
```

Default evaluation uses `val_n=12`, `temperature=1.0`, and tensor parallel size
equal to `OPSD_NUM_GPUS`.

## Notes

- H100 uses a separate venv from A100 by default to avoid wheel/cache conflicts.
- Models are still shared through `$OPSD_MODELS_DIR`.
- Set `WANDB_MODE=disabled` to disable WandB network logging.
- Set `OPSD_SKIP_FLASH_ATTN=1` if flash-attn installation is blocked, then run
  with `OPSD_ATTN_IMPLEMENTATION=sdpa`.
