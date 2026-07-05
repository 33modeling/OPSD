# OPSD A100 setup

This fork keeps the upstream OPSD training code unchanged and adds a thin
A100-oriented setup layer for the 33modeling group-volume environment.

Upstream: https://github.com/siyan-zhao/OPSD

## What OPSD uses

Training data is fixed in `opsd_train.py`:

- `siyanzhao/Openthoughts_math_30k_opsd`
- split: `train`
- examples: 29,434
- required fields: `problem`, `solution`

The evaluator supports:

- `aime24` -> `HuggingFaceH4/aime_2024`
- `aime25` -> `yentinglin/aime_2025`
- `hmmt25` -> `MathArena/hmmt_feb_2025`
- `math500` -> `HuggingFaceH4/MATH-500`
- `amc23` -> `math-ai/amc23`
- `minerva` -> `math-ai/minervamath`
- `amo-bench` -> `meituan-longcat/AMO-Bench`

The README-reported main evaluation tables focus on `AIME24`, `AIME25`, and
`HMMT25`.

## One-time setup

Run this on an A100 80G node, preferably inside `tmux`. The setup uses plain
`python -m venv` and `pip`. It does not use `uv` or conda.

```bash
cd /home/kms/dev/OPSD

# Optional. Defaults to /group-volume/<user>/opsd.
export GROUP_VOLUME=/group-volume
source scripts/setup_env_a100.sh 4          # 1, 2, 4, or 8 visible A100s

bash scripts/install_a100.sh
bash scripts/download_models_a100.sh 1.7b   # or: 4b, 8b, all
bash scripts/download_data_a100.sh train eval
bash scripts/preflight_a100.sh
```

`install_a100.sh` creates one venv at `$OPSD_VENV`, installs the upstream pinned
Python stack from `environment.yml`, then installs `flash-attn==2.8.3`.

If your node has no outbound Hugging Face access, pre-populate the HF cache under
`$HF_HOME` or set the usual `HF_ENDPOINT` mirror before running
`download_models_a100.sh` / `download_data_a100.sh`.

## Smoke run

Start with Qwen3-1.7B. This is the fastest way to validate the whole path.

```bash
cd /home/kms/dev/OPSD
source scripts/setup_env_a100.sh

# Short run: verifies model load, vLLM colocate, dataset, and OPSD loss path.
OPSD_MAX_STEPS=2 OPSD_SAVE_STEPS=2 bash scripts/run_opsd_a100.sh 1.7b
```

For a longer run:

```bash
bash scripts/run_opsd_a100.sh 1.7b
```

Model choices:

```bash
bash scripts/run_opsd_a100.sh 1.7b
bash scripts/run_opsd_a100.sh 4b
bash scripts/run_opsd_a100.sh 8b
```

The wrapper auto-detects visible GPUs. Override when needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OPSD_NUM_GPUS=4 bash scripts/run_opsd_a100.sh 4b
```

## Evaluation

```bash
cd /home/kms/dev/OPSD
source scripts/setup_env_a100.sh

bash scripts/eval_a100.sh 1.7b aime24
bash scripts/eval_a100.sh 1.7b aime25 /path/to/checkpoint-100
bash scripts/eval_a100.sh 1.7b hmmt25 /path/to/checkpoint-100
```

Default evaluation uses `val_n=12`, `temperature=1.0`, and thinking mode,
matching the upstream README's main reported setting.

## Important knobs

- `OPSD_WORK`: heavy workspace for cache, logs, outputs.
- `OPSD_VENV`: venv path, default `$OPSD_WORK/venvs/opsd`.
- `OPSD_MODELS_DIR`: model directory, default `$GROUP_VOLUME/nait-models`.
- `MODEL_PATH_QWEN3_17B`, `MODEL_PATH_QWEN3_4B`, `MODEL_PATH_QWEN3_8B`: explicit
  model path overrides.
- `OPSD_NUM_GPUS`: number of accelerate processes.
- `OPSD_MAX_STEPS`: set for smoke tests. Unset for full epoch-based training.
- `OPSD_OUTPUT_ROOT`: checkpoint/output root.
- `WANDB_MODE=disabled`: disable wandb network logging.
- `OPSD_SKIP_FLASH_ATTN=1`: skip flash-attn installation if the node cannot
  build/install it. In that case also override the run with
  `--attn_implementation sdpa`.

The upstream scripts assume fixed paths such as `/data0/shared/Qwen3-1.7B`.
Use the A100 wrappers instead so paths and GPU count are taken from the current
server.
