#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export OPSD_SETUP_SCRIPT="${OPSD_SETUP_SCRIPT:-scripts/setup_env_h100.sh}"
export OPSD_PROFILE_LABEL="${OPSD_PROFILE_LABEL:-h100}"
exec scripts/download_data_a100.sh "$@"
