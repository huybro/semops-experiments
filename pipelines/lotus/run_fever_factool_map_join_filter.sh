#!/usr/bin/env bash
set -euo pipefail

CONDA_PREFIX="${CONDA_PREFIX:-/scratch/hojaeson_umass/miniforge3/envs/rtx}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python "$(dirname "$0")/fever_factool_map_join_filter.py" "$@"
