#!/usr/bin/env bash
set -euo pipefail

# Repairs a partially missing Miniforge install by re-extracting already cached
# .conda package payloads. This script intentionally does not remove envs,
# lib directories, or package caches.

PREFIX="${PREFIX:-/scratch/hojaeson_umass/miniforge3}"
RTX_PREFIX="${RTX_PREFIX:-$PREFIX/envs/rtx}"
PKGS_DIR="${PKGS_DIR:-$PREFIX/pkgs}"
BASE_PY_VER="${BASE_PY_VER:-3.13}"
RTX_PY_VER="${RTX_PY_VER:-3.12}"

BASE_SITE="$PREFIX/lib/python$BASE_PY_VER"
PLACEHOLDER="/home/conda/feedstock_root/build_artifacts/conda_1777457670951/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_place"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

extract_pkg_to_prefix() {
  local spec="$1"
  local dest="$2"
  local archive="$PKGS_DIR/$spec.conda"

  require_file "$archive"
  echo "restoring $spec -> $dest"
  unzip -p "$archive" "pkg-$spec.tar.zst" | zstd -d | tar -x -C "$dest"
}

extract_noarch_to_base_python() {
  local spec="$1"
  local archive="$PKGS_DIR/$spec.conda"

  require_file "$archive"
  echo "restoring noarch $spec -> $BASE_SITE"
  mkdir -p "$BASE_SITE"
  unzip -p "$archive" "pkg-$spec.tar.zst" | zstd -d | tar -x -C "$BASE_SITE" site-packages
}

patch_conda_launchers() {
  local files=(
    "$PREFIX/etc/profile.d/conda.sh"
    "$PREFIX/bin/conda"
    "$PREFIX/condabin/conda"
  )

  for file in "${files[@]}"; do
    [[ -f "$file" ]] || continue
    sed -i "s|$PLACEHOLDER|$PREFIX|g" "$file"
  done
}

main() {
  require_file "$PKGS_DIR/python-3.13.13-h6add32d_100_cp313.conda"
  require_file "$PKGS_DIR/python-3.12.13-hd63d673_0_cpython.conda"

  echo "repairing base Miniforge at $PREFIX"
  extract_pkg_to_prefix "python-3.13.13-h6add32d_100_cp313" "$PREFIX"
  extract_pkg_to_prefix "conda-26.3.2-py313h78bf25f_1" "$PREFIX"
  extract_pkg_to_prefix "ruamel.yaml-0.18.17-py313h54dd161_2" "$PREFIX"
  extract_pkg_to_prefix "frozendict-2.4.7-py313h07c4f96_0" "$PREFIX"
  extract_pkg_to_prefix "backports.zstd-1.3.0-py313h18e8e13_0" "$PREFIX"
  extract_pkg_to_prefix "fmt-12.1.0-hff5e90c_0" "$PREFIX"
  extract_pkg_to_prefix "yaml-cpp-0.8.0-h3f2d84a_0" "$PREFIX"
  extract_pkg_to_prefix "libmsgpack-c-6.1.0-h54a6638_6" "$PREFIX"
  extract_pkg_to_prefix "bzip2-1.0.8-hda65f42_9" "$PREFIX"
  extract_pkg_to_prefix "reproc-14.2.7.post0-hb03c661_0" "$PREFIX"
  extract_pkg_to_prefix "reproc-cpp-14.2.7.post0-hecca717_0" "$PREFIX"
  extract_pkg_to_prefix "libssh2-1.11.1-hcf80075_0" "$PREFIX"
  extract_pkg_to_prefix "libstdcxx-15.2.0-h934c35e_19" "$PREFIX"
  extract_pkg_to_prefix "libgcc-ng-15.2.0-h69a702a_19" "$PREFIX"
  extract_pkg_to_prefix "libstdcxx-ng-15.2.0-hdf11a46_19" "$PREFIX"
  extract_pkg_to_prefix "libnghttp2-1.68.1-h877daf1_0" "$PREFIX"
  extract_pkg_to_prefix "libsqlite-3.53.1-h0c1763c_0" "$PREFIX"
  extract_pkg_to_prefix "sqlite-3.53.1-hbc0de68_0" "$PREFIX"

  extract_noarch_to_base_python "pip-26.0.1-pyh145f28c_0"
  extract_noarch_to_base_python "urllib3-2.6.3-pyhd8ed1ab_0"
  extract_noarch_to_base_python "boltons-25.0.0-pyhd8ed1ab_0"
  extract_noarch_to_base_python "charset-normalizer-3.4.7-pyhd8ed1ab_0"
  extract_noarch_to_base_python "pluggy-1.6.0-pyhf9edf01_1"
  extract_noarch_to_base_python "platformdirs-4.9.6-pyhcf101f3_0"
  extract_noarch_to_base_python "tqdm-4.67.3-pyh8f84b5b_0"

  echo "repairing rtx Python at $RTX_PREFIX"
  extract_pkg_to_prefix "python-3.12.13-hd63d673_0_cpython" "$RTX_PREFIX"
  extract_pkg_to_prefix "bzip2-1.0.8-hda65f42_9" "$RTX_PREFIX"
  extract_pkg_to_prefix "openssl-3.6.2-h35e630c_0" "$RTX_PREFIX"
  extract_pkg_to_prefix "libfaiss-1.14.1-h1b31e9c_0_cuda12.6" "$RTX_PREFIX"
  extract_pkg_to_prefix "faiss-gpu-1.14.1-py3.12_h1b31e9c_0_cuda12.6" "$RTX_PREFIX"
  extract_pkg_to_prefix "mkl-2025.3.0-h0e700b2_463" "$RTX_PREFIX"
  extract_pkg_to_prefix "cuda-cudart_linux-64-12.6.77-h3f2d84a_0" "$RTX_PREFIX"
  extract_pkg_to_prefix "cuda-cudart-12.6.77-h5888daf_0" "$RTX_PREFIX"
  extract_pkg_to_prefix "libcublas-12.6.4.1-h5888daf_1" "$RTX_PREFIX"

  echo "patching conda launcher paths"
  patch_conda_launchers

  echo "verifying base conda"
  "$PREFIX/bin/python" --version
  "$PREFIX/bin/python" -m pip --version
  "$PREFIX/bin/conda" --version

  echo "verifying rtx activation"
  # shellcheck disable=SC1091
  source "$PREFIX/etc/profile.d/conda.sh"
  conda activate "$RTX_PREFIX"
  which python
  python --version
  python -c 'import sys; print(sys.prefix); import torch, pandas; print(torch.__version__); print(pandas.__version__)'

  echo "repair complete"
}

main "$@"
