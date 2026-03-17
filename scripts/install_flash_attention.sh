#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
VENV_PYTHON="${VENV_PYTHON:-/ext3/venvs/platonic-init/bin/python}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}"
TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
SINGULARITY_FAKEROOT="${SINGULARITY_FAKEROOT:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-1}"

SINGULARITY_ARGS=(exec --nv --overlay "${OVERLAY}:rw")
if [[ "${SINGULARITY_FAKEROOT}" == "1" ]]; then
  SINGULARITY_ARGS=(exec --fakeroot --nv --overlay "${OVERLAY}:rw")
fi

exec singularity "${SINGULARITY_ARGS[@]}" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export UV_CACHE_DIR='${UV_CACHE_DIR}'
    export TMPDIR='${TMPDIR}'
    export MAX_JOBS='${MAX_JOBS}'
    export NVCC_THREADS='${NVCC_THREADS}'
    mkdir -p '${UV_CACHE_DIR}' '${TMPDIR}'
    cd '${REPO_ROOT}'
    '${VENV_PYTHON}' -m pip uninstall -y flash-attn || true
    '${VENV_PYTHON}' -m pip install -v --no-build-isolation --no-cache-dir 'flash-attn==${FLASH_ATTN_VERSION}'
    '${VENV_PYTHON}' -m platonic_init.check_flash_attention --require-fa2
  "
