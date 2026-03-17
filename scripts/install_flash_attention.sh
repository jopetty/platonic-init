#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}"
TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"

exec singularity exec --nv \
  --overlay "${OVERLAY}:rw" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export UV_CACHE_DIR='${UV_CACHE_DIR}'
    export TMPDIR='${TMPDIR}'
    mkdir -p '${UV_CACHE_DIR}' '${TMPDIR}'
    cd '${REPO_ROOT}'
    uv run pip uninstall -y flash-attn || true
    uv run pip install --no-build-isolation --no-cache-dir 'flash-attn==${FLASH_ATTN_VERSION}'
    uv run python -m platonic_init.check_flash_attention --require-fa2
  "
