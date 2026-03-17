#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
VENV_PYTHON="${VENV_PYTHON:-/ext3/venvs/platonic-init/bin/python}"
SINGULARITY_FAKEROOT="${SINGULARITY_FAKEROOT:-0}"

SINGULARITY_ARGS=(exec --nv --overlay "${OVERLAY}:ro")
if [[ "${SINGULARITY_FAKEROOT}" == "1" ]]; then
  SINGULARITY_ARGS=(exec --fakeroot --nv --overlay "${OVERLAY}:ro")
fi

exec singularity "${SINGULARITY_ARGS[@]}" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    cd '${REPO_ROOT}'
    '${VENV_PYTHON}' -m platonic_init.check_flash_attention --require-fa2
  "
