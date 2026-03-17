#!/usr/bin/env bash
set -euo pipefail

OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
VENV_DIR="${VENV_DIR:-/ext3/venvs/platonic-init}"
SINGULARITY_FAKEROOT="${SINGULARITY_FAKEROOT:-0}"

SINGULARITY_ARGS=(exec --nv --overlay "${OVERLAY}:rw")
if [[ "${SINGULARITY_FAKEROOT}" == "1" ]]; then
  SINGULARITY_ARGS=(exec --fakeroot --nv --overlay "${OVERLAY}:rw")
fi

exec singularity "${SINGULARITY_ARGS[@]}" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    rm -rf '${VENV_DIR}'
  "
