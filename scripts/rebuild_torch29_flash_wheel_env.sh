#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
VENV_DIR="${VENV_DIR:-/ext3/venvs/platonic-init}"
VENV_PYTHON="${VENV_PYTHON:-${VENV_DIR}/bin/python}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}"
TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
SINGULARITY_FAKEROOT="${SINGULARITY_FAKEROOT:-0}"
TORCH_SPEC="${TORCH_SPEC:-torch>=2.9,<2.10}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl}"

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
    mkdir -p '${UV_CACHE_DIR}' '${TMPDIR}'
    cd '${REPO_ROOT}'
    rm -rf '${VENV_DIR}'
    uv venv --python 3.10 '${VENV_DIR}'
    uv pip install --python '${VENV_PYTHON}' setuptools wheel ninja packaging
    uv pip install --python '${VENV_PYTHON}' \
      '${TORCH_SPEC}' \
      'transformers>=4.46.0' \
      'trl>=0.11.0' \
      'datasets>=2.20.0' \
      'accelerate>=0.34.0' \
      'safetensors>=0.4.5' \
      'numpy>=1.26.0' \
      'scipy>=1.12.0' \
      'scikit-learn>=1.5.0' \
      'tqdm>=4.66.0' \
      'pyyaml>=6.0.2' \
      'wandb>=0.19.0' \
      'python-dotenv>=1.0.1'
    uv pip install --python '${VENV_PYTHON}' '${FLASH_ATTN_WHEEL_URL}'
    uv pip install --python '${VENV_PYTHON}' --no-deps -e .
    '${VENV_PYTHON}' -m platonic_init.check_flash_attention --require-fa2
  "
