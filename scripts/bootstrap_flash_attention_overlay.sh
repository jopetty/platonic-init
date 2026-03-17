#!/usr/bin/env bash
set -euo pipefail

OVERLAY_DIR="${OVERLAY_DIR:-/scratch/$USER/platonic-init-fa}"
OVERLAY_NAME="${OVERLAY_NAME:-platonic-init-fa.ext3}"
OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz}"
OVERLAY_PATH="${OVERLAY_PATH:-${OVERLAY_DIR}/${OVERLAY_NAME}}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}"
REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
UV_ROOT="${UV_ROOT:-/ext3/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/ext3/uv-python}"
VENV_DIR="${VENV_DIR:-/ext3/venvs/platonic-init}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl}"

mkdir -p "${OVERLAY_DIR}" "${UV_CACHE_DIR}"

if [[ ! -f "${OVERLAY_PATH}" ]]; then
  cp "${OVERLAY_TEMPLATE}" "${OVERLAY_DIR}/"
  gunzip -f "${OVERLAY_DIR}/$(basename "${OVERLAY_TEMPLATE}")"
  mv "${OVERLAY_DIR}/$(basename "${OVERLAY_TEMPLATE%.gz}")" "${OVERLAY_PATH}"
fi

apptainer exec --fakeroot --nv \
  --overlay "${OVERLAY_PATH}:rw" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    export UV_UNMANAGED_INSTALL='${UV_ROOT}'
    export UV_NO_MODIFY_PATH=1
    export UV_PYTHON_INSTALL_DIR='${UV_PYTHON_INSTALL_DIR}'
    export UV_CACHE_DIR='${UV_CACHE_DIR}'
    export UV_LINK_MODE=copy
    mkdir -p '${UV_CACHE_DIR}'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH='${UV_ROOT}':\$PATH
    uv python install --install-dir '${UV_PYTHON_INSTALL_DIR}' '${PYTHON_VERSION}'
    rm -rf '${VENV_DIR}'
    uv venv --python '${PYTHON_VERSION}' '${VENV_DIR}'
    source '${VENV_DIR}/bin/activate'
    uv pip install \
      'torch>=2.4,<2.5' \
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
    uv pip install '${FLASH_ATTN_WHEEL_URL}'
    cd '${REPO_ROOT}'
    uv pip install --no-deps -e .
    python -m platonic_init.check_flash_attention --require-fa2
    mkdir -p /ext3
    cat >/ext3/env.sh <<'SH'
#!/bin/bash
export PATH='${UV_ROOT}':\$PATH
export UV_PYTHON_INSTALL_DIR='${UV_PYTHON_INSTALL_DIR}'
export UV_CACHE_DIR='${UV_CACHE_DIR}'
export UV_LINK_MODE=copy
source '${VENV_DIR}/bin/activate'
SH
    chmod +x /ext3/env.sh
  "
