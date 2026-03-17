#!/bin/bash
#SBATCH --job-name=paper-prepare-data
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}"
VENV_PYTHON="${VENV_PYTHON:-/scratch/$USER/venvs/platonic-init/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/gpt2_tiny_c4_ppt_reproduction.yaml}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

cd "${REPO_ROOT}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

OVERLAY_ARGS=()
if [[ -n "${OVERLAY}" && -f "${OVERLAY}" ]]; then
  OVERLAY_ARGS=(--overlay "${OVERLAY}:ro")
fi

singularity exec \
  "${OVERLAY_ARGS[@]}" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    export HF_HOME='${HF_HOME}'
    export HUGGINGFACE_HUB_CACHE='${HF_HUB_CACHE}'
    export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'
    cd '${REPO_ROOT}'
    ${VENV_PYTHON} scripts/cache_init_eval_data.py --config ${CONFIG_PATH}
  "
