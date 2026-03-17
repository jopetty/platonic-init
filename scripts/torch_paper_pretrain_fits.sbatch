#!/bin/bash
#SBATCH --job-name=paper-pretrain-fits
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
VENV_PYTHON="${VENV_PYTHON:-/ext3/venvs/platonic-init/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/gpt2_tiny_c4_ppt_reproduction.yaml}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
FIT_NAMES="${FIT_NAMES:-}"

cd "${REPO_ROOT}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" logs

FIT_NAMES_ARGS=""
if [[ -n "${FIT_NAMES}" ]]; then
  FIT_NAMES_ARGS="--fit-names ${FIT_NAMES}"
fi

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export HF_HOME='${HF_HOME}'
    export HUGGINGFACE_HUB_CACHE='${HF_HUB_CACHE}'
    export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'
    cd '${REPO_ROOT}'
    ${VENV_PYTHON} -m platonic_init.pipeline \
      --config ${CONFIG_PATH} \
      --stages fit_initializations pretrain \
      --skip-random \
      --skip-transfer \
      ${FIT_NAMES_ARGS}
  "
