#!/bin/bash
#SBATCH --job-name=paper-pretrain
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
VENV_PYTHON="${VENV_PYTHON:-/ext3/venvs/platonic-init/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment_tinygpt2_shuffle_dyck_c4_weight_transfer.yaml}"

cd "${REPO_ROOT}"

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    cd '${REPO_ROOT}'
    ${VENV_PYTHON} -m platonic_init.pipeline --config ${CONFIG_PATH} --stages pretrain --skip-fits
  "
