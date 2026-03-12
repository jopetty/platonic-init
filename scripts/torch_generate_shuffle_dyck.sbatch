#!/bin/bash
#SBATCH --job-name=shuffle-dyck
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/$USER/platonic-init}"
OVERLAY="${OVERLAY:-/scratch/$USER/uv-env/uv-python.ext3}"
IMAGE="${IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
VENV_PYTHON="${VENV_PYTHON:-/ext3/venvs/platonic-init/bin/python}"

cd "${REPO_ROOT}"

singularity exec \
  --overlay "${OVERLAY}:ro" \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    cd '${REPO_ROOT}'
    ${VENV_PYTHON} scripts/generate_formal_language.py --language shuffle_dyck --k 64 --n-samples 20000 --max-depth 10 --output data/shuffle_dyck_k64_d10_20k.txt
  "
