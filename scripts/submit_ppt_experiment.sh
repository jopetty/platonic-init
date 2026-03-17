#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage: scripts/submit_ppt_experiment.sh <config-path> <stage> [sbatch args...]

Stages:
  prepare-data
  prepretrain
  pretrain
  pretrain-fits
EOF
  exit 1
fi

CONFIG_PATH="$1"
STAGE="$2"
shift 2

case "$STAGE" in
  prepare-data)
    SBATCH_SCRIPT="scripts/torch_paper_prepare_data.sbatch"
    ;;
  prepretrain)
    SBATCH_SCRIPT="scripts/torch_paper_prepretrain.sbatch"
    ;;
  pretrain)
    SBATCH_SCRIPT="scripts/torch_paper_pretrain.sbatch"
    ;;
  pretrain-fits)
    SBATCH_SCRIPT="scripts/torch_paper_pretrain_fits.sbatch"
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 1
    ;;
esac

echo "Submitting ${SBATCH_SCRIPT} with ${CONFIG_PATH}"
sbatch --export=CONFIG_PATH="${CONFIG_PATH}" "$@" "${SBATCH_SCRIPT}"
