#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
shift || true

case "${MODE}" in
  check)
    TARGET_SCRIPT="./scripts/check_flash_attention.sh"
    ;;
  install)
    TARGET_SCRIPT="./scripts/install_flash_attention.sh"
    ;;
  *)
    echo "Usage: $0 [check|install] [extra srun args...]" >&2
    exit 2
    ;;
esac

ACCOUNT="${ACCOUNT:-torch_pr_287_general}"
PARTITION="${PARTITION:-}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
TIME="${TIME:-01:00:00}"

SRUN_ARGS=(
  --account="${ACCOUNT}"
  --gres="${GRES}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --time="${TIME}"
)

if [[ -n "${PARTITION}" ]]; then
  SRUN_ARGS+=(--partition="${PARTITION}")
fi

exec srun \
  "${SRUN_ARGS[@]}" \
  "$@" \
  "${TARGET_SCRIPT}"
