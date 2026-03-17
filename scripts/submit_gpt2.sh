#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/submit_ppt_experiment.sh" \
  "configs/gpt2_c4_ppt_reproduction.yaml" \
  "$@"
