#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/submit_ppt_experiment.sh" \
  "configs/gpt2_tiny_c4_ppt_reproduction_muon.yaml" \
  "$@"
