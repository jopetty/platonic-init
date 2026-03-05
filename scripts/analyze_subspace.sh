#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/experiment.yaml}"
ROOT="runs/synthetic_prepretrain/gpt2"

uv run python -m platonic_init.analyze \
  --config "$CFG" \
  --checkpoints "$ROOT"/seed_* \
  --out artifacts/weight_subspace.pt \
  --summary-out artifacts/weight_subspace_summary.json
