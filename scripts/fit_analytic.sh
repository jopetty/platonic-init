#!/usr/bin/env bash
set -euo pipefail

uv run python -m platonic_init.analytic \
  --config "${1:-configs/experiment.yaml}" \
  --subspace artifacts/weight_subspace.pt \
  --out artifacts/analytic_subspace.pt \
  --report-out artifacts/analytic_fit_report.json
