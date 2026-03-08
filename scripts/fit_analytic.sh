#!/usr/bin/env bash
set -euo pipefail

uv run python -m platonic_init.analytic \
  --config "${1:-configs/experiment.yaml}" \
  --fit-name "${2:-chebyshev}"
