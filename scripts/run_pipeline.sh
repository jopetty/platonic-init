#!/usr/bin/env bash
set -euo pipefail

uv run python -m platonic_init.pipeline --config "${1:-configs/experiment.yaml}"
