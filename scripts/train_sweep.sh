#!/usr/bin/env bash
set -euo pipefail

uv run python -m platonic_init.train --config "${1:-configs/experiment.yaml}"
