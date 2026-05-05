#!/usr/bin/env bash
set -euo pipefail

# Run the built-in cfy_plugin training pipeline (LPM backbone + CFY plugin modes).
# This script is primarily for understanding how CFY trains/freeze-backbone.
# Intended env: one with perturb_lib installed/available.
#
# Usage:
#   conda activate <env_with_perturb_lib>
#   bash scripts/cfy/run_cfy_training_comparison.sh

cd "$(dirname "$0")/../.."

python3 -u cfy_plugin/training_comparison.py
