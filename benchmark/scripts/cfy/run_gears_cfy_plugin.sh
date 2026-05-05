#!/usr/bin/env bash
set -euo pipefail

# Apply CFY post-hoc enhancement on top of a finished GEARS baseline run.
# Intended env: gears_env2
# Usage:
#   conda activate gears_env2
#   bash scripts/cfy/run_gears_cfy_plugin.sh \
#     norman seed_1_norman_split \
#     gears_norman_e20 gears_norman_e20_cfy 5

DATASET_NAME=${1:-norman}
SPLIT_ID=${2:-seed_1_norman_split}
BASE_RESULT_ID=${3:-gears_baseline}
RESULT_ID=${4:-gears_cfy_plugin}
EPOCHS=${5:-5}

cd "$(dirname "$0")/../.."

python3 src/run_gears_cfy_plugin.py \
  --dataset_name "${DATASET_NAME}" \
  --test_train_config_id "${SPLIT_ID}" \
  --working_dir . \
  --base_result_id "${BASE_RESULT_ID}" \
  --result_id "${RESULT_ID}" \
  --epochs "${EPOCHS}" \
  --model_name gears

echo "Wrote: results/${RESULT_ID}/all_predictions.json"
echo "Wrote: results/${RESULT_ID}/baseline_all_predictions.json"
echo "Wrote: results/${RESULT_ID}/cfy_history.json"
