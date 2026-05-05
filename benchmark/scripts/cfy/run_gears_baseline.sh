#!/usr/bin/env bash
set -euo pipefail

# Train GEARS baseline and write predictions json.
# Intended to run inside conda env: gears_env2
# Usage:
#   conda activate gears_env2
#   bash scripts/cfy/run_gears_baseline.sh norman seed_1_norman_split gears_norman_e20

DATASET_NAME=${1:-norman}
SPLIT_ID=${2:-seed_1_norman_split}
RESULT_ID=${3:-gears_baseline}
EPOCHS=${4:-20}

cd "$(dirname "$0")/../.."

python3 src/run_gears.py \
  --dataset_name "${DATASET_NAME}" \
  --test_train_config_id "${SPLIT_ID}" \
  --working_dir . \
  --result_id "${RESULT_ID}" \
  --epochs "${EPOCHS}"

echo "Wrote: results/${RESULT_ID}/all_predictions.json"
echo "Wrote: results/${RESULT_ID}/gene_names.json"
