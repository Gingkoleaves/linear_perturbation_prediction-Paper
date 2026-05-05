#!/usr/bin/env bash
set -euo pipefail

# Train scGPT baseline and write predictions json.
# Intended to run inside conda env: scgpt_env
# Usage:
#   conda activate scgpt_env
#   bash scripts/cfy/run_scgpt_baseline.sh norman seed_1_norman_split scgpt_norman_e15

DATASET_NAME=${1:-norman}
SPLIT_ID=${2:-seed_1_norman_split}
RESULT_ID=${3:-scgpt_baseline}
EPOCHS=${4:-15}

cd "$(dirname "$0")/../.."

python3 src/run_scgpt.py \
  --dataset_name "${DATASET_NAME}" \
  --test_train_config_id "${SPLIT_ID}" \
  --working_dir . \
  --result_id "${RESULT_ID}" \
  --epochs "${EPOCHS}"

echo "Wrote: results/${RESULT_ID}/all_predictions.json"
echo "Wrote: results/${RESULT_ID}/gene_names.json"
