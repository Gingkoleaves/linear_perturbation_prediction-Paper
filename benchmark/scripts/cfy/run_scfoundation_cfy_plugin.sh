#!/usr/bin/env bash
set -euo pipefail

# Apply CFY post-hoc enhancement on top of a finished scFoundation baseline run.
# Intended env: scfoundation_env
# Usage:
#   conda activate scfoundation_env
#   bash scripts/cfy/run_scfoundation_cfy_plugin.sh \
#     norman_from_scfoundation seed_1_norman_from_scfoundation_split \
#     scf_norman_e15 scf_norman_e15_cfy 5

DATASET_NAME=${1:-norman_from_scfoundation}
SPLIT_ID=${2:-seed_1_norman_from_scfoundation_split}
BASE_RESULT_ID=${3:-scfoundation_baseline}
RESULT_ID=${4:-scfoundation_cfy_plugin}
EPOCHS=${5:-5}

cd "$(dirname "$0")/../.."

python3 src/run_scfoundation_cfy_plugin.py \
  --dataset_name "${DATASET_NAME}" \
  --test_train_config_id "${SPLIT_ID}" \
  --working_dir . \
  --base_result_id "${BASE_RESULT_ID}" \
  --result_id "${RESULT_ID}" \
  --epochs "${EPOCHS}" \
  --model_name scfoundation

echo "Wrote: results/${RESULT_ID}/all_predictions.json"
echo "Wrote: results/${RESULT_ID}/baseline_all_predictions.json"
echo "Wrote: results/${RESULT_ID}/cfy_history.json"
