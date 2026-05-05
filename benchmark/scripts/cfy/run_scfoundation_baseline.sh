#!/usr/bin/env bash
set -euo pipefail

# Train scFoundation baseline and write predictions json.
# Intended to run inside conda env: scfoundation_env
# Usage:
#   conda activate scfoundation_env
#   bash scripts/cfy/run_scfoundation_baseline.sh norman_from_scfoundation seed_1_norman_from_scfoundation_split scf_norman_e15
#
# Note: norman_from_scfoundation can be very slow to load due to large data_pyg.

DATASET_NAME=${1:-norman_from_scfoundation}
SPLIT_ID=${2:-seed_1_norman_from_scfoundation_split}
RESULT_ID=${3:-scfoundation_baseline}
EPOCHS=${4:-15}

cd "$(dirname "$0")/../.."

python3 src/run_scfoundation.py \
  --dataset_name "${DATASET_NAME}" \
  --test_train_config_id "${SPLIT_ID}" \
  --working_dir . \
  --result_id "${RESULT_ID}" \
  --epochs "${EPOCHS}"

echo "Wrote: results/${RESULT_ID}/all_predictions.json"
echo "Wrote: results/${RESULT_ID}/gene_names.json"
