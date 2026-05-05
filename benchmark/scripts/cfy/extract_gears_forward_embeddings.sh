#!/usr/bin/env bash
set -euo pipefail

# Extract per-(pert1,pert2,gene) forward embeddings from GEARS (hook-based).
# Intended to run inside conda env: gears_env2
# Usage:
#   conda activate gears_env2
#   bash scripts/cfy/extract_gears_forward_embeddings.sh norman seed_1_norman_split gears_norman_e0_forward

DATASET_NAME=${1:-norman}
SPLIT_ID=${2:-seed_1_norman_split}
RESULT_ID=${3:-gears_forward_emb}
HOOK=${4:-transform}
EPOCHS=${5:-0}

cd "$(dirname "$0")/../.."

python3 src/extract_forward_embedding_from_gears.py \
  --dataset_name "${DATASET_NAME}" \
  --test_train_config_id "${SPLIT_ID}" \
  --working_dir . \
  --result_id "${RESULT_ID}" \
  --hook "${HOOK}" \
  --epochs "${EPOCHS}"

ls -la "results/${RESULT_ID}" | head
