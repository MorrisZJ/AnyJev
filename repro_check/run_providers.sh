#!/usr/bin/env bash
# Rerun the three Laya provider rows through the repo's own provider script,
# on the same 2,000 decisions. Verifies two README claims independently:
#   - laya-typed-decisions "reproduces its published 0.766"
#   - the fine-tuned Laya's ECE is "six times" AnyJev L1's
set -u
cd /mnt/persist/project/AnyJev
export HF_HOME=/mnt/persist/hf-cache
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${1:-2}"

echo "### using GPU $CUDA_VISIBLE_DEVICES  $(date -Is)"
for ck in convaiinnovations/laya-typed-decisions convaiinnovations/laya convaiinnovations/laya-multilingual; do
  slug="${ck//\//__}"
  echo "### START $ck  $(date -Is)"
  python3 -m bench.providers.laya --checkpoint "$ck" --device cuda \
    --out repro_check/results/typed \
    > "repro_check/logs/P3.provider.${slug}.log" 2>&1
  echo "### DONE  $ck rc=$?  $(date -Is)"
done
echo "### PROVIDER ROWS FINISHED $(date -Is)"
