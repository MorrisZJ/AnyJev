#!/usr/bin/env bash
# Qwen3-30B-A3B at the bench's default --batch-size 32, which is what the
# committed results were produced with. The earlier attempt used --batch-size 8
# to be safe on memory; that is both ~4x slower and, per repro_check/probe_batchsize.py,
# guaranteed not to land on the committed numbers bit-exactly.
#
# Sequential, one prior at a time, so the 61 GB of weights never doubles up.
set -u
cd /mnt/persist/project/AnyJev
export HF_HOME=/mnt/persist/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${1:-1}"

echo "### gpu=$CUDA_VISIBLE_DEVICES  $(date -Is)"
for prior in batch content_free; do
  echo "### START Qwen3-30B-A3B prior=$prior bs=32  $(date -Is)"
  python3 -m bench.run \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tasks banking20,newsgroups,injection \
    --n 300 --calib 200 --seed 0 --prior "$prior" --combine logmean --batch-size 32 \
    --out "repro_check/results/bench_${prior}" \
    > "repro_check/logs/P1c.bench.Qwen__Qwen3-30B-A3B-Instruct-2507.${prior}.log" 2>&1
  echo "### DONE  Qwen3-30B-A3B prior=$prior rc=$?  $(date -Is)"
done
echo "### 30B FINISHED $(date -Is)"
