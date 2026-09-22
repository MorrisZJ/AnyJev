#!/usr/bin/env bash
# Supplementary jobs on one GPU: the two cells my skip-logic bug dropped, plus
# the stateful-batch-prior probe. Sequential so the GPU is never oversubscribed.
set -u
cd /mnt/persist/project/AnyJev
export HF_HOME=/mnt/persist/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${1:-2}"

echo "### using GPU $CUDA_VISIBLE_DEVICES  $(date -Is)"

for prior in batch content_free; do
  echo "### START Qwen2.5-7B-Instruct prior=$prior  $(date -Is)"
  python3 -m bench.run \
    --model Qwen/Qwen2.5-7B-Instruct --tasks banking20,newsgroups,injection \
    --n 300 --calib 200 --seed 0 --prior "$prior" --combine logmean --batch-size 16 \
    --out "repro_check/results/bench_${prior}" \
    > "repro_check/logs/P1b.bench.Qwen__Qwen2.5-7B-Instruct.${prior}.log" 2>&1
  echo "### DONE  Qwen2.5-7B-Instruct prior=$prior rc=$?  $(date -Is)"
done

echo "### START stateful-prior probe  $(date -Is)"
python3 repro_check/probe_stateful_prior.py --model Qwen/Qwen3-8B --task banking20 --batch-size 16 \
  > repro_check/logs/probe_stateful_prior.log 2>&1
echo "### DONE  stateful-prior probe rc=$?  $(date -Is)"

echo "### SUPPLEMENTARY JOBS FINISHED $(date -Is)"
