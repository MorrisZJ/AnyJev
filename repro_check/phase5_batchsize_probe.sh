#!/usr/bin/env bash
# Why does Qwen3-32B's typed run deviate by up to 0.005 while Qwen3-8B's is exact?
#
# Hypothesis: batch size. The readout reads a next-token distribution, and the
# last bits of a logit depend on how prompts were grouped into batches
# (cuBLAS reduction order). My 32B run used --batch-size 8; bench.run_typed
# defaults to 32. At 2,000 decisions a handful of near-tie argmaxes flip.
#
# Test: rerun the same model and data at batch-size 32 and 16 and see whether
# the deviation tracks the batch size rather than the model.
#
# Waits for a GPU with enough free memory before starting.
set -u
cd /mnt/persist/project/AnyJev
export HF_HOME=/mnt/persist/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NEED_MIB="${NEED_MIB:-80000}"

wait_for_gpu () {
  while true; do
    for g in 0 1 2 3; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
      free=$(( total - used ))
      if [ "$free" -ge "$NEED_MIB" ]; then echo "$g"; return; fi
    done
    sleep 60
  done
}

for bs in 32 16; do
  gpu=$(wait_for_gpu)
  echo "### START typed Qwen3-32B batch-size=$bs on gpu $gpu  $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -m bench.run_typed \
    --model Qwen/Qwen3-32B --levels raw,L0,L1 --calib-cases 50 --prior batch \
    --batch-size "$bs" --out "repro_check/results/typed_bs${bs}" \
    > "repro_check/logs/P5.typed.Qwen__Qwen3-32B.bs${bs}.log" 2>&1
  echo "### DONE  typed Qwen3-32B batch-size=$bs rc=$?  $(date -Is)"
done

echo "### BATCH-SIZE PROBE FINISHED $(date -Is)"
