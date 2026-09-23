#!/usr/bin/env bash
# L2 on the image tasks: three Qwen3-VL sizes x three split seeds, one model per GPU at a time.
# Every run fits L1 and L2 on the same 200 calibration labels and scores both on the same 300
# test items (bench.mm_l2_study); bench.mm_l2_audit prints the tables in docs/results_multimodal.md.
#
#   bash bench/run_mm_l2.sh
#   GPUS="2 3" SEEDS="0 1 2" CKPT=/path/to/checkpoints bash bench/run_mm_l2.sh
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"                                  # needs transformers >= 4.57 for qwen3_vl
CKPT="${CKPT:-}"                                    # local checkpoint root; empty = load from the hub
MODELS="${MODELS:-2B 4B 8B}"
SEEDS="${SEEDS:-0 1 2}"
BS="${BS:-16}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
LOGS="$REPO/bench/logs_mm"
mkdir -p "$LOGS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"      # image preprocessing is CPU bound
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

i=0
for size in $MODELS; do
  name="Qwen/Qwen3-VL-${size}-Instruct"
  src=(--model "$name")
  [ -n "$CKPT" ] && src+=(--model-path "$CKPT/Qwen3-VL-${size}-Instruct")
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  echo "== $name on GPU $gpu, seeds $SEEDS"
  (
    cd "$REPO" || exit 1
    export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="$gpu"
    for seed in $SEEDS; do
      "$PY" -m bench.mm_l2_study "${src[@]}" --tasks pets20,pope --seed "$seed" --batch-size "$BS" \
        > "$LOGS/l2_${size}_s${seed}.log" 2>&1
    done
  ) &
  i=$((i + 1))
done

wait
echo "done; python -m bench.mm_l2_audit bench/results_mm_l2/<date>"
