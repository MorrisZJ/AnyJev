#!/usr/bin/env bash
# The multimodal bench: three Qwen3-VL sizes, one model per GPU.
#
#   pets20, pope -> bench/results_mm      library default, as the text bench: batch prior
#   pope         -> bench/results_mm_cf   content-free prior as L0, as bench/results_cf is
#                                         for text; the L1 on top of it is part of the story
#   ai2d         -> bench/results_mcq     every item has its own options: bench.run_mcq
#
#   bash bench/run_mm.sh
#   GPUS="0 1" MODELS="2B 8B" bash bench/run_mm.sh
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"                                  # needs transformers >= 4.57 for qwen3_vl
CKPT="${CKPT:-}"                                    # local checkpoint root; empty = load from the hub
N="${N:-300}"
CALIB="${CALIB:-200}"
BS="${BS:-16}"
MODELS="${MODELS:-2B 4B 8B}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
LOGS="$REPO/bench/logs_mm"
mkdir -p "$LOGS"

# These runs are image-preprocessing bound, not GPU bound: the fast image
# processor resizes on CPU and grabs a thread per core, so concurrent models
# oversubscribe the box and spend their time context-switching.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

i=0
for size in $MODELS; do
  name="Qwen/Qwen3-VL-${size}-Instruct"
  src=(--model "$name")
  [ -n "$CKPT" ] && src+=(--model-path "$CKPT/Qwen3-VL-${size}-Instruct")
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  common=(--backend vlm --n "$N" --calib "$CALIB" --batch-size "$BS")
  echo "== $name on GPU $gpu -> $LOGS/${size}.log"
  (
    cd "$REPO" || exit 1
    export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" -m bench.run "${src[@]}" "${common[@]}" --tasks pets20,pope --out bench/results_mm
    "$PY" -m bench.run "${src[@]}" "${common[@]}" --tasks pope --prior content_free --out bench/results_mm_cf
    "$PY" -m bench.run_mcq "${src[@]}" "${common[@]}" --tasks ai2d --out bench/results_mcq
  ) > "$LOGS/${size}.log" 2>&1 &
  i=$((i + 1))
done

wait
echo "done; results in bench/results_mm, bench/results_mm_cf and bench/results_mcq"
