#!/usr/bin/env bash
# Reproduce every committed row in bench/results_nanojev, using NanoJev's own
# frozen exploration harness and their released episode data.
#
#   usage: bash repro_check/run_maze.sh <gpu>
set -u
cd /mnt/persist/project/AnyJev
export HF_HOME=/mnt/persist/hf-cache
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${1:-0}"

NANOJEV=/mnt/persist/project/AnyJev/repro_check/external/NanoJev
EPISODES=$(python3 - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("C-Tianyu/NanoJev-Data",
                      "games_v4/data/scaled_games_v4b/episodes.jsonl",
                      repo_type="dataset"))
PY
)
echo "### gpu=$CUDA_VISIBLE_DEVICES  episodes=$EPISODES  $(date -Is)"
OUT=repro_check/results/nanojev

# NanoJev pins Qwen3-0.6B at this commit and refuses any other cached snapshot;
# the committed AnyJev rows use the same pin, so the comparison is apples to apples.
PIN=c1899de289a04d12100db370d81485cdf75e47ca

run_anyjev () {  # model level prior [revision]
  local model="$1" level="$2" prior="$3" rev="${4:-}"
  local slug="${model//\//__}.${level}.${prior}"
  echo "### START anyjev $slug  $(date -Is)"
  local revarg=()
  [ -n "$rev" ] && revarg=(--revision "$rev")
  python3 -m bench.providers.nanojev_maze \
    --nanojev "$NANOJEV" --episodes "$EPISODES" --splits test,ood \
    --model "$model" --level "$level" --prior "$prior" "${revarg[@]}" \
    --window-size 5 --max-steps 0 --batch-states 8 \
    --out "$OUT" > "repro_check/logs/P3.maze.${slug}.log" 2>&1
  echo "### DONE  anyjev $slug rc=$?  $(date -Is)"
}

# the four Qwen3-0.6B rows (pinned revision) and the two Qwen3-8B rows (no pin)
run_anyjev Qwen/Qwen3-0.6B raw batch        "$PIN"
run_anyjev Qwen/Qwen3-0.6B L0  batch        "$PIN"
run_anyjev Qwen/Qwen3-0.6B L0  content_free "$PIN"
run_anyjev Qwen/Qwen3-0.6B L0  none         "$PIN"
run_anyjev Qwen/Qwen3-8B   raw batch
run_anyjev Qwen/Qwen3-8B   L0  batch

# NanoJev's own "Untuned Qwen" A/B readout, as reimplemented by the repo.
# It takes no --model: it loads NanoJev's own pinned snapshot offline.
echo "### START native A/B readout  $(date -Is)"
python3 -m bench.providers.nanojev_native_maze \
  --nanojev "$NANOJEV" --episodes "$EPISODES" --splits test,ood \
  --window-size 5 --max-steps 0 --batch-states 8 \
  --out "$OUT" > repro_check/logs/P3.maze.native.Qwen3-0.6B.log 2>&1
echo "### DONE  native A/B readout rc=$?  $(date -Is)"

echo "### MAZE ROWS FINISHED $(date -Is)"
