#!/usr/bin/env bash
# Regenerate every results document from committed JSON except docs/results_adaptive.md and
# docs/results_latency.md, which are hand-assembled and carry their own regeneration command.
# Nothing in the documents below, or in the README tables, is typed by hand; run this after any
# bench run and paste the README blocks.
#   bash scripts/regen_docs.sh bench/results_v01 bench/results_typed_v01 bench/results_typed
set -euo pipefail
BENCH=${1:-bench/results_v01}
TYPED=${2:-bench/results_typed_v01}
TYPED_LAYA=${3:-bench/results_typed}   # holds the Laya provider JSONs, copied alongside our runs
latest() { ls -d "$1"/*/ | tail -1; }

echo "# bench (all models, every level)" 
${PYTHON:-python3} -m bench.table "$(latest "$BENCH")" > docs/results_bench.md
echo "# README headline rows"
${PYTHON:-python3} -m bench.readme_table "$BENCH" --models Qwen3-8B,Qwen2.5-7B-Instruct,Qwen3-30B-A3B-Instruct-2507
echo "# small models"
{ echo "# Small open models, 1.7B to 8B, seven architectures"; echo;
  echo "Same three tasks and the same typed-decisions set as the main tables, same protocol, one H100 per model, bf16. Regenerate with \`bash scripts/regen_docs.sh\`."; echo;
  echo "## One row per model"; echo; ${PYTHON:-python3} -m bench.models_table "$BENCH" "$TYPED"; echo;
  echo "## Every task, every level"; echo; ${PYTHON:-python3} -m bench.table "$(latest "$BENCH")"; echo;
  echo "## typed-decisions"; echo; ${PYTHON:-python3} -m bench.typed_table "$(latest "$TYPED")"; } > docs/results_small_models.md
echo "# typed-decisions (ours + Laya)"
mkdir -p /tmp/typed_merged && rm -f /tmp/typed_merged/* && cp "$(latest "$TYPED")"/*.json /tmp/typed_merged/ && cp "$(latest "$TYPED_LAYA")"/laya__*.json /tmp/typed_merged/ 2>/dev/null || true
${PYTHON:-python3} -m bench.typed_table /tmp/typed_merged > docs/results_typed.md
echo "# maze"
${PYTHON:-python3} -m bench.maze_table "$(latest bench/results_nanojev)" > docs/results_maze.md
echo "# depth, Jev mode, latency, shipped heads"
${PYTHON:-python3} -m bench.exit_table --date "$(basename "$(latest bench/results_exit)")" > docs/results_exit.md
echo "# when L0 helps"
${PYTHON:-python3} -m bench.diag_l0 bench/results_typed_diag bench/results_batchprior_v0 bench/results_small > docs/diag_l0_output.txt   # the input set docs/when_l0_helps.md names
echo "done; README tables: paste the readme_table and models_table outputs into README.md / README.zh-CN.md"
