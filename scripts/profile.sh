#!/bin/bash
# Profile per-step wall-clock and peak GPU memory of each ZO optimizer.
#
# Each method runs a short training job on RTE (5 warm-up + 20 measured steps,
# global batch 16, fp16). With GRZO_PROFILE_OUT set, the trainer times every
# stage of each ZO step (torch.cuda.synchronize + perf_counter) and records the
# per-stage peak GPU memory, then writes one JSON per rank at exit.
#
#   export MODEL_PATH=/path/to/Meta-Llama-3-8B
#   bash scripts/profile.sh
#   python scripts/summarize_profile.py profiling
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT="${OUT:-$ROOT/profiling}"
METHODS="${METHODS:-mezo mezo_lozo mezo_sparse mezo_quzo fzoo flipout grzo_lozo_strict grzo_sparse grzo_quzo}"
mkdir -p "$OUT"

for m in $METHODS; do
    echo "==== profiling $m ===="
    GRZO_PROFILE_OUT="$OUT/$m.json" GRZO_PROFILE_WARMUP="${WARMUP:-5}" \
    TASK=RTE OPTIMIZER="$m" MAX_STEPS="${STEPS:-25}" EVAL_STEPS=100000 NUM_EVAL=50 \
    OUT_DIR="$OUT/run-$m" EXTRA="--do_eval False ${EXTRA:-}" \
    bash "$ROOT/scripts/run_llama.sh"
done

python "$ROOT/scripts/summarize_profile.py" "$OUT"
