#!/bin/bash
# Fine-tune Llama-3-8B (or any causal LM) on a SuperGLUE task with GRZO or a
# baseline ZO optimizer. Runs data-parallel over NPROC GPUs of one node via
# torchrun; NPROC=1 uses the single-GPU code path.
#
#   export MODEL_PATH=/path/to/Meta-Llama-3-8B      # local dir or HF id (+ HF_TOKEN)
#   bash scripts/run_llama.sh                       # GRZO on RTE, 4 GPUs
#   TASK=BoolQ OPTIMIZER=grzo_lozo LR=1e-7 bash scripts/run_llama.sh
#
# Every setting below can be overridden from the environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?set MODEL_PATH to a local model directory or a HF model id}"

TASK="${TASK:-RTE}"                 # SST2 RTE CB BoolQ WSC WIC MultiRC Copa ReCoRD SQuAD DROP
OPTIMIZER="${OPTIMIZER:-flipout}"   # flipout (=GRZO) | grzo_lozo | grzo_lozo_strict | grzo_sparse | grzo_quzo
                                    # | mezo | mezo_lozo | mezo_sparse | mezo_quzo | fzoo
LR="${LR:-1e-7}"
SEED="${SEED:-42}"
NPROC="${NPROC:-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"   # global batch = NPROC * PER_DEVICE_BATCH (16 in the paper)
MAX_STEPS="${MAX_STEPS:-20000}"
EVAL_STEPS="${EVAL_STEPS:-4000}"
EPS="${EPS:-1e-3}"                          # perturbation scale sigma
EST_SIDE="${EST_SIDE:-two_norm}"            # two_norm = group-relative advantage; two = raw loss difference
U_DIST="${U_DIST:-gaussian}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_DEV="${NUM_DEV:-500}"
NUM_EVAL="${NUM_EVAL:-1000}"
REPORT_TO="${REPORT_TO:-none}"              # set to wandb to log
EXTRA="${EXTRA:-}"                          # any additional run.py flags
case "$TASK" in CB|Copa) NUM_DEV=100 ;; esac

TAG="${TAG:-${TASK}-${OPTIMIZER}-lr${LR}-seed${SEED}}"
OUT_DIR="${OUT_DIR:-$ROOT/results/$TAG}"

# Optimizer-specific flags (defaults match the paper).
case "$OPTIMIZER" in
    grzo_sparse|mezo_sparse)
        EXTRA="$EXTRA --sparse_ratio ${SPARSE_RATIO:-0.25} --sparse_rule ${SPARSE_RULE:-small}" ;;
    grzo_lozo|grzo_lozo_strict|mezo_lozo)
        EXTRA="$EXTRA --lozo_rank ${LOZO_RANK:-8} --lozo_step_interval ${LOZO_STEP_INTERVAL:-50}" ;;
    grzo_quzo|mezo_quzo)
        EXTRA="$EXTRA --quant_bits ${QUANT_BITS:-4}"
        [[ "${QUZO_WBITS:-0}" -gt 0 ]] && EXTRA="$EXTRA --quzo_weight_bits $QUZO_WBITS" ;;
    fzoo)
        EXTRA="$EXTRA --fzoo_n ${FZOO_N:-8}" ;;
esac

# Multiple-choice / generation tasks are trained with LM loss, not as classification.
TRAIN_AS_CLF="--train_as_classification"
case "$TASK" in Copa|ReCoRD|SQuAD|DROP) TRAIN_AS_CLF="--train_as_classification False" ;; esac

export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_DIR"
echo "== $TAG  model=$MODEL_PATH  gpus=$NPROC  batch=$((NPROC * PER_DEVICE_BATCH))  steps=$MAX_STEPS  out=$OUT_DIR"

cd "$ROOT/MeZO/large_models"
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" run.py \
    --model_name "$MODEL_PATH" \
    --task_name "$TASK" \
    --output_dir "$OUT_DIR" --overwrite_output_dir \
    --num_train "$NUM_TRAIN" --num_dev "$NUM_DEV" --num_eval "$NUM_EVAL" \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$EVAL_STEPS" --save_steps "$EVAL_STEPS" --save_total_limit 1 \
    --eval_strategy steps --save_strategy steps \
    --trainer zo --zo_optimizer "$OPTIMIZER" \
    --learning_rate "$LR" --lr_scheduler_type constant \
    --zo_eps "$EPS" --zo_group_size 1 \
    --u_distribution "$U_DIST" --estimation_side "$EST_SIDE" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --load_float16 \
    $TRAIN_AS_CLF \
    --remove_unused_columns False \
    --train_set_seed "$SEED" \
    --logging_steps 10 \
    --report_to "$REPORT_TO" --run_name "$TAG" \
    $EXTRA
