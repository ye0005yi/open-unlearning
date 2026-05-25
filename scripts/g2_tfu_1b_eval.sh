#!/bin/bash
###############################################################################
# G2 1B TFU Sweep for MUSE — Full w × method × threshold grid
#
# Models:
#   - News: saves/finetune/V7_muse_news_full_1epoch (exists)
#   - Books: saves/finetune/G2_muse_books_1b_full (created by g2_finetune_1b.sh)
#
# Covers:
#   - Naive: w=0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0
#   - Static: w={1.5,2.0,2.5,3.0,4.0,5.0} × th={0.55,0.65,0.75,0.85}
#   - Similarity: same as Static
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 SPLIT=News bash scripts/g2_tfu_1b_eval.sh
#   CUDA_VISIBLE_DEVICES=1 SPLIT=Books bash scripts/g2_tfu_1b_eval.sh
###############################################################################
set -o pipefail
source /path/to/workdir/hf_setup.sh

WORK_DIR="/path/to/workdir/open-unlearning"
PYTHON="/path/to/workdir/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/path/to/workdir/miniconda/envs/tfu"
export HF_HOME="/path/to/workdir/.cache/huggingface"
cd "$WORK_DIR"

SPLIT=${SPLIT:-News}
SPLIT_LOWER="${SPLIT,,}"

case $SPLIT in
    News)
        MODEL_PATH="saves/finetune/V7_muse_news_full_1epoch"
        RETAIN_LOGS="saves/eval/G2_baseline_1b_news_retain/MUSE_EVAL.json"
        ;;
    Books)
        MODEL_PATH="saves/finetune/G2_muse_books_1b_full"
        RETAIN_LOGS="saves/eval/G2_baseline_1b_books_retain/MUSE_EVAL.json"
        ;;
    *)
        echo "Unknown SPLIT: $SPLIT (use News or Books)"
        exit 1
        ;;
esac

W_NAIVE="0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0 4.0 5.0"
W_STATIC="1.5 2.0 2.5 3.0 4.0 5.0"
THRESHOLDS="0.55 0.65 0.75 0.85"

echo "=== G2 MUSE 1B TFU Sweep: SPLIT=$SPLIT, GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Model: $MODEL_PATH"
echo "Started: $(date)"

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found: $MODEL_PATH"
    echo "Run scripts/g2_finetune_1b.sh first for Books split."
    exit 1
fi

run_eval() {
    local task_name="$1"
    shift
    local result_file="saves/eval/${task_name}/MUSE_SUMMARY.json"
    if [ -f "$result_file" ]; then
        echo "[SKIP] $task_name"
        return 0
    fi
    echo "[RUN]  $task_name ($(date '+%H:%M:%S'))"
    $PYTHON src/eval.py --config-name=eval.yaml "$@" task_name="$task_name"
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$result_file" ]; then
        echo "[DONE] $task_name"
    else
        echo "[FAIL] $task_name (exit=$rc)"
    fi
}

# --- Naive sweep ---
echo "--- Naive 1B $SPLIT ---"
for w in $W_NAIVE; do
    run_eval "G2_tfu_1b_naive_w${w}_${SPLIT_LOWER}" \
        experiment=eval/tfu/muse \
        model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
        tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
        tfu.activation_method=naive \
        model.w=$w \
        data_split=$SPLIT \
        retain_logs_path="$RETAIN_LOGS"
done

# --- Static sweep ---
echo "--- Static 1B $SPLIT ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G2_tfu_1b_static_w${w}_th${th}_${SPLIT_LOWER}" \
            experiment=eval/tfu/muse \
            model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
            tfu.activation_method=static \
            tfu.activation_threshold=$th \
            model.w=$w \
            data_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

# --- Similarity sweep ---
echo "--- Similarity 1B $SPLIT ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G2_tfu_1b_sim_w${w}_th${th}_${SPLIT_LOWER}" \
            experiment=eval/tfu/muse \
            model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
            tfu.activation_method=similarity \
            tfu.activation_threshold=$th \
            model.w=$w \
            data_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "=== DONE: G2 MUSE 1B TFU $SPLIT at $(date) ==="
