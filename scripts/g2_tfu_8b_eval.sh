#!/bin/bash
###############################################################################
# G2 8B TFU Sweep for MUSE — lower w + full static/sim grids
# Complements /data/open-unlearning/ which has high-w (2.25-6.0) naive only.
#
# Covers:
#   - Naive: w=0.5,0.75,1.0,1.25,1.5,1.75,2.0 (complement existing 2.25-6.0)
#   - Static: key w values × thresholds
#   - Similarity: key w values × thresholds
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 SPLIT=News bash scripts/g2_tfu_8b_eval.sh
#   CUDA_VISIBLE_DEVICES=1 SPLIT=Books bash scripts/g2_tfu_8b_eval.sh
###############################################################################
set -o pipefail
source /path/to/workdir/hf_setup.sh

WORK_DIR="/path/to/workdir/open-unlearning"
PYTHON="/path/to/workdir/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/path/to/workdir/miniconda/envs/tfu"
export HF_HOME="/path/to/workdir/.cache/huggingface"
cd "$WORK_DIR"

SPLIT=${SPLIT:-News}
FINETUNE_BASE="/data/open-unlearning/saves/finetune"
MODEL_PATH="${FINETUNE_BASE}/muse_${SPLIT,,}_Llama-3.1-8B-Instruct_full"

# Retain logs (8B retain model eval from reference)
RETAIN_LOGS="/data/open-unlearning/saves/eval/muse_${SPLIT,,}_Llama-3.1-8B-Instruct_retain/MUSE_EVAL.json"

# W grid: lower values that complement /data/open-unlearning/ (has 2.25-6.0)
W_NAIVE="0.5 0.75 1.0 1.25 1.5 1.75 2.0"
W_STATIC="1.5 2.0 2.5 3.0 4.0 5.0"
THRESHOLDS="0.55 0.65 0.75 0.85"

echo "=== G2 MUSE 8B TFU Sweep: SPLIT=$SPLIT, GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Model: $MODEL_PATH"
echo "Started: $(date)"

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found: $MODEL_PATH"
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

SPLIT_LOWER="${SPLIT,,}"

# --- 8B+8B Naive sweep ---
echo "--- 8B+8B Naive ---"
for w in $W_NAIVE; do
    run_eval "G2_tfu_8b8b_naive_w${w}_${SPLIT_LOWER}" \
        experiment=eval/tfu/muse \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
        tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
        tfu.activation_method=naive \
        model.w=$w \
        data_split=$SPLIT \
        retain_logs_path="$RETAIN_LOGS"
done

# --- 8B+8B Static sweep ---
echo "--- 8B+8B Static ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G2_tfu_8b8b_static_w${w}_th${th}_${SPLIT_LOWER}" \
            experiment=eval/tfu/muse \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
            tfu.activation_method=static \
            tfu.activation_threshold=$th \
            model.w=$w \
            data_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

# --- 8B+8B Similarity sweep ---
echo "--- 8B+8B Similarity ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G2_tfu_8b8b_sim_w${w}_th${th}_${SPLIT_LOWER}" \
            experiment=eval/tfu/muse \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
            tfu.activation_method=similarity \
            tfu.activation_threshold=$th \
            model.w=$w \
            data_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

# --- 8B+1B Naive sweep ---
echo "--- 8B+1B Naive ---"
for w in $W_NAIVE; do
    run_eval "G2_tfu_8b1b_naive_w${w}_${SPLIT_LOWER}" \
        experiment=eval/tfu/muse \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
        tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
        tfu.activation_method=naive \
        model.w=$w \
        data_split=$SPLIT \
        retain_logs_path="$RETAIN_LOGS"
done

# --- 8B+1B Static sweep ---
echo "--- 8B+1B Static ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G2_tfu_8b1b_static_w${w}_th${th}_${SPLIT_LOWER}" \
            experiment=eval/tfu/muse \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
            tfu.activation_method=static \
            tfu.activation_threshold=$th \
            model.w=$w \
            data_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

# --- 8B+1B Similarity sweep ---
echo "--- 8B+1B Similarity ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G2_tfu_8b1b_sim_w${w}_th${th}_${SPLIT_LOWER}" \
            experiment=eval/tfu/muse \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=$MODEL_PATH \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
            tfu.activation_method=similarity \
            tfu.activation_threshold=$th \
            model.w=$w \
            data_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "=== DONE: G2 MUSE 8B TFU $SPLIT at $(date) ==="
