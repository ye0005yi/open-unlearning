#!/bin/bash
###############################################################################
# G1 8B TFU with 1B Helper — Smaller helper model sweep
# Tests whether a 1B helper provides sufficient "clean" signal for 8B main.
#
# Covers:
#   - Naive: w=0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0 per split
#     (w=2.25-6.0 already done in /data/open-unlearning/, w=3.75 for all splits)
#   - Static: key w values × thresholds for forget10
#   - Similarity: key w values × thresholds for forget10
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/g1_tfu_8b1b_eval.sh
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh

WORK_DIR="/data/tfu_jx/open-unlearning"
PYTHON="/data/tfu_jx/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/data/tfu_jx/miniconda/envs/tfu"
export HF_HOME="/data/tfu_jx/.cache/huggingface"
cd "$WORK_DIR"

# Lower w values that complement /data/open-unlearning/ (has 2.25-6.0)
W_VALUES="0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0"
THRESHOLDS="0.55 0.65 0.75 0.85"
# Key w values for static/sim (focused on effective range)
W_STATIC="1.5 2.0 2.5 3.0 4.0 5.0"

echo "=== 8B TFU with 1B Helper: GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Started: $(date)"

run_eval() {
    local task_name="$1"
    shift
    local result_file="saves/eval/${task_name}/TOFU_SUMMARY.json"
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

# --- Naive sweep (all splits) ---
for SPLIT in forget10 forget05 forget01; do
    case $SPLIT in
        forget01) RETAIN=retain99; SUFFIX="_forget01" ;;
        forget05) RETAIN=retain95; SUFFIX="_forget05" ;;
        forget10) RETAIN=retain90; SUFFIX="" ;;
    esac
    RETAIN_LOGS="saves/eval/G1_baseline_llama8b_${RETAIN}/TOFU_EVAL.json"

    echo "--- Naive 8B+1B $SPLIT ---"
    for w in $W_VALUES; do
        run_eval "G1_tfu_8b1b_naive_w${w}${SUFFIX}" \
            experiment=eval/tfu/default \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
            tfu.activation_method=naive \
            model.w=$w \
            forget_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

# --- Static/Similarity sweep (forget10 only, focused w range) ---
RETAIN_LOGS="saves/eval/G1_baseline_llama8b_retain90/TOFU_EVAL.json"

echo "--- Static 8B+1B forget10 ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G1_tfu_8b1b_static_w${w}_th${th}" \
            experiment=eval/tfu/default \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
            tfu.activation_method=static \
            tfu.activation_threshold=$th \
            model.w=$w \
            forget_split=forget10 \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "--- Similarity 8B+1B forget10 ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G1_tfu_8b1b_sim_w${w}_th${th}" \
            experiment=eval/tfu/default \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
            tfu.activation_method=similarity \
            tfu.activation_threshold=$th \
            model.w=$w \
            forget_split=forget10 \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "=== DONE: 8B+1B TFU at $(date) ==="
