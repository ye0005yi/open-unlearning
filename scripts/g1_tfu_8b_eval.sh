#!/bin/bash
###############################################################################
# G1 8B TFU Sweep — denser low + sparse high w grid
# Idempotent: skips runs that already have TOFU_SUMMARY.json
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 SPLIT=forget05 bash scripts/g1_tfu_8b_eval.sh
#   CUDA_VISIBLE_DEVICES=1 SPLIT=forget01 bash scripts/g1_tfu_8b_eval.sh
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh

WORK_DIR="/data/tfu_jx/open-unlearning"
PYTHON="/data/tfu_jx/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/data/tfu_jx/miniconda/envs/tfu"
export HF_HOME="/data/tfu_jx/.cache/huggingface"
cd "$WORK_DIR"

SPLIT=${SPLIT:-forget10}

case $SPLIT in
  forget01) RETAIN=retain99; SUFFIX="_forget01" ;;
  forget05) RETAIN=retain95; SUFFIX="_forget05" ;;
  forget10) RETAIN=retain90; SUFFIX="" ;;
  *) echo "Unknown SPLIT: $SPLIT"; exit 1 ;;
esac

RETAIN_LOGS="saves/eval/G1_baseline_llama8b_${RETAIN}/TOFU_EVAL.json"

# W grid: denser low + sparse high (matches workspace.md)
if [ "$SPLIT" == "forget10" ]; then
  W_VALUES="0.75 1.25 1.75 2.0 2.25 2.5 2.75 3.0 4.0 5.0"
else
  W_VALUES="0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 4.0 5.0"
fi
THRESHOLDS="0.55 0.65 0.75 0.85"

echo "=== 8B TFU Sweep: SPLIT=$SPLIT, GPU=$CUDA_VISIBLE_DEVICES ==="
echo "W values: $W_VALUES"
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

# Naive sweep
echo "--- Naive ---"
for w in $W_VALUES; do
    run_eval "G1_tfu_8b_naive_w${w}${SUFFIX}" \
        experiment=eval/tfu/default \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
        tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
        tfu.activation_method=naive \
        model.w=$w \
        forget_split=$SPLIT \
        retain_logs_path="$RETAIN_LOGS"
done

# Static sweep
echo "--- Static ---"
for w in $W_VALUES; do
    for th in $THRESHOLDS; do
        run_eval "G1_tfu_8b_static_w${w}_th${th}${SUFFIX}" \
            experiment=eval/tfu/default \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
            tfu.activation_method=static \
            tfu.activation_threshold=$th \
            model.w=$w \
            forget_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

# Similarity sweep
echo "--- Similarity ---"
for w in $W_VALUES; do
    for th in $THRESHOLDS; do
        run_eval "G1_tfu_8b_sim_w${w}_th${th}${SUFFIX}" \
            experiment=eval/tfu/default \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
            tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
            tfu.activation_method=similarity \
            tfu.activation_threshold=$th \
            model.w=$w \
            forget_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "=== DONE: 8B TFU $SPLIT at $(date) ==="
