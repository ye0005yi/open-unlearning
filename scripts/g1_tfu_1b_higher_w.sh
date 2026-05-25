#!/bin/bash
###############################################################################
# G1 1B TFU Higher-w Sweep — extends existing {0.5-1.5} to {1.75-5.0}
# Idempotent: skips completed runs
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/g1_tfu_1b_higher_w.sh
###############################################################################
set -o pipefail
source /path/to/workdir/hf_setup.sh

WORK_DIR="/path/to/workdir/open-unlearning"
PYTHON="/path/to/workdir/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/path/to/workdir/miniconda/envs/tfu"
export HF_HOME="/path/to/workdir/.cache/huggingface"
cd "$WORK_DIR"

W_HIGH="1.75 2.0 2.25 2.5 2.75 3.0 4.0 5.0"
THRESHOLDS="0.55 0.65 0.75 0.85"

echo "=== 1B Higher-w Sweep: GPU=$CUDA_VISIBLE_DEVICES ==="
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

# Naive higher-w for all splits
for SPLIT in forget10 forget05 forget01; do
    case $SPLIT in
        forget01) RETAIN=retain99; SUFFIX="_forget01" ;;
        forget05) RETAIN=retain95; SUFFIX="_forget05" ;;
        forget10) RETAIN=retain90; SUFFIX="" ;;
    esac
    RETAIN_LOGS="saves/eval/G1_baseline_llama1b_${RETAIN}/TOFU_EVAL.json"

    echo "--- Naive $SPLIT ---"
    for w in $W_HIGH; do
        run_eval "G1_tfu_naive_w${w}${SUFFIX}" \
            experiment=eval/tfu/default \
            tfu.activation_method=naive \
            model.w=$w \
            forget_split=$SPLIT \
            retain_logs_path="$RETAIN_LOGS"
    done
done

# Static + Similarity higher-w (forget10 only)
RETAIN_LOGS="saves/eval/G1_baseline_llama1b_retain90/TOFU_EVAL.json"

echo "--- Static forget10 ---"
for w in $W_HIGH; do
    for th in $THRESHOLDS; do
        run_eval "G1_tfu_static_w${w}_th${th}" \
            experiment=eval/tfu/default \
            tfu.activation_method=static \
            tfu.activation_threshold=$th \
            model.w=$w \
            forget_split=forget10 \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "--- Similarity forget10 ---"
for w in $W_HIGH; do
    for th in $THRESHOLDS; do
        run_eval "G1_tfu_sim_w${w}_th${th}" \
            experiment=eval/tfu/default \
            tfu.activation_method=similarity \
            tfu.activation_threshold=$th \
            model.w=$w \
            forget_split=forget10 \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "=== DONE: 1B higher-w at $(date) ==="
