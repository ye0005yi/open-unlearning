#!/bin/bash
###############################################################################
# G3 WMDP TFU Sweep — Zephyr-7b on cyber + bio splits
#
# Covers:
#   - Naive: w={0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,5.0}
#   - Static: w={1.25,1.5,1.75,2.0,2.5,3.0} × th={0.55,0.65,0.75,0.85}
#   - Similarity: same as static
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 SPLIT=cyber bash scripts/g3_tfu_eval.sh
#   CUDA_VISIBLE_DEVICES=1 SPLIT=bio bash scripts/g3_tfu_eval.sh
###############################################################################
set -o pipefail
source /path/to/workdir/hf_setup.sh

WORK_DIR="/path/to/workdir/open-unlearning"
PYTHON="/path/to/workdir/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/path/to/workdir/miniconda/envs/tfu"
export HF_HOME="/path/to/workdir/.cache/huggingface"
cd "$WORK_DIR"

SPLIT=${SPLIT:-cyber}
W_NAIVE="0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 5.0"
W_STATIC="1.25 1.5 1.75 2.0 2.5 3.0"
THRESHOLDS="0.55 0.65 0.75 0.85"

echo "=== G3 WMDP TFU Sweep: SPLIT=$SPLIT, GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Started: $(date)"

run_eval() {
    local task_name="$1"
    shift
    local result_file="saves/eval/${task_name}/WMDP_SUMMARY.json"
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
echo "--- Naive ---"
for w in $W_NAIVE; do
    run_eval "G3_tfu_zephyr_naive_w${w}_${SPLIT}" \
        experiment=eval/tfu/wmdp_custom \
        data_split=$SPLIT \
        model.w=$w \
        tfu.activation_method=naive
done

# --- Static sweep ---
echo "--- Static ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G3_tfu_zephyr_static_w${w}_th${th}_${SPLIT}" \
            experiment=eval/tfu/wmdp_custom \
            data_split=$SPLIT \
            model.w=$w \
            tfu.activation_method=static \
            tfu.activation_threshold=$th
    done
done

# --- Similarity sweep ---
echo "--- Similarity ---"
for w in $W_STATIC; do
    for th in $THRESHOLDS; do
        run_eval "G3_tfu_zephyr_sim_w${w}_th${th}_${SPLIT}" \
            experiment=eval/tfu/wmdp_custom \
            data_split=$SPLIT \
            model.w=$w \
            tfu.activation_method=similarity \
            tfu.activation_threshold=$th
    done
done

echo ""
echo "=== DONE: G3 WMDP TFU $SPLIT at $(date) ==="
echo "Results: $(ls -d saves/eval/G3_tfu_zephyr_*_${SPLIT}/WMDP_SUMMARY.json 2>/dev/null | wc -l) summaries"
