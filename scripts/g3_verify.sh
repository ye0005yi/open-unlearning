#!/bin/bash
###############################################################################
# G3 WMDP Quick Verification — Confirm TFU pipeline works with WMDP eval
#
# Runs 3 configs and compares against /data/open-unlearning/ references:
#   1. TFU Zephyr w=1.0 cyber (baseline-like)
#   2. TFU Zephyr w=2.0 cyber (moderate unlearning)
#   3. TFU Zephyr w=1.5 bio (different split)
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/g3_verify.sh
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh

WORK_DIR="/data/tfu_jx/open-unlearning"
PYTHON="/data/tfu_jx/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/data/tfu_jx/miniconda/envs/tfu"
export HF_HOME="/data/tfu_jx/.cache/huggingface"
cd "$WORK_DIR"

echo "=== G3 WMDP Quick Verification: GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Started: $(date)"

run_eval() {
    local task_name="$1"
    shift
    local result_file="saves/eval/${task_name}/LMEval_SUMMARY.json"
    if [ -f "$result_file" ]; then
        echo "[SKIP] $task_name (already exists)"
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

# --- Verify 1: TFU Zephyr w=1.0 cyber (expect: mmlu~0.58, wmdp_cyber~0.45) ---
run_eval "G3_verify_tfu_zephyr_w1.0_cyber" \
    experiment=eval/tfu/wmdp \
    data_split=cyber \
    model.w=1.0 \
    tfu.activation_method=naive

# --- Verify 2: TFU Zephyr w=2.0 cyber (expect lower wmdp_cyber) ---
run_eval "G3_verify_tfu_zephyr_w2.0_cyber" \
    experiment=eval/tfu/wmdp \
    data_split=cyber \
    model.w=2.0 \
    tfu.activation_method=naive

# --- Verify 3: TFU Zephyr w=1.5 bio ---
run_eval "G3_verify_tfu_zephyr_w1.5_bio" \
    experiment=eval/tfu/wmdp \
    data_split=bio \
    model.w=1.5 \
    tfu.activation_method=naive

echo ""
echo "=== G3 WMDP Verification Complete: $(date) ==="
echo ""
echo "Compare against /data/open-unlearning/saves/eval/TFU_wmdp_zephyr/:"
echo "  G3_verify_tfu_zephyr_w1.0_cyber vs TFU_wmdp_cyber_zephyr7b1b_w1_0"
echo "  G3_verify_tfu_zephyr_w2.0_cyber vs TFU_wmdp_cyber_zephyr7b1b_w2_0"
echo "  G3_verify_tfu_zephyr_w1.5_bio   vs TFU_wmdp_bio_zephyr7b1b_w1_5"
echo ""
echo "Expected reference values:"
cat /data/open-unlearning/saves/eval/TFU_wmdp_zephyr/TFU_wmdp_cyber_zephyr7b1b_w1_0/LMEval_SUMMARY.json 2>/dev/null
cat /data/open-unlearning/saves/eval/TFU_wmdp_zephyr/TFU_wmdp_cyber_zephyr7b1b_w2_0/LMEval_SUMMARY.json 2>/dev/null
cat /data/open-unlearning/saves/eval/TFU_wmdp_zephyr/TFU_wmdp_bio_zephyr7b1b_w1_5/LMEval_SUMMARY.json 2>/dev/null
