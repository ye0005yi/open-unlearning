#!/bin/bash
###############################################################################
# G1 Cross-Verification — Sample results from /data/open-unlearning/ and
# re-run with our pipeline to verify consistency.
#
# Picks 4 representative configs:
#   1. 8B DPO forget10 (traditional method)
#   2. 8B UNDIAL forget05 (traditional method, different split)
#   3. TFU 8B+8B w=3.25 forget10 (high-w TFU)
#   4. TFU 8B+1B w=3.75 forget10 (smaller helper)
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/g1_cross_verify.sh
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh

WORK_DIR="/data/tfu_jx/open-unlearning"
PYTHON="/data/tfu_jx/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/data/tfu_jx/miniconda/envs/tfu"
export HF_HOME="/data/tfu_jx/.cache/huggingface"
cd "$WORK_DIR"

CKPT_BASE="/data/open-unlearning/saves/unlearn"
RETAIN_F10="saves/eval/G1_baseline_llama8b_retain90/TOFU_EVAL.json"
RETAIN_F05="saves/eval/G1_baseline_llama8b_retain95/TOFU_EVAL.json"

echo "=== Cross-Verification: GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Started: $(date)"

run_eval() {
    local task_name="$1"
    shift
    local result_file="saves/eval/${task_name}/TOFU_SUMMARY.json"
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

# --- Verify 1: 8B DPO forget10 ---
# Original: /data/open-unlearning/saves/eval/tofu_8b_dpo_forget10_s42_eval
# Expected: MU=0.6125 ES=0.6319
run_eval "G1_verify_8b_dpo_forget10" \
    experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path="${CKPT_BASE}/tofu_8b_dpo_forget10_s42" \
    eval.tofu.forget_split=forget10 \
    retain_logs_path="$RETAIN_F10"

# --- Verify 2: 8B UNDIAL forget05 ---
# Original: /data/open-unlearning/saves/eval/tofu_8b_undial_forget05_s42_eval
# Expected: MU=0.4540 ES=0.0478
run_eval "G1_verify_8b_undial_forget05" \
    experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path="${CKPT_BASE}/tofu_8b_undial_forget05_s42" \
    eval.tofu.forget_split=forget05 \
    retain_logs_path="$RETAIN_F05"

# --- Verify 3: TFU 8B+8B w=3.25 forget10 ---
# Original: /data/open-unlearning/saves/eval/TFU_8b8b_w3_25
# Expected: MU=0.6267 ES=0.1048
run_eval "G1_verify_tfu_8b8b_w3.25" \
    experiment=eval/tfu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    tfu.activation_method=naive \
    model.w=3.25 \
    forget_split=forget10 \
    retain_logs_path="$RETAIN_F10"

# --- Verify 4: TFU 8B+1B w=3.75 forget10 ---
# Original: /data/open-unlearning/saves/eval/TFU_8b1b_w3_75
# Expected: MU=0.6266 ES=0.1474
run_eval "G1_verify_tfu_8b1b_w3.75" \
    experiment=eval/tfu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
    tfu.activation_method=naive \
    model.w=3.75 \
    forget_split=forget10 \
    retain_logs_path="$RETAIN_F10"

echo ""
echo "=== Cross-Verification Complete: $(date) ==="
echo ""
echo "Compare results:"
echo "  G1_verify_8b_dpo_forget10   vs tofu_8b_dpo_forget10_s42_eval   (expect MU≈0.6125 ES≈0.6319)"
echo "  G1_verify_8b_undial_forget05 vs tofu_8b_undial_forget05_s42_eval (expect MU≈0.4540 ES≈0.0478)"
echo "  G1_verify_tfu_8b8b_w3.25   vs TFU_8b8b_w3_25                  (expect MU≈0.6267 ES≈0.1048)"
echo "  G1_verify_tfu_8b1b_w3.75   vs TFU_8b1b_w3_75                  (expect MU≈0.6266 ES≈0.1474)"
