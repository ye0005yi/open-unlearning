#!/bin/bash
###############################################################################
# G2 Cross-Verification — Sample MUSE results from /data/open-unlearning/
# and re-run with our pipeline to verify consistency.
#
# Picks 4 representative configs:
#   1. 8B GradDiff News (traditional method)
#   2. 8B NPO Books (traditional method, different split)
#   3. TFU 8B+8B w=4.0 News (high-w TFU)
#   4. TFU 8B+1B w=4.0 Books (smaller helper, different split)
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/g2_cross_verify.sh
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh

WORK_DIR="/data/tfu_jx/open-unlearning"
PYTHON="/data/tfu_jx/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/data/tfu_jx/miniconda/envs/tfu"
export HF_HOME="/data/tfu_jx/.cache/huggingface"
cd "$WORK_DIR"

CKPT_BASE="/data/open-unlearning/saves/unlearn"
FINETUNE_BASE="/data/open-unlearning/saves/finetune"

echo "=== G2 MUSE Cross-Verification: GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Started: $(date)"

run_eval() {
    local task_name="$1"
    shift
    local result_file="saves/eval/${task_name}/MUSE_SUMMARY.json"
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

# Retain logs for privleak computation
RETAIN_LOGS_NEWS="/data/open-unlearning/saves/eval/muse_news_Llama-3.1-8B-Instruct_retain/MUSE_EVAL.json"
RETAIN_LOGS_BOOKS="/data/open-unlearning/saves/eval/muse_books_Llama-3.1-8B-Instruct_retain/MUSE_EVAL.json"

# --- Verify 1: 8B GradDiff News ---
run_eval "G2_verify_8b_grad_diff_news" \
    experiment=eval/tofu/default \
    defaults/model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path="${CKPT_BASE}/muse_news_8b_grad_diff_s42" \
    eval=muse \
    eval.muse.data_split=News \
    eval.muse.retain_logs_path="$RETAIN_LOGS_NEWS" \
    data.forget.MUSE_forget.args.hf_args.path=muse-bench/MUSE-News \
    data.retain.MUSE_retain.args.hf_args.path=muse-bench/MUSE-News

# --- Verify 2: 8B NPO Books ---
run_eval "G2_verify_8b_npo_books" \
    experiment=eval/tofu/default \
    defaults/model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path="${CKPT_BASE}/muse_books_8b_npo_s42" \
    eval=muse \
    eval.muse.data_split=Books \
    eval.muse.retain_logs_path="$RETAIN_LOGS_BOOKS" \
    data.forget.MUSE_forget.args.hf_args.path=muse-bench/MUSE-Books \
    data.retain.MUSE_retain.args.hf_args.path=muse-bench/MUSE-Books

# --- Verify 3: TFU 8B+8B w=4.0 News ---
run_eval "G2_verify_tfu_8b8b_w4.0_news" \
    experiment=eval/tfu/muse \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path="${FINETUNE_BASE}/muse_news_Llama-3.1-8B-Instruct_full" \
    tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    tfu.activation_method=naive \
    model.w=4.0 \
    data_split=News \
    retain_logs_path="$RETAIN_LOGS_NEWS"

# --- Verify 4: TFU 8B+1B w=4.0 Books ---
run_eval "G2_verify_tfu_8b1b_w4.0_books" \
    experiment=eval/tfu/muse \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path="${FINETUNE_BASE}/muse_books_Llama-3.1-8B-Instruct_full" \
    tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
    tfu.activation_method=naive \
    model.w=4.0 \
    data_split=Books \
    retain_logs_path="$RETAIN_LOGS_BOOKS"

echo ""
echo "=== G2 MUSE Cross-Verification Complete: $(date) ==="
echo ""
echo "Compare results:"
echo "  G2_verify_8b_graddiff_news    vs muse_news_8b_graddiff_s42_eval"
echo "  G2_verify_8b_npo_books        vs muse_books_8b_npo_s42_eval"
echo "  G2_verify_tfu_8b8b_w4.0_news  vs TFU_muse_news_8b8b_w4_0"
echo "  G2_verify_tfu_8b1b_w4.0_books vs TFU_muse_books_8b1b_w4_0"
