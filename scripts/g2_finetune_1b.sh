#!/bin/bash
###############################################################################
# G2 Finetune 1B models for MUSE
#
# Creates missing models:
#   1. 1B Books full (target model for TFU)
#   2. 1B News retain (for retain_logs_path)
#   3. 1B Books retain (for retain_logs_path)
#
# Note: 1B News full already exists at:
#   /path/to/workdir/open-unlearning/saves/finetune/V7_muse_news_full_1epoch
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/g2_finetune_1b.sh
###############################################################################
set -o pipefail
source /path/to/workdir/hf_setup.sh

WORK_DIR="/path/to/workdir/open-unlearning"
PYTHON="/path/to/workdir/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/path/to/workdir/miniconda/envs/tfu"
export HF_HOME="/path/to/workdir/.cache/huggingface"
cd "$WORK_DIR"

echo "=== G2 MUSE 1B Finetuning: GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Started: $(date)"

# --- 1. 1B Books full ---
CKPT="saves/finetune/G2_muse_books_1b_full"
if [ -d "$CKPT" ]; then
    echo "[SKIP] 1B Books full (exists)"
else
    echo "[RUN]  1B Books full ($(date '+%H:%M:%S'))"
    $PYTHON src/train.py --config-name=train.yaml \
        experiment=finetune/muse/1b \
        data_split=Books \
        data_sub_set=full \
        task_name=G2_muse_books_1b_full
    echo "[DONE] 1B Books full"
fi

# --- 2. 1B News retain ---
CKPT="saves/finetune/G2_muse_news_1b_retain"
if [ -d "$CKPT" ]; then
    echo "[SKIP] 1B News retain (exists)"
else
    echo "[RUN]  1B News retain ($(date '+%H:%M:%S'))"
    $PYTHON src/train.py --config-name=train.yaml \
        experiment=finetune/muse/1b \
        data_split=News \
        data_sub_set=retain \
        task_name=G2_muse_news_1b_retain
    echo "[DONE] 1B News retain"
fi

# --- 3. 1B Books retain ---
CKPT="saves/finetune/G2_muse_books_1b_retain"
if [ -d "$CKPT" ]; then
    echo "[SKIP] 1B Books retain (exists)"
else
    echo "[RUN]  1B Books retain ($(date '+%H:%M:%S'))"
    $PYTHON src/train.py --config-name=train.yaml \
        experiment=finetune/muse/1b \
        data_split=Books \
        data_sub_set=retain \
        task_name=G2_muse_books_1b_retain
    echo "[DONE] 1B Books retain"
fi

# --- 4. Eval retain models for retain_logs_path ---
echo ""
echo "--- Evaluating retain models for retain_logs_path ---"

# 1B News retain eval
EVAL_FILE="saves/eval/G2_baseline_1b_news_retain/MUSE_EVAL.json"
if [ -f "$EVAL_FILE" ]; then
    echo "[SKIP] 1B News retain eval"
else
    echo "[RUN]  1B News retain eval ($(date '+%H:%M:%S'))"
    $PYTHON src/eval.py --config-name=eval.yaml \
        experiment=eval/tfu/muse \
        model.model_args.pretrained_model_name_or_path=saves/finetune/G2_muse_news_1b_retain \
        tfu.activation_method=naive \
        model.w=1.0 \
        data_split=News \
        task_name=G2_baseline_1b_news_retain
    echo "[DONE] 1B News retain eval"
fi

# 1B Books retain eval
EVAL_FILE="saves/eval/G2_baseline_1b_books_retain/MUSE_EVAL.json"
if [ -f "$EVAL_FILE" ]; then
    echo "[SKIP] 1B Books retain eval"
else
    echo "[RUN]  1B Books retain eval ($(date '+%H:%M:%S'))"
    $PYTHON src/eval.py --config-name=eval.yaml \
        experiment=eval/tfu/muse \
        model.model_args.pretrained_model_name_or_path=saves/finetune/G2_muse_books_1b_retain \
        tfu.activation_method=naive \
        model.w=1.0 \
        data_split=Books \
        task_name=G2_baseline_1b_books_retain
    echo "[DONE] 1B Books retain eval"
fi

echo ""
echo "=== G2 1B Finetune DONE: $(date) ==="
echo "Models created:"
echo "  saves/finetune/G2_muse_books_1b_full"
echo "  saves/finetune/G2_muse_news_1b_retain"
echo "  saves/finetune/G2_muse_books_1b_retain"
