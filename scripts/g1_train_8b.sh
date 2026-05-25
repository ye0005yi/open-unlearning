#!/bin/bash
###############################################################################
# G1 8B Traditional Methods — EVAL ONLY (checkpoints already exist)
# Checkpoints at: /data/open-unlearning/saves/unlearn/tofu_8b_{method}_{split}_s42
# Idempotent: skips completed evals
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/g1_train_8b.sh
###############################################################################
set -o pipefail
source /path/to/workdir/hf_setup.sh

WORK_DIR="/path/to/workdir/open-unlearning"
PYTHON="/path/to/workdir/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/path/to/workdir/miniconda/envs/tfu"
export HF_HOME="/path/to/workdir/.cache/huggingface"
cd "$WORK_DIR"

CKPT_BASE="/data/open-unlearning/saves/unlearn"

echo "=== 8B Traditional Methods Eval: GPU=$CUDA_VISIBLE_DEVICES ==="
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

# Methods missing from our G1_ results: dpo, undial
# Already done by killed batch: grad_ascent, grad_diff, npo, simnpo
for method in dpo undial; do
    for split in forget10 forget05 forget01; do
        case $split in
            forget01) RETAIN=retain99 ;;
            forget05) RETAIN=retain95 ;;
            forget10) RETAIN=retain90 ;;
        esac
        RETAIN_LOGS="saves/eval/G1_baseline_llama8b_${RETAIN}/TOFU_EVAL.json"
        CKPT="${CKPT_BASE}/tofu_8b_${method}_${split}_s42"

        if [ ! -d "$CKPT" ]; then
            echo "[WARN] Checkpoint not found: $CKPT"
            continue
        fi

        run_eval "G1_traditional_8b_${method}_${split}" \
            experiment=eval/tofu/default \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=$CKPT \
            eval.tofu.forget_split=$split \
            retain_logs_path="$RETAIN_LOGS"
    done
done

echo "=== DONE: 8B Traditional Eval at $(date) ==="
