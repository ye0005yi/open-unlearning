#!/bin/bash
###############################################################################
# G2 MUSE Master Orchestrator — Optimal GPU parallelism
#
# Run inside tmux: tmux new -s g2
# Usage: nohup bash scripts/g2_run_all.sh > g2_master.log 2>&1 &
#
# Monitor: tail -f g2_master.log
#
# Phases:
#   1. Cross-verify (GPU0) + Finetune 1B models (GPU1)          — ~30 min
#   2. 8B TFU News (GPU0) + 8B TFU Books (GPU1)                 — ~4 hrs
#   3. 1B TFU News (GPU0) + 1B TFU Books (GPU1)                 — ~2 hrs
#
# Run counts:
#   - 8B per split: 7 naive×2helpers + 6w×4th×2helpers×2methods = 110 runs
#   - 1B per split: 11 naive + 6w×4th×2methods = 59 runs
#   - Total: 2×110 + 2×59 + 4 verify = 342 runs
###############################################################################
set -o pipefail
source /path/to/workdir/hf_setup.sh
cd /path/to/workdir/open-unlearning
mkdir -p logs

echo "========================================"
echo "G2 MUSE Complete Execution"
echo "Started: $(date)"
echo "========================================"

# Phase 1: Cross-verify (GPU0) + Finetune 1B (GPU1)
echo ""
echo "=== Phase 1: Cross-verify (GPU0) + Finetune 1B (GPU1) ==="

CUDA_VISIBLE_DEVICES=0 bash scripts/g2_cross_verify.sh > logs/g2_cross_verify.log 2>&1 &
PID_VERIFY=$!

CUDA_VISIBLE_DEVICES=1 bash scripts/g2_finetune_1b.sh > logs/g2_finetune_1b.log 2>&1 &
PID_FT=$!

wait $PID_VERIFY
echo "Cross-verify done: $(date)"

wait $PID_FT
echo "1B Finetune done: $(date)"

# Phase 2: 8B TFU sweep (News GPU0, Books GPU1)
echo ""
echo "=== Phase 2: 8B TFU News (GPU0) + 8B TFU Books (GPU1) ==="

CUDA_VISIBLE_DEVICES=0 SPLIT=News bash scripts/g2_tfu_8b_eval.sh > logs/g2_8b_news.log 2>&1 &
PID_NEWS=$!

CUDA_VISIBLE_DEVICES=1 SPLIT=Books bash scripts/g2_tfu_8b_eval.sh > logs/g2_8b_books.log 2>&1 &
PID_BOOKS=$!

wait $PID_NEWS
echo "8B News done: $(date)"

wait $PID_BOOKS
echo "8B Books done: $(date)"

# Phase 3: 1B TFU sweep (News GPU0, Books GPU1)
echo ""
echo "=== Phase 3: 1B TFU News (GPU0) + 1B TFU Books (GPU1) ==="

CUDA_VISIBLE_DEVICES=0 SPLIT=News bash scripts/g2_tfu_1b_eval.sh > logs/g2_1b_news.log 2>&1 &
PID_1B_NEWS=$!

CUDA_VISIBLE_DEVICES=1 SPLIT=Books bash scripts/g2_tfu_1b_eval.sh > logs/g2_1b_books.log 2>&1 &
PID_1B_BOOKS=$!

wait $PID_1B_NEWS
echo "1B News done: $(date)"

wait $PID_1B_BOOKS
echo "1B Books done: $(date)"

echo ""
echo "========================================"
echo "G2 MUSE ALL COMPLETE: $(date)"
echo "========================================"
echo ""
echo "Results:"
echo "  Cross-verify: $(ls -d saves/eval/G2_verify_*/MUSE_SUMMARY.json 2>/dev/null | wc -l) / 4"
echo "  8B TFU News:  $(ls -d saves/eval/G2_tfu_8b*_news/MUSE_SUMMARY.json 2>/dev/null | wc -l)"
echo "  8B TFU Books: $(ls -d saves/eval/G2_tfu_8b*_books/MUSE_SUMMARY.json 2>/dev/null | wc -l)"
echo "  1B TFU News:  $(ls -d saves/eval/G2_tfu_1b_*_news/MUSE_SUMMARY.json 2>/dev/null | wc -l)"
echo "  1B TFU Books: $(ls -d saves/eval/G2_tfu_1b_*_books/MUSE_SUMMARY.json 2>/dev/null | wc -l)"
echo "  G2 TOTAL:     $(ls -d saves/eval/G2_*/MUSE_SUMMARY.json 2>/dev/null | wc -l)"
