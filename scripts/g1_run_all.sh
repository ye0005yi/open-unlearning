#!/bin/bash
###############################################################################
# G1 Master Orchestrator — Optimal GPU parallelism
#
# Run inside tmux: tmux new -s g1
# Usage: nohup bash scripts/g1_run_all.sh > g1_master.log 2>&1 &
#
# Monitor: tail -f g1_master.log
#
# Phases:
#   1. Cross-verify (GPU0) + 8B Traditional DPO/UNDIAL (GPU1) — ~30 min
#   2. 1B higher-w (GPU0) + 8B TFU forget05 (GPU1)           — ~5 hrs
#   3. 8B TFU forget01 (GPU0) + forget05 continues (GPU1)    — ~5 hrs
#   4. 8B TFU forget10 remaining (GPU0)                       — ~3 hrs
#   5. 8B+1B helper sweep (GPU0 + GPU1 not needed simultaneously)
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh
cd /data/tfu_jx/open-unlearning
mkdir -p logs

echo "========================================"
echo "G1 TOFU Complete Execution"
echo "Started: $(date)"
echo "========================================"

# Phase 1: Cross-verify (GPU0) + 8B Traditional remaining (GPU1)
# Cross-verify: 4 runs × ~8 min = ~32 min
# 8B Traditional: DPO + UNDIAL × 3 splits = 6 runs × 8 min = ~48 min
echo ""
echo "=== Phase 1: Cross-verify (GPU0) + 8B Traditional DPO/UNDIAL (GPU1) ==="

CUDA_VISIBLE_DEVICES=0 bash scripts/g1_cross_verify.sh > logs/g1_cross_verify.log 2>&1 &
PID_VERIFY=$!

CUDA_VISIBLE_DEVICES=1 bash scripts/g1_train_8b.sh > logs/g1_8b_traditional.log 2>&1 &
PID_TRAD=$!

wait $PID_VERIFY
echo "Cross-verify done: $(date)"

# Phase 2: 1B higher-w (GPU0) + 8B TFU forget05 (GPU1)
# 1B: ~80 runs × 4 min = ~5.5 hrs
# 8B forget05: ~130 runs × 8 min = ~17 hrs (spans phases 2-3)
echo ""
echo "=== Phase 2: 1B higher-w (GPU0) + 8B TFU forget05 (GPU1) ==="

CUDA_VISIBLE_DEVICES=0 bash scripts/g1_tfu_1b_higher_w.sh > logs/g1_1b_higher_w.log 2>&1 &
PID1=$!

wait $PID_TRAD
echo "8B Traditional done: $(date)"

CUDA_VISIBLE_DEVICES=1 SPLIT=forget05 bash scripts/g1_tfu_8b_eval.sh > logs/g1_8b_forget05.log 2>&1 &
PID_F05=$!

wait $PID1
echo "1B higher-w done: $(date)"

# Phase 3: 8B TFU forget01 (GPU0) + forget05 continues on GPU1
echo ""
echo "=== Phase 3: 8B TFU forget01 (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 SPLIT=forget01 bash scripts/g1_tfu_8b_eval.sh > logs/g1_8b_forget01.log 2>&1 &
PID_F01=$!

wait $PID_F05
echo "8B forget05 done: $(date)"
wait $PID_F01
echo "8B forget01 done: $(date)"

# Phase 4: 8B TFU forget10 remaining (fills gaps in our G1_ naming)
echo ""
echo "=== Phase 4: 8B TFU forget10 remaining ==="
CUDA_VISIBLE_DEVICES=0 SPLIT=forget10 bash scripts/g1_tfu_8b_eval.sh > logs/g1_8b_forget10.log 2>&1 &
PID_F10=$!
wait $PID_F10
echo "8B forget10 done: $(date)"

# Phase 5: 8B+1B helper sweep (single GPU, loads 8B+1B = ~24GB)
echo ""
echo "=== Phase 5: 8B+1B helper sweep ==="
CUDA_VISIBLE_DEVICES=0 bash scripts/g1_tfu_8b1b_eval.sh > logs/g1_8b1b.log 2>&1
echo "8B+1B done: $(date)"

echo ""
echo "========================================"
echo "G1 ALL COMPLETE: $(date)"
echo "========================================"
echo ""
echo "Results:"
echo "  Cross-verify:   $(ls -d saves/eval/G1_verify_*/TOFU_SUMMARY.json 2>/dev/null | wc -l) / 4"
echo "  8B TFU (8B+8B): $(ls -d saves/eval/G1_tfu_8b_*/TOFU_SUMMARY.json 2>/dev/null | wc -l)"
echo "  8B TFU (8B+1B): $(ls -d saves/eval/G1_tfu_8b1b_*/TOFU_SUMMARY.json 2>/dev/null | wc -l)"
echo "  1B higher-w:    $(ls -d saves/eval/G1_tfu_naive_w[2-9]*/TOFU_SUMMARY.json saves/eval/G1_tfu_static_w[2-9]*/TOFU_SUMMARY.json saves/eval/G1_tfu_sim_w[2-9]*/TOFU_SUMMARY.json 2>/dev/null | wc -l)"
echo "  8B Traditional: $(ls -d saves/eval/G1_traditional_8b_*/TOFU_SUMMARY.json 2>/dev/null | wc -l)"
echo "  G1 TOTAL:       $(ls -d saves/eval/G1_*/TOFU_SUMMARY.json 2>/dev/null | wc -l)"
