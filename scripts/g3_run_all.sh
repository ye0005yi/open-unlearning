#!/bin/bash
###############################################################################
# G3 WMDP Master Orchestrator
#
# Run inside tmux: tmux new -s g3
# Usage: nohup bash scripts/g3_run_all.sh > g3_master.log 2>&1 &
#
# Phases:
#   1. Verify (GPU0) — quick sanity check (3 runs, ~30 min)
#   2. TFU cyber (GPU0) + TFU bio (GPU1) — full sweep (~5 hrs)
#
# Run counts per split:
#   - Naive: 12 w values = 12 runs
#   - Static: 6w × 4th = 24 runs
#   - Similarity: 6w × 4th = 24 runs
#   - Total per split: 60 runs
#   - Grand total: 2 splits × 60 = 120 runs + 3 verify = 123
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh
cd /data/tfu_jx/open-unlearning
mkdir -p logs

echo "========================================"
echo "G3 WMDP Complete Execution"
echo "Started: $(date)"
echo "========================================"

# Phase 1: Quick verification
echo ""
echo "=== Phase 1: Quick Verification (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 bash scripts/g3_verify.sh > logs/g3_verify.log 2>&1
echo "Verification done: $(date)"
echo "Check logs/g3_verify.log for results"

# Phase 2: Full TFU sweep (cyber GPU0, bio GPU1)
echo ""
echo "=== Phase 2: TFU Sweep — cyber (GPU0) + bio (GPU1) ==="

CUDA_VISIBLE_DEVICES=0 SPLIT=cyber bash scripts/g3_tfu_eval.sh > logs/g3_tfu_cyber.log 2>&1 &
PID_CYBER=$!

CUDA_VISIBLE_DEVICES=1 SPLIT=bio bash scripts/g3_tfu_eval.sh > logs/g3_tfu_bio.log 2>&1 &
PID_BIO=$!

wait $PID_CYBER
echo "TFU cyber done: $(date)"

wait $PID_BIO
echo "TFU bio done: $(date)"

echo ""
echo "========================================"
echo "G3 WMDP ALL COMPLETE: $(date)"
echo "========================================"
echo ""
echo "Results:"
echo "  Verify:    $(ls -d saves/eval/G3_verify_*/WMDP_SUMMARY.json 2>/dev/null | wc -l) / 3"
echo "  TFU cyber: $(ls -d saves/eval/G3_tfu_zephyr_*_cyber/WMDP_SUMMARY.json 2>/dev/null | wc -l) / 60"
echo "  TFU bio:   $(ls -d saves/eval/G3_tfu_zephyr_*_bio/WMDP_SUMMARY.json 2>/dev/null | wc -l) / 60"
echo "  G3 TOTAL:  $(ls -d saves/eval/G3_*/WMDP_SUMMARY.json 2>/dev/null | wc -l) / 123"
