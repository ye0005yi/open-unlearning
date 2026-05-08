#!/bin/bash
###############################################################################
# Long-Term GPU1 — Robust 36hr autonomous run
#
# Phases (sequential on GPU1):
#   1. G2 Finetune 1B models (MUSE prereq)             — ~30 min
#   2. G2 Cross-verify (remaining MUSE configs)         — ~30 min
#   3. G3 WMDP bio (custom eval, TFU-aware)             — ~5 hrs,  60 runs
#   4. G2 8B TFU Books (MUSE)                           — ~4 hrs, 110 runs
#   5. G2 1B TFU Books (MUSE)                           — ~2 hrs,  59 runs
#
# Total: ~12-13 hrs, ~233 runs
#
# Error handling:
#   - Never exits on individual run failures
#   - Logs all errors to logs/longterm_gpu1_errors.log
#   - Phases always continue to next on any failure
#
# Usage:
#   tmux new -s gpu1
#   nohup bash scripts/longterm_gpu1.sh > logs/longterm_gpu1.log 2>&1 &
#   # or just: bash scripts/longterm_gpu1.sh 2>&1 | tee logs/longterm_gpu1.log
#
# Monitor: tail -f logs/longterm_gpu1.log
###############################################################################
set +e
source /data/tfu_jx/hf_setup.sh

WORK_DIR="/data/tfu_jx/open-unlearning"
export CUDA_HOME="/data/tfu_jx/miniconda/envs/tfu"
export HF_HOME="/data/tfu_jx/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=1
cd "$WORK_DIR"
mkdir -p logs

ERRLOG="logs/longterm_gpu1_errors.log"
echo "=== Long-Term GPU1 Error Log ===" > "$ERRLOG"
echo "Started: $(date)" >> "$ERRLOG"

run_phase() {
    local phase_name="$1"
    shift
    echo ""
    echo "========================================================"
    echo "PHASE: $phase_name"
    echo "Started: $(date)"
    echo "========================================================"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[PHASE-ERROR] $phase_name exited $rc at $(date)" | tee -a "$ERRLOG"
    else
        echo "[PHASE-OK] $phase_name completed at $(date)"
    fi
    echo ""
}

echo "========================================================"
echo "LONG-TERM GPU1 RUN"
echo "Started: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# Phase 1: G2 Finetune 1B (creates Books 1B model + retain evals)
run_phase "G2 Finetune 1B Models (MUSE prereq)" \
    bash scripts/g2_finetune_1b.sh

# Phase 2: G2 Cross-verify (remaining configs)
run_phase "G2 Cross-Verify (MUSE)" \
    bash scripts/g2_cross_verify.sh

# Phase 3: G3 WMDP bio (custom evaluator with TFU composition)
run_phase "G3 WMDP Custom Eval: bio" \
    bash -c "SPLIT=bio bash scripts/g3_tfu_eval.sh"

# Phase 4: G2 8B TFU Books (MUSE)
run_phase "G2 8B TFU Sweep: Books" \
    bash -c "SPLIT=Books bash scripts/g2_tfu_8b_eval.sh"

# Phase 5: G2 1B TFU Books (MUSE)
run_phase "G2 1B TFU Sweep: Books" \
    bash -c "SPLIT=Books bash scripts/g2_tfu_1b_eval.sh"

echo ""
echo "========================================================"
echo "LONG-TERM GPU1 ALL COMPLETE: $(date)"
echo "========================================================"
echo ""
echo "Summary:"
echo "  G2 Finetune:  $(ls -d saves/finetune/G2_muse_*_1b_* 2>/dev/null | wc -l) models"
echo "  G2 Verify:    $(ls -d saves/eval/G2_verify_*/MUSE_SUMMARY.json 2>/dev/null | wc -l) / 4"
echo "  G3 bio:       $(ls -d saves/eval/G3_tfu_zephyr_*_bio/WMDP_SUMMARY.json 2>/dev/null | wc -l) / 60"
echo "  G2 8B Books:  $(ls -d saves/eval/G2_tfu_8b*_books/MUSE_SUMMARY.json 2>/dev/null | wc -l) results"
echo "  G2 1B Books:  $(ls -d saves/eval/G2_tfu_1b_*_books/MUSE_SUMMARY.json 2>/dev/null | wc -l) results"
echo ""
echo "Errors logged: $(grep -c 'PHASE-ERROR\|FAIL' "$ERRLOG" 2>/dev/null || echo 0)"
cat "$ERRLOG"
