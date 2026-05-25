#!/bin/bash
###############################################################################
# Long-Term GPU0 — Robust 36hr autonomous run
#
# Phases (sequential on GPU0):
#   1. G1 Phase 5: 8B+1B helper sweep (TOFU)         — ~2-3 hrs, 54 runs
#   2. G3 WMDP cyber (custom eval, TFU-aware)         — ~5 hrs,  60 runs
#   3. G2 8B TFU News (MUSE)                          — ~4 hrs, 110 runs
#   4. G2 1B TFU News (MUSE)                          — ~2 hrs,  59 runs
#
# Total: ~13-14 hrs, ~283 runs
#
# Error handling:
#   - Never exits on individual run failures
#   - Logs all errors to logs/longterm_gpu0_errors.log
#   - Phases always continue to next on any failure
#
# Usage:
#   tmux new -s gpu0
#   nohup bash scripts/longterm_gpu0.sh > logs/longterm_gpu0.log 2>&1 &
#   # or just: bash scripts/longterm_gpu0.sh 2>&1 | tee logs/longterm_gpu0.log
#
# Monitor: tail -f logs/longterm_gpu0.log
###############################################################################
set +e
source /path/to/workdir/hf_setup.sh

WORK_DIR="/path/to/workdir/open-unlearning"
export CUDA_HOME="/path/to/workdir/miniconda/envs/tfu"
export HF_HOME="/path/to/workdir/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0
cd "$WORK_DIR"
mkdir -p logs

ERRLOG="logs/longterm_gpu0_errors.log"
echo "=== Long-Term GPU0 Error Log ===" > "$ERRLOG"
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
echo "LONG-TERM GPU0 RUN"
echo "Started: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# Phase 1: G1 8B+1B helper sweep (TOFU)
run_phase "G1 Phase 5: 8B+1B Helper Sweep (TOFU)" \
    bash scripts/g1_tfu_8b1b_eval.sh

# Phase 2: G3 WMDP cyber (custom evaluator with TFU composition)
run_phase "G3 WMDP Custom Eval: cyber" \
    bash -c "SPLIT=cyber bash scripts/g3_tfu_eval.sh"

# Phase 3: G2 8B TFU News (MUSE)
run_phase "G2 8B TFU Sweep: News" \
    bash -c "SPLIT=News bash scripts/g2_tfu_8b_eval.sh"

# Phase 4: G2 1B TFU News (MUSE)
run_phase "G2 1B TFU Sweep: News" \
    bash -c "SPLIT=News bash scripts/g2_tfu_1b_eval.sh"

echo ""
echo "========================================================"
echo "LONG-TERM GPU0 ALL COMPLETE: $(date)"
echo "========================================================"
echo ""
echo "Summary:"
echo "  G1 8B+1B: $(ls -d saves/eval/G1_tfu_8b1b_*/TOFU_SUMMARY.json 2>/dev/null | wc -l) results"
echo "  G3 cyber: $(ls -d saves/eval/G3_tfu_zephyr_*_cyber/WMDP_SUMMARY.json 2>/dev/null | wc -l) / 60"
echo "  G2 8B News: $(ls -d saves/eval/G2_tfu_8b*_news/MUSE_SUMMARY.json 2>/dev/null | wc -l) results"
echo "  G2 1B News: $(ls -d saves/eval/G2_tfu_1b_*_news/MUSE_SUMMARY.json 2>/dev/null | wc -l) results"
echo ""
echo "Errors logged: $(grep -c 'PHASE-ERROR\|FAIL' "$ERRLOG" 2>/dev/null || echo 0)"
cat "$ERRLOG"
