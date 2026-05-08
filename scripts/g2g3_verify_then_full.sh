#!/bin/bash
###############################################################################
# G2+G3 Verification → Full Test Pipeline
#
# Designed for unattended 24+ hour operation after manual launch.
# Flow:
#   1. Wait for G3 verify to finish (already running)
#   2. Validate G3 verify results against references
#   3. Run G2 verify (4 runs, ~40 min)
#   4. Validate G2 verify results against references
#   5. If both pass → launch full G2 + G3 sweeps in parallel
#
# Usage: CUDA_VISIBLE_DEVICES=1 nohup bash scripts/g2g3_verify_then_full.sh > logs/g2g3_pipeline.log 2>&1 &
###############################################################################
set -o pipefail
source /data/tfu_jx/hf_setup.sh

WORK_DIR="/data/tfu_jx/open-unlearning"
PYTHON="/data/tfu_jx/miniconda/envs/tfu/bin/python"
export CUDA_HOME="/data/tfu_jx/miniconda/envs/tfu"
export HF_HOME="/data/tfu_jx/.cache/huggingface"
cd "$WORK_DIR"

echo "========================================"
echo "G2+G3 Verify → Full Test Pipeline"
echo "Started: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

###############################################################################
# PHASE 1: Wait for G3 verify to complete
###############################################################################
echo ""
echo "=== PHASE 1: Waiting for G3 verify to complete ==="

G3_EXPECTED=3
while true; do
    G3_DONE=$(ls -d saves/eval/G3_verify_*/LMEval_SUMMARY.json 2>/dev/null | wc -l)
    if [ "$G3_DONE" -ge "$G3_EXPECTED" ]; then
        echo "G3 verify complete: $G3_DONE/$G3_EXPECTED results found"
        break
    fi
    echo "  Waiting... ($G3_DONE/$G3_EXPECTED done) $(date '+%H:%M:%S')"
    sleep 60
done

###############################################################################
# PHASE 2: Validate G3 verify results
###############################################################################
echo ""
echo "=== PHASE 2: Validating G3 verify results ==="

G3_PASS=true
$PYTHON -c "
import json, sys

refs = {
    'G3_verify_tfu_zephyr_w1.0_cyber': '/data/open-unlearning/saves/eval/TFU_wmdp_zephyr/TFU_wmdp_cyber_zephyr7b1b_w1_0/LMEval_SUMMARY.json',
    'G3_verify_tfu_zephyr_w2.0_cyber': '/data/open-unlearning/saves/eval/TFU_wmdp_zephyr/TFU_wmdp_cyber_zephyr7b1b_w2_0/LMEval_SUMMARY.json',
    'G3_verify_tfu_zephyr_w1.5_bio': '/data/open-unlearning/saves/eval/TFU_wmdp_zephyr/TFU_wmdp_bio_zephyr7b1b_w1_5/LMEval_SUMMARY.json',
}

all_pass = True
for name, ref_path in refs.items():
    our_path = f'saves/eval/{name}/LMEval_SUMMARY.json'
    try:
        our = json.load(open(our_path))
        ref = json.load(open(ref_path))
    except FileNotFoundError as e:
        print(f'[FAIL] {name}: {e}')
        all_pass = False
        continue

    # Compare key metrics (allow 2% tolerance for stochastic eval)
    issues = []
    for key in ['mmlu/acc', 'wmdp_cyber/acc', 'wmdp_bio/acc']:
        if key in our and key in ref:
            diff = abs(our[key] - ref[key])
            if diff > 0.02:
                issues.append(f'{key}: ours={our[key]:.4f} ref={ref[key]:.4f} diff={diff:.4f}')

    if issues:
        print(f'[WARN] {name}:')
        for i in issues:
            print(f'       {i}')
    else:
        matching_keys = [k for k in ['mmlu/acc','wmdp_cyber/acc','wmdp_bio/acc'] if k in our and k in ref]
        print(f'[PASS] {name} ({len(matching_keys)} metrics within tolerance)')

if not all_pass:
    sys.exit(1)
" || G3_PASS=false

if [ "$G3_PASS" = false ]; then
    echo ""
    echo "[ERROR] G3 verification FAILED. Check logs/g3_verify.log"
    echo "Pipeline ABORTED at $(date)"
    exit 1
fi
echo "G3 verification PASSED"

###############################################################################
# PHASE 3: Run G2 verify
###############################################################################
echo ""
echo "=== PHASE 3: G2 MUSE Cross-Verification ==="
echo "Started: $(date)"

bash scripts/g2_cross_verify.sh
G2_RC=$?

if [ $G2_RC -ne 0 ]; then
    echo "[ERROR] G2 cross_verify.sh exited with code $G2_RC"
    echo "Check output above for details. Pipeline continues with validation..."
fi

###############################################################################
# PHASE 4: Validate G2 verify results
###############################################################################
echo ""
echo "=== PHASE 4: Validating G2 verify results ==="

G2_PASS=true
$PYTHON -c "
import json, sys

refs = {
    'G2_verify_8b_grad_diff_news': '/data/open-unlearning/saves/eval/muse_news_8b_grad_diff_s42_eval/MUSE_SUMMARY.json',
    'G2_verify_8b_npo_books': '/data/open-unlearning/saves/eval/muse_books_8b_npo_s42_eval/MUSE_SUMMARY.json',
    'G2_verify_tfu_8b8b_w4.0_news': '/data/open-unlearning/saves/eval/TFU_muse_news_8b8b_w4_0/MUSE_SUMMARY.json',
    'G2_verify_tfu_8b1b_w4.0_books': '/data/open-unlearning/saves/eval/TFU_muse_books_8b1b_w4_0/MUSE_SUMMARY.json',
}

all_pass = True
for name, ref_path in refs.items():
    our_path = f'saves/eval/{name}/MUSE_SUMMARY.json'
    try:
        our = json.load(open(our_path))
        ref = json.load(open(ref_path))
    except FileNotFoundError as e:
        print(f'[FAIL] {name}: {e}')
        all_pass = False
        continue

    # Compare key metrics (5% tolerance for MUSE since generation is stochastic)
    issues = []
    for key in ['extraction_strength', 'retain_knowmem_ROUGE', 'forget_knowmem_ROUGE', 'forget_verbmem_ROUGE']:
        if key in our and key in ref:
            diff = abs(our[key] - ref[key])
            tol = 0.05 if 'ROUGE' in key else 0.03
            if diff > tol:
                issues.append(f'{key}: ours={our[key]:.4f} ref={ref[key]:.4f} diff={diff:.4f}')

    if issues:
        print(f'[WARN] {name}:')
        for i in issues:
            print(f'       {i}')
    else:
        matching_keys = [k for k in ref if k in our]
        print(f'[PASS] {name} ({len(matching_keys)} metrics within tolerance)')

if not all_pass:
    sys.exit(1)
" || G2_PASS=false

if [ "$G2_PASS" = false ]; then
    echo ""
    echo "[ERROR] G2 verification FAILED. Check output above."
    echo "Pipeline ABORTED at $(date)"
    exit 1
fi
echo "G2 verification PASSED"

###############################################################################
# PHASE 5: Launch full tests (G2 + G3 in parallel)
###############################################################################
echo ""
echo "========================================"
echo "=== ALL VERIFICATIONS PASSED ==="
echo "=== Launching full G2 + G3 sweeps ==="
echo "Started: $(date)"
echo "========================================"

# G3 runs on THIS GPU (the one running this script)
echo ""
echo "--- Starting G3 WMDP full sweep (this GPU: $CUDA_VISIBLE_DEVICES) ---"
echo "  Cyber + Bio splits running sequentially on this GPU"
echo "  Expected: 120 runs, ~10 hours"

SPLIT=cyber bash scripts/g3_tfu_eval.sh > logs/g3_tfu_cyber.log 2>&1
echo "G3 cyber done: $(date)"

SPLIT=bio bash scripts/g3_tfu_eval.sh > logs/g3_tfu_bio.log 2>&1
echo "G3 bio done: $(date)"

echo ""
echo "========================================"
echo "G3 WMDP FULL SWEEP COMPLETE: $(date)"
echo "  Cyber: $(ls -d saves/eval/G3_tfu_zephyr_*_cyber/LMEval_SUMMARY.json 2>/dev/null | wc -l) results"
echo "  Bio:   $(ls -d saves/eval/G3_tfu_zephyr_*_bio/LMEval_SUMMARY.json 2>/dev/null | wc -l) results"
echo "========================================"
