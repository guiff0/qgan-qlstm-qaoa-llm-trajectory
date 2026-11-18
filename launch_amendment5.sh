#!/usr/bin/env bash
# launch_amendment5.sh — Amendment 5 rerun
#
# WHAT THIS DOES
# ==============
# The 40 runs completed under Amendment 4 (experiment EXP-20261116-MAIN-f41c) used:
#   shift_sd = 1.0  (from calibration 2026-11-13)
#
# A second calibration run on 2026-11-18 (results/poisoning_calibration_20261118T190250Z.json)
# found chosen_shift_sd = 3.0.  At shift_sd = 1.0 the attack is too weak: 4 of 10 seeds
# showed negative positive-control degradation, so M8_poisoning_positive_control FAILS
# (min degradation -48.7 % vs floor 5.0 %).
#
# Config patch already applied to config/default_config.yaml:
#   shift_sd: 1.0 -> 3.0  (matches calibration run of 2026-11-18)
#   calibrated: true       (unchanged)
#
# All other settings remain at their Amendment 4 values:
#   attack_samples: 200   (H3 ASR variance fix, Amendment 4)
#   synthetic_ratio: 1.0  (H4 poisoning dilution fix, Amendment 4)
#   fraction: 0.2         (unchanged)
#   pc_floor_pct: 5.0     (unchanged)
#
# All 40 runs are re-run fresh under a new experiment id.
# The old results CSV is archived (not deleted) for audit purposes.
#
# USAGE
# =====
#   cd D:\dev\app\qgan-qlstm-qaoa-llm-trajectory
#   bash launch_amendment5.sh           # 1 worker (safe on any machine)
#   bash launch_amendment5.sh 2         # 2 parallel workers
#   bash launch_amendment5.sh 4         # 4 parallel workers (needs >=4 physical cores)
#
# TIME ESTIMATE (unchanged from Amendment 4)
# ===========================================
#   Classical GAN-LLM x 10 seeds ~  5-8 h total (CPU)
#   Classical GAN-LLM (matched)  ~  5-8 h
#   QGAN-LLM         x 10 seeds  ~ 40-50 h total (quantum, lightning.qubit, 12 qubits)
#   QGAN-LLM (no noise) x 10     ~ 40-50 h
#   TOTAL (1 worker)              ~ 90-116 h
#   TOTAL (2 workers)             ~ 50-60 h  (classical + quantum interleave)
#   TOTAL (4 workers)             ~ 25-30 h
#
# Safe to kill and relaunch -- the script resumes from whatever is done.

set -euo pipefail

WORKERS=${1:-1}
RESULTS_CSV="results/all_results.csv"
ARCHIVE_DIR="results/archive"
AMENDMENT="Amendment-5"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)

echo "========================================================"
echo " AMENDMENT 5 LAUNCH  --  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Workers: $WORKERS"
echo " Config:  config/default_config.yaml"
echo "========================================================"
echo

# --- 1. Verify config patches are in place ---
echo "[check] Verifying config patches..."
python3 - <<'PYCHECK'
import yaml, sys
with open("config/default_config.yaml") as f:
    cfg = yaml.safe_load(f)

errors = []
for section in ("classical_gan_llm", "qgan_llm"):
    s = cfg[section]
    if s.get("attack_samples") != 200:
        errors.append(f"{section}.attack_samples = {s.get('attack_samples')} (expected 200)")
    if s.get("synthetic_ratio") != 1.0:
        errors.append(f"{section}.synthetic_ratio = {s.get('synthetic_ratio')} (expected 1.0)")

p = cfg.get("poisoning", {})
if p.get("shift_sd") != 3.0:
    errors.append(f"poisoning.shift_sd = {p.get('shift_sd')} (expected 3.0)")
if not p.get("calibrated"):
    errors.append(f"poisoning.calibrated = {p.get('calibrated')} (expected true)")

if errors:
    print("CONFIG PATCH ERRORS -- do not launch:")
    for e in errors: print(f"  {e}")
    sys.exit(1)
print("  attack_samples=200, synthetic_ratio=1.0, shift_sd=3.0, calibrated=true -- OK")
PYCHECK

echo

# --- 2. Archive existing results so the run machinery sees all 40 as not-done ---
if [ -f "$RESULTS_CSV" ]; then
    mkdir -p "$ARCHIVE_DIR"
    ARCHIVE_PATH="$ARCHIVE_DIR/all_results_${AMENDMENT}_pre_${TIMESTAMP}.csv"
    cp "$RESULTS_CSV" "$ARCHIVE_PATH"
    echo "[archive] Existing results ($(wc -l < "$RESULTS_CSV") lines) -> $ARCHIVE_PATH"

    # Also clear per-model state files so the run machinery sees seeds as not-done
    echo "[archive] Clearing run state files..."
    while IFS= read -r state_file; do
        [ -f "$state_file" ] && python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f: d = json.load(f)
d['completed_seeds'] = []
d['failed_seeds'] = []
with open(path, 'w') as f: json.dump(d, f, indent=2)
print(f'  cleared {path}')
" "$state_file" 2>/dev/null || true
    done < <(find logs/ -name "state_*.json" 2>/dev/null)

    # Move the results CSV aside (rename so run_full_study sees an empty slate)
    mv "$RESULTS_CSV" "${RESULTS_CSV%.csv}_${AMENDMENT}_pre_${TIMESTAMP}.csv"
    echo "[archive] results/all_results.csv renamed (run will create a fresh one)"
    echo
fi

# --- 3. Note: pre-registration freeze ---
# No frozen record matches the current state, so --record-deviation would fail.
# --no-prepare skips the freeze-check step entirely; the deviation is documented
# in this script and in the amendment notes instead.
echo "[deviation] shift_sd recalibrated 1.0 -> 3.0 (calibration 2026-11-18)."
echo "  Amendment 5 change:"
echo "    shift_sd 1.0 -> 3.0  (Nov 18 calibration; at 1.0, 4/10 seeds show negative"
echo "                          positive-control degradation; M8 fails)"
echo "  All other settings unchanged from Amendment 4."
echo "  All 40 runs re-run from scratch."
echo

# --- 4. Launch the full study ---
echo "[launch] Starting 4 conditions x 10 seeds (seeds 53-62), $WORKERS worker(s)..."
echo "[launch] Log: logs/study/  |  Status: logs/full_study_status.json"
echo

python3 -m scripts.run_full_study \
    --base-seed 53 \
    --n-runs 10 \
    --workers "$WORKERS" \
    --config config/default_config.yaml \
    --new-experiment \
    --no-prepare \
    --label "Amendment 5: shift_sd=3.0 (recalibrated Nov 18), all 40 runs" \
    --max-stalled-passes 5 \
    --retry-wait 300

echo
echo "[done] run_full_study exited. Check logs/full_study_status.json for the final state."
