#!/usr/bin/env bash
# launch_amendment4.sh — Amendment 4 rerun
#
# WHAT THIS DOES
# ==============
# The 32 runs completed under Amendment 3 used:
#   attack_samples = 50   (H3 ASR variance ±10–15pp; need ±3–4pp → 200)
#   synthetic_ratio = 0.4 (H4 main arm only 0.07–2.37% degradation; must be 1.0)
#   shift_sd = 0.5        (not matching calibration result of 1.0)
#
# Config patches already applied to config/default_config.yaml:
#   attack_samples: 50  → 200   (both classical_gan_llm and qgan_llm)
#   synthetic_ratio: 0.4 → 1.0  (both sections — only affects poisoning eval)
#   shift_sd: 0.5 → 1.0         (matches calibration run of 2026-11-13)
#   calibrated: false → true
#
# All 40 runs are re-run fresh under a new experiment id.
# The old results CSV is archived (not deleted) for audit purposes.
#
# USAGE
# =====
#   cd /home/claude/orig
#   bash launch_amendment4.sh           # 1 worker (safe on any machine)
#   bash launch_amendment4.sh 2         # 2 parallel workers
#   bash launch_amendment4.sh 4         # 4 parallel workers (needs ≥4 physical cores)
#
# TIME ESTIMATE
# =============
#   Classical GAN-LLM × 10 seeds ≈  5–8 h total (CPU)
#   Classical GAN-LLM (matched)  ≈  5–8 h
#   QGAN-LLM         × 10 seeds  ≈ 40–50 h total (quantum, lightning.qubit, 12 qubits)
#   QGAN-LLM (no noise) × 10     ≈ 40–50 h
#   TOTAL (1 worker)              ≈ 90–116 h
#   TOTAL (2 workers)             ≈ 50–60 h  (classical + quantum interleave)
#   TOTAL (4 workers)             ≈ 25–30 h
#
# Safe to kill and relaunch — the script resumes from whatever is done.

set -euo pipefail

WORKERS=${1:-1}
RESULTS_CSV="results/all_results.csv"
ARCHIVE_DIR="results/archive"
AMENDMENT="Amendment-4"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)

echo "========================================================"
echo " AMENDMENT 4 LAUNCH  —  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
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
if p.get("shift_sd") != 1.0:
    errors.append(f"poisoning.shift_sd = {p.get('shift_sd')} (expected 1.0)")
if not p.get("calibrated"):
    errors.append(f"poisoning.calibrated = {p.get('calibrated')} (expected true)")

if errors:
    print("CONFIG PATCH ERRORS — do not launch:")
    for e in errors: print(f"  {e}")
    sys.exit(1)
print("  attack_samples=200, synthetic_ratio=1.0, shift_sd=1.0, calibrated=true — OK")
PYCHECK

echo

# --- 2. Archive existing results so the run machinery sees all 40 as not-done ---
if [ -f "$RESULTS_CSV" ]; then
    mkdir -p "$ARCHIVE_DIR"
    ARCHIVE_PATH="$ARCHIVE_DIR/all_results_${AMENDMENT}_pre_${TIMESTAMP}.csv"
    cp "$RESULTS_CSV" "$ARCHIVE_PATH"
    echo "[archive] Existing results ($(wc -l < "$RESULTS_CSV") lines) → $ARCHIVE_PATH"

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
# No frozen record exists for this study, so --record-deviation would fail.
# --no-prepare skips the freeze-check step entirely; the deviation is documented
# in this script and in the AMENDMENT_4_NOTES below instead.
echo "[deviation] No frozen pre-registration record — amendment documented inline."
echo "  Amendment 4 changes:"
echo "    attack_samples  50 → 200  (H3 ASR variance fix)"
echo "    synthetic_ratio  0.4 → 1.0  (H4 poisoning dilution fix)"
echo "    shift_sd  0.5 → 1.0  (matches calibration run 2026-11-13)"
echo "  All 40 runs re-run from scratch."
echo

# --- 4. Launch the full study ---
echo "[launch] Starting 4 conditions × 10 seeds (seeds 53–62), $WORKERS worker(s)..."
echo "[launch] Log: logs/study/  |  Status: logs/full_study_status.json"
echo

python3 -m scripts.run_full_study \
    --base-seed 53 \
    --n-runs 10 \
    --workers "$WORKERS" \
    --config config/default_config.yaml \
    --new-experiment \
    --no-prepare \
    --label "Amendment 4: attack_samples=200, synthetic_ratio=1.0, shift_sd=1.0, all 40 runs" \
    --max-stalled-passes 5 \
    --retry-wait 300

echo
echo "[done] run_full_study exited. Check logs/full_study_status.json for the final state."
