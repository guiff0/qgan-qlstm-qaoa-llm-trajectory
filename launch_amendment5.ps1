# =============================================================================
# launch_amendment5.ps1 — Amendment 5 rerun (Windows PowerShell)
#
# WHAT THIS DOES
# ==============
# The 40 runs completed under Amendment 4 (experiment EXP-20261116-MAIN-f41c)
# used shift_sd=1.0 and an attack with a mean-zero sign bug. Dashboard showed:
#   [FAIL] M8_poisoning_positive_control  min degradation -48.7% vs floor 5.0%
#
# ROOT CAUSE (diagnosed 2026-11-18):
#   inject_triggered_shift used sign(z[idx] - median(z)) as a multiplier,
#   producing a ~50/50 mixed +-shift. A ridge head sees no consistent adversarial
#   direction, so degradation sign is random across seeds. shift_sd alone cannot fix.
#
# FIX (src/attacks/poisoning.py, 2026-11-18):
#   1. Direction u is now adversarial: u = X.T @ residuals from a clean ridge fit,
#      guaranteeing u opposes the fitted model regardless of seed.
#   2. All selected rows shifted in the same (+) direction (no sign flip).
#   3. Recalibration (results/poisoning_calibration_20261118T192121Z.json) chose
#      shift_sd = 2.0 (median +67.5%, min +67.5%, deterministic across all seeds).
#
# Config patches applied to config/default_config.yaml:
#   shift_sd: 3.0 -> 2.0  (matches recalibration of 2026-11-18)
#   calibrated: true       (unchanged)
#
# USAGE
# =====
#   cd D:\dev\app\qgan-qlstm-qaoa-llm-trajectory
#   .\launch_amendment5.ps1           # 1 worker (safe on any machine)
#   .\launch_amendment5.ps1 -Workers 2
#   .\launch_amendment5.ps1 -Workers 4
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
# Safe to kill and relaunch — the script resumes from whatever is done.
# =============================================================================

param(
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$ResultsCsv    = "results\all_results.csv"
$ArchiveDir    = "results\archive"
$Amendment     = "Amendment-5"
$Timestamp     = (Get-Date -Format "yyyyMMddTHHmmss")

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " AMENDMENT 5 LAUNCH  --  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') local" -ForegroundColor Cyan
Write-Host " Workers: $Workers" -ForegroundColor Cyan
Write-Host " Config:  config/default_config.yaml" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# STEP 1 — Verify config patches
# ---------------------------------------------------------------------------
Write-Host "[check] Verifying config patches..." -ForegroundColor Yellow

python -c @"
import yaml, sys
with open('config/default_config.yaml') as f:
    cfg = yaml.safe_load(f)

errors = []
for section in ('classical_gan_llm', 'qgan_llm'):
    s = cfg[section]
    if s.get('attack_samples') != 200:
        errors.append(f'{section}.attack_samples = {s.get(\"attack_samples\")} (expected 200)')
    if s.get('synthetic_ratio') != 1.0:
        errors.append(f'{section}.synthetic_ratio = {s.get(\"synthetic_ratio\")} (expected 1.0)')

p = cfg.get('poisoning', {})
if p.get('shift_sd') != 2.0:
    errors.append(f'poisoning.shift_sd = {p.get(\"shift_sd\")} (expected 2.0)')
if not p.get('calibrated'):
    errors.append(f'poisoning.calibrated = {p.get(\"calibrated\")} (expected true)')

if errors:
    print('CONFIG PATCH ERRORS -- do not launch:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
print('  attack_samples=200, synthetic_ratio=1.0, shift_sd=2.0, calibrated=true -- OK')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Config check FAILED. Apply patches before launching." -ForegroundColor Red
    exit 1
}
Write-Host ""

# ---------------------------------------------------------------------------
# STEP 2 — Archive existing results
# ---------------------------------------------------------------------------
if (Test-Path $ResultsCsv) {
    if (-not (Test-Path $ArchiveDir)) {
        New-Item -ItemType Directory -Path $ArchiveDir | Out-Null
    }
    $ArchivePath = "$ArchiveDir\all_results_${Amendment}_pre_${Timestamp}.csv"
    Copy-Item $ResultsCsv $ArchivePath
    $LineCount = (Get-Content $ResultsCsv | Measure-Object -Line).Lines
    Write-Host "[archive] Existing results ($LineCount lines) -> $ArchivePath" -ForegroundColor Yellow

    # Clear per-model state files so run machinery sees seeds as not-done
    Write-Host "[archive] Clearing run state files..." -ForegroundColor Yellow
    Get-ChildItem -Path "logs" -Recurse -Filter "state_*.json" -ErrorAction SilentlyContinue | ForEach-Object {
        python -c @"
import json, sys
path = r'$($_.FullName)'
with open(path) as f:
    d = json.load(f)
d['completed_seeds'] = []
d['failed_seeds'] = []
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
print(f'  cleared {path}')
"@
    }

    # Rename results CSV so run_full_study starts fresh
    $ArchivedCsv = $ResultsCsv -replace '\.csv$', "_${Amendment}_pre_${Timestamp}.csv"
    Rename-Item -Path $ResultsCsv -NewName (Split-Path $ArchivedCsv -Leaf)
    Write-Host "[archive] results\all_results.csv renamed (run will create a fresh one)" -ForegroundColor Yellow
    Write-Host ""
}

# ---------------------------------------------------------------------------
# STEP 3 — Log the deviation
# ---------------------------------------------------------------------------
Write-Host "[deviation] Attack direction bug fixed + shift_sd recalibrated (2026-11-18)." -ForegroundColor Yellow
Write-Host "  Amendment 5 changes:"
Write-Host "    inject_triggered_shift: mean-zero sign bug fixed; direction is now X.T@residuals"
Write-Host "    shift_sd 3.0 -> 2.0  (recalibration 2026-11-18; deterministic +67.5% all seeds)"
Write-Host "    Old: mean-zero sign-mixed shift -> random degradation sign per seed"
Write-Host "    New: unidirectional adversarial shift -> guaranteed positive degradation"
Write-Host "  All other settings unchanged from Amendment 4."
Write-Host "  All 40 runs re-run from scratch."
Write-Host ""

# ---------------------------------------------------------------------------
# STEP 4 — Launch full study
# ---------------------------------------------------------------------------
Write-Host "[launch] Starting 4 conditions x 10 seeds (seeds 53-62), $Workers worker(s)..." -ForegroundColor Green
Write-Host "[launch] Log: logs\study\  |  Status: logs\full_study_status.json" -ForegroundColor Green
Write-Host ""

python -m scripts.run_full_study `
    --base-seed 53 `
    --n-runs 10 `
    --workers $Workers `
    --config config/default_config.yaml `
    --new-experiment `
    --no-prepare `
    --label "Amendment 5: attack direction fix + shift_sd=2.0 (recalibrated Nov 18), all 40 runs" `
    --max-stalled-passes 5 `
    --retry-wait 300

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "run_full_study exited with non-zero status. Check logs above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host " Done. Check logs\full_study_status.json for final state." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
