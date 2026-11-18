# =============================================================================
# run_confirmatory_gate.ps1
# Amendment 4 confirmatory gate — run from the repo root on Windows PowerShell.
#
# Step 1: freeze_preregistration --check   (no unrecorded changes)
# Step 2: snapshot_results --verify        (csv matches recorded snapshot)
# Step 3: run_study --only hypothesis --force --run-confirmatory
#
# DO NOT edit code or data after seeing the output.
# A second run requires --rerun-reason and is disclosed in the output.
# =============================================================================

$ErrorActionPreference = "Stop"
$EXPERIMENT_ID = "EXP-20261116-VM-7fdd"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Amendment 4 Confirmatory Gate" -ForegroundColor Cyan
Write-Host " Experiment: $EXPERIMENT_ID" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# STEP 1 — Preregistration integrity check
# ---------------------------------------------------------------------------
Write-Host "[1/3] Checking preregistration freeze..." -ForegroundColor Yellow
python -m scripts.freeze_preregistration --check
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "GATE FAILED: unrecorded changes detected since the freeze." -ForegroundColor Red
    Write-Host "Disclose each change with:" -ForegroundColor Red
    Write-Host "  python -m scripts.freeze_preregistration --record-deviation `"<reason>`"" -ForegroundColor Red
    exit 1
}
Write-Host "  OK — no unrecorded changes." -ForegroundColor Green
Write-Host ""

# ---------------------------------------------------------------------------
# STEP 2 — Snapshot / data integrity check
# ---------------------------------------------------------------------------
Write-Host "[2/3] Verifying results snapshot..." -ForegroundColor Yellow
python -m scripts.snapshot_results --verify
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "GATE FAILED: results/all_results.csv does not match any recorded snapshot." -ForegroundColor Red
    Write-Host "If data collection is complete, take a fresh snapshot first:" -ForegroundColor Red
    Write-Host "  python -m scripts.snapshot_results --note `"Amendment 4 runs complete`"" -ForegroundColor Red
    Write-Host "Then re-run this script." -ForegroundColor Red
    exit 1
}
Write-Host "  OK — csv matches recorded snapshot." -ForegroundColor Green
Write-Host ""

# ---------------------------------------------------------------------------
# STEP 3 — Confirmatory hypothesis tests
# ---------------------------------------------------------------------------
Write-Host "[3/3] Running confirmatory hypothesis tests..." -ForegroundColor Yellow
Write-Host "  Driver: python -m scripts.run_study --only hypothesis --force --run-confirmatory --experiment-id $EXPERIMENT_ID"
Write-Host ""
python -m scripts.run_study `
    --only hypothesis `
    --force `
    --run-confirmatory `
    --experiment-id $EXPERIMENT_ID

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Hypothesis tests exited with non-zero status." -ForegroundColor Red
    Write-Host "Check output above. Do NOT re-run without --rerun-reason." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Gate complete. Results in results/hypothesis_test_results.json" -ForegroundColor Green
Write-Host " DO NOT edit code or data after this point." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
