<#
.SYNOPSIS  Launch the whole confirmatory study once (all conditions x seeds), resumable, then the hypothesis tests.
           FIRST it calibrates H3 (python -m scripts.run_study --only calibrate --apply-calibration) and checks the
           pre-registration freeze; add --record-deviation "<reason>" to record an intended Amendment.
.EXAMPLE   .\scripts\run_full_study.ps1 --workers 4          # DEFAULT: reduced 25 runs (5 conditions x 5 seeds)
.EXAMPLE   .\scripts\run_full_study.ps1 --full --workers 4   # pre-registered 8 x 10 = 80 runs
.EXAMPLE   .\scripts\run_full_study.ps1 --n-runs 8 --levels all
.EXAMPLE   .\scripts\run_full_study.ps1 --status
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Python not found at $py (activate/create the venv first)." }
& $py -m scripts.run_full_study @args
exit $LASTEXITCODE
