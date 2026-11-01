<#
.SYNOPSIS  Launch the whole confirmatory study once (all conditions x seeds), resumable, then the hypothesis tests.
.EXAMPLE   .\scripts\run_full_study.ps1
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
