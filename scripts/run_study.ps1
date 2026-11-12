<#
.SYNOPSIS  ONE entry point for the whole study (calibrate, freeze check, runs, three-arm feature map, matrix, hypothesis tests).
.EXAMPLE   .\scripts\run_study.ps1 --plan
.EXAMPLE   .\scripts\run_study.ps1 --new-experiment --label "Amendment 2" --n-runs 10 --base-seed 53 --workers 4 --apply-calibration --record-deviation "Amendment 2"
.EXAMPLE   .\scripts\run_study.ps1            # resume
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. (Join-Path $PSScriptRoot '_python.ps1')
$py = Resolve-ProjectPython $root
& $py -m scripts.run_study @args
exit $LASTEXITCODE
