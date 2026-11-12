# Arm 1 (naive baseline) only; safe beside a running study. Usage: .\scripts\run_naive_baseline.ps1 [--status] [--wait-for-others] ...
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. (Join-Path $PSScriptRoot '_python.ps1')
$py = Resolve-ProjectPython $root
& $py -m scripts.run_naive_baseline @args
exit $LASTEXITCODE
