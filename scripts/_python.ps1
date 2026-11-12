# Shared by the launchers: find the project's Python without assuming one folder name.
# Order: $env:QGAN_PYTHON, the ACTIVATED venv ($env:VIRTUAL_ENV), <repo>\venv, .venv, env, then `python` / `py -3.11` on PATH.
# A candidate must actually run (the Microsoft Store 'python' stub is rejected) and import numpy.
function Test-Py([string]$exe) {
    if (-not $exe) { return $false }
    try {
        $out = & $exe -c "import sys, numpy; print(sys.version_info[0])" 2>$null
        return ($LASTEXITCODE -eq 0 -and "$out".Trim() -eq '3')
    } catch { return $false }
}
function Resolve-ProjectPython([string]$root) {
    $cands = @()
    if ($env:QGAN_PYTHON) { $cands += $env:QGAN_PYTHON }
    if ($env:VIRTUAL_ENV) { $cands += (Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe') }
    foreach ($d in 'venv', '.venv', 'env') { $cands += (Join-Path $root "$d\Scripts\python.exe") }
    foreach ($c in $cands) { if ((Test-Path $c) -and (Test-Py $c)) { return $c } }
    foreach ($n in 'python', 'python3') {
        $g = Get-Command $n -ErrorAction SilentlyContinue | Where-Object { $_.Source -and $_.Source -notmatch 'WindowsApps' } | Select-Object -First 1
        if ($g -and (Test-Py $g.Source)) { return $g.Source }
    }
    $pyl = Get-Command py -ErrorAction SilentlyContinue
    if ($pyl) {
        foreach ($v in '-3.11', '-3') {
            $o = & $pyl.Source $v -c "import numpy, sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $o) { return "$o".Trim() }
        }
    }
    $tried = ($cands | ForEach-Object { "  $_" }) -join "`n"
    throw ("No working Python (with numpy) found. Tried:`n$tried`n" +
           "and 'python' / 'py' on PATH.`nFix: run  .\rebuild_env.ps1  (creates venv\ and installs the requirements), " +
           "or activate your environment first, or set  `$env:QGAN_PYTHON = 'C:\path\to\python.exe'.")
}
