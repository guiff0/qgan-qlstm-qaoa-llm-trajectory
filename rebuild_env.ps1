<#
.SYNOPSIS
  Idempotent environment (re)build for the QGAN-LLM project.

.DESCRIPTION
  Reuses an existing healthy Python 3.11 venv and installs ONLY packages that are
  missing or at a version that does not satisfy requirements.txt.
  Use -Force to delete and recreate the venv from scratch.

.EXAMPLE
  .\rebuild_env.ps1            # incremental: skip everything already satisfied
  .\rebuild_env.ps1 -Force     # wipe venv and rebuild
  .\rebuild_env.ps1 -SkipPrepare   # environment only, do not run prepare_data

  After the environment is verified this also runs `python -m scripts.prepare_data` (raw files -> data\processed).
  It needs the raw files from scripts.acquire_all_data. Before preparing, it ASKS whether ingestion is needed:
  raw data missing -> "run ingestion?"; raw data present -> "re-run ingestion to refresh?". Default is No, so the
  ~90-minute download never starts on its own. -AcquireData answers yes without asking; -NoPrompt answers no.
#>
param([switch]$Force, [switch]$SkipPrepare, [switch]$AcquireData, [switch]$NoPrompt)

$ErrorActionPreference = "Continue"   # native stderr (pip warnings) must not abort; Invoke-Step checks exit codes
Write-Host "=== QGAN-LLM Environment Rebuild Script ===" -ForegroundColor Cyan

# Run a native command and abort if it fails (pip errors don't throw by themselves).
function Invoke-Step([string]$What, [scriptblock]$Cmd) {
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED: $What (exit code $LASTEXITCODE). Aborting."
        exit 1
    }
}

# -----------------------------
# 1. Locate / install Python 3.11
# -----------------------------
Write-Host "`nChecking Python 3.11 installation..."
$python311 = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe"

if (-Not (Test-Path $python311)) {
    $found = $null
    try { $found = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null) } catch { }
    if (-Not $found) { $cmd = Get-Command python3.11 -ErrorAction SilentlyContinue; if ($cmd) { $found = $cmd.Path } }

    if ($found -and (Test-Path $found)) {
        $python311 = $found
    } else {
        Write-Host "Python 3.11 not found. Installing via winget..." -ForegroundColor Yellow
        winget install -e --id Python.Python.3.11 --silent
        if (-Not (Test-Path $python311)) {
            $cmd = Get-Command python3.11 -ErrorAction SilentlyContinue
            if ($cmd) { $python311 = $cmd.Path }
        }
    }
}
if (-Not $python311 -or -Not (Test-Path $python311)) {
    Write-Error "Python 3.11 executable could not be found. Please install Python 3.11 manually."
    exit 1
}
Write-Host "Using Python: $python311" -ForegroundColor Green

# -----------------------------
# 2. Reuse the venv if healthy, otherwise (re)create it
# -----------------------------
$venvPython = ".\venv\Scripts\python.exe"
$reuse = $false

if ((-Not $Force) -and (Test-Path $venvPython)) {
    $ver = & $venvPython -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ver -eq "3.11") {
        $reuse = $true
        Write-Host "`nExisting venv (Python $ver) is healthy -- reusing it." -ForegroundColor Green
    } else {
        Write-Host "`nExisting venv is broken or not Python 3.11 -- recreating." -ForegroundColor Yellow
    }
}

if (-Not $reuse) {
    if (Test-Path ".\venv") {
        Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".\venv"
    }
    Write-Host "Creating new virtual environment using Python 3.11..." -ForegroundColor Cyan
    Invoke-Step "create venv" { & $python311 -m venv venv }
}

# -----------------------------
# 3. Activate the venv
# -----------------------------
# Run this script dot-sourced / in-process if you want the activation to persist afterwards.
Write-Host "`nActivating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"
if (-Not $env:VIRTUAL_ENV) {
    Write-Error "Venv activation failed -- `$env:VIRTUAL_ENV was not set. Aborting."
    exit 1
}
Write-Host "Venv active: $env:VIRTUAL_ENV" -ForegroundColor Green

# -----------------------------
# 4. pip tooling (only when the venv is new)
# -----------------------------
if (-Not $reuse) {
    Write-Host "`nUpgrading pip, setuptools, wheel..." -ForegroundColor Cyan
    Invoke-Step "upgrade pip" { python -m pip install --upgrade pip setuptools wheel }
} else {
    Write-Host "`npip tooling: skipped (existing venv)." -ForegroundColor Green
}

# -----------------------------
# 5. NumPy 2.x (installed before everything else; must not be downgraded later)
# -----------------------------
python scripts/check_requirements.py --numpy2
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nNumPy >= 2 already installed -- skipping." -ForegroundColor Green
} else {
    Write-Host "`nInstalling NumPy 2.x (binary wheel)..." -ForegroundColor Cyan
    Invoke-Step "install numpy" { pip install "numpy>=2.0,<3" --only-binary=:all: }
}

# -----------------------------
# 6. PyTorch (CUDA 12.1 build)
# -----------------------------
python scripts/check_requirements.py --torch-cuda 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nPyTorch with CUDA already installed -- skipping." -ForegroundColor Green
} else {
    Write-Host "`nInstalling PyTorch (cu121)..." -ForegroundColor Cyan
    Invoke-Step "install torch" { pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 }
}

# -----------------------------
# 7. Everything else: install only unsatisfied requirements
#    (scientific stack, PennyLane, data ingestion: yfinance / dukascopy-python /
#     histdata / pandas-datareader, LLM stack, ...)
# -----------------------------
Write-Host "`nChecking requirements.txt against installed packages..." -ForegroundColor Cyan
python scripts/check_requirements.py --report --skip torch

$todo = @(python scripts/check_requirements.py --skip torch | Where-Object { $_.Trim() -ne "" })
if ($todo.Count -eq 0) {
    Write-Host "`nAll requirements already satisfied -- nothing to install." -ForegroundColor Green
} else {
    Write-Host "`nInstalling $($todo.Count) missing/outdated package(s):" -ForegroundColor Yellow
    $todo | ForEach-Object { Write-Host "  $_" }
    # Quote each spec so '>=' is not treated as a redirect by PowerShell.
    Invoke-Step "install missing requirements" { pip install @todo }
}

# -----------------------------
# 8. Verify
# -----------------------------
Write-Host "`nRunning environment verification..." -ForegroundColor Cyan

# PyPI "dukascopy-python" is imported as `dukascopy_python` (no top-level `dukascopy`).
# Known package bug: its __init__.py annotates with `requests.Session` without importing requests, so the
# import raises NameError even when requests is installed. Reinstalling cannot fix it, and
# scripts/acquire_all_data.py falls back to its own Dukascopy client, so this is a warning only.
$null = python -c "import dukascopy_python" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "dukascopy_python does not import (known package bug, not a missing dependency); acquire_all_data.py uses its direct Dukascopy client instead. Continuing." -ForegroundColor Yellow
    $global:LASTEXITCODE = 0
}

# --env-only skips raw/processed data-file checks (they don't exist right after a rebuild).
# Run `python -m scripts.verify_environment` after data acquisition + prepare_data.
python -m scripts.verify_environment --env-only

# -----------------------------
# 9. Prepare data (raw -> data\processed)
# -----------------------------
if ($SkipPrepare) {
    Write-Host "`nSkipping prepare_data (-SkipPrepare)." -ForegroundColor Yellow
} else {
    # Paths come from config/default_config.yaml (anchored at the repo root by src/utils/config.py).
    $rawPaths = @(python -c "from src.utils.config import load_config; d = load_config()['data']; print(d['dukascopy_file']); print(d['fred_file'])")
    $missing = @($rawPaths | Where-Object { $_ -and -not (Test-Path $_) })

    # Ask whether ingestion is needed (default No). acquire_all_data reuses its per-day cache and returns
    # quickly when the master CSV already covers the study window, so answering yes on present data is cheap.
    $ingest = [bool]$AcquireData
    if (-not $ingest -and -not $NoPrompt) {
        if ($missing.Count -gt 0) {
            $q = "Raw data missing ($($missing.Count) file(s)). Run ingestion now (acquire_all_data, can take ~90 min)? [y/N]"
        } else {
            $q = "Raw data already present. Re-run ingestion to refresh/extend it? [y/N]"
        }
        $ans = Read-Host $q
        $ingest = ($ans -match '^\s*(y|yes)\s*$')
    }
    if ($ingest) {
        Write-Host "`nRunning acquire_all_data..." -ForegroundColor Cyan
        Invoke-Step "acquire_all_data" { python -m scripts.acquire_all_data }
        $missing = @($rawPaths | Where-Object { $_ -and -not (Test-Path $_) })
    }

    if ($missing.Count -gt 0) {
        Write-Host "`nSkipping prepare_data: raw data file(s) not found:" -ForegroundColor Yellow
        $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        Write-Host "Run  python -m scripts.acquire_all_data  (or rerun this script and answer y, or pass -AcquireData), then  python -m scripts.prepare_data" -ForegroundColor Yellow
    } else {
        Write-Host "`nRunning prepare_data..." -ForegroundColor Cyan
        Invoke-Step "prepare_data" { python -m scripts.prepare_data }
        Write-Host "Running full verification (including data files)..." -ForegroundColor Cyan
        python -m scripts.verify_environment
    }
}

Write-Host "`n=== Environment rebuild complete ===" -ForegroundColor Green
