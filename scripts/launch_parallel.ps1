<#
.SYNOPSIS
  Runs each model's train + evaluate step in its OWN PowerShell window, several at a time.

.DESCRIPTION
  Equivalent to the train_eval steps of run_pipeline.py, but every model gets a separate,
  visible, detached window (its output is also saved to logs\parallel_<model>.txt), so one slow
  model (QGAN-LLM, 20 qubits) does not hold up the others. Safe to run concurrently: results\all_results.csv
  is written under a cross-process file lock (src/utils/io.py file_lock), and LLM replies are cached per prompt.

  Slowest models start first. At most -MaxParallel windows run at once. Torch/BLAS threads per window are
  set to cores / MaxParallel so the windows do not fight over the CPU.

  Prerequisites: data prepared (python -m scripts.prepare_data), $env:NVIDIA_API_KEY set,
  and `python scripts\check_llm_endpoint.py` passed.

.EXAMPLE
  .\scripts\launch_parallel.ps1 -MaxParallel 4
  .\scripts\launch_parallel.ps1 -MaxParallel 3 -KeepOpen     # leave the windows open after each finishes
  .\scripts\launch_parallel.ps1 -Models "QLSTM Forecaster","Classical LSTM"
#>
param(
    [int]$MaxParallel = 4,
    [switch]$KeepOpen,
    [string[]]$Models = @(
        'QGAN-LLM', 'QGAN-LLM (12-qubit)', 'QGAN-LLM (linear topology)', 'QGAN-LLM (no noise)',
        'QLSTM Forecaster', 'QLSTM-LLM', 'QAOA-LLM', 'Classical-Features-LLM', 'Classical GAN-LLM',
        'QAOA-Enhanced LLM', 'LLM Forecaster', 'Classical LSTM'
    )
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Python not found at $py (activate/create the venv first)." }
if (-not $env:NVIDIA_API_KEY) {
    Write-Warning 'NVIDIA_API_KEY is not set: the LLM stage and the LLM Forecaster will be skipped.'
}
$threads = [Math]::Max(1, [int][Math]::Floor([Environment]::ProcessorCount / $MaxParallel))
$env:OMP_NUM_THREADS = "$threads"; $env:MKL_NUM_THREADS = "$threads"; $env:OPENBLAS_NUM_THREADS = "$threads"
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

$running = @()
foreach ($m in $Models) {
    while (@($running | Where-Object { -not $_.Proc.HasExited }).Count -ge $MaxParallel) { Start-Sleep -Seconds 5 }
    $slug = ($m -replace '[^A-Za-z0-9]+', '_').Trim('_')
    $log = Join-Path $root "logs\parallel_$slug.txt"
    $inner = "Set-Location '$root'; `$host.UI.RawUI.WindowTitle = '$m'; " +
             "& '$py' -m src.experiments.run_all --only '$m' *>&1 | Tee-Object -FilePath '$log'; " +
             "`$code = `$LASTEXITCODE; Write-Host \"[exit code `$code]\"; exit `$code"
    $args = @('-NoProfile') + $(if ($KeepOpen) { @('-NoExit') } else { @() }) + @('-Command', $inner)
    $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $args -PassThru -WindowStyle Normal
    $null = $p.Handle   # keep the handle so ExitCode stays readable after exit
    $running += [pscustomobject]@{ Model = $m; Proc = $p; Log = $log }
    Write-Host ("[{0}] started: {1}  (pid {2}, log {3})" -f (Get-Date -Format 'HH:mm:ss'), $m, $p.Id, $log)
}

Write-Host "`nAll windows started; waiting for them to finish..."
$running | ForEach-Object { $_.Proc.WaitForExit() }
$failed = @($running | Where-Object { $_.Proc.ExitCode -ne 0 })
$running | ForEach-Object { Write-Host ("{0,-32} exit {1}" -f $_.Model, $_.Proc.ExitCode) }
if ($failed.Count -gt 0) {
    Write-Host "`n$($failed.Count) model(s) failed; see their logs\parallel_*.txt. Not running the verification step."
    exit 1
}
Write-Host "`nAll models finished. Verifying and testing..."
& $py -m scripts.verify_chapter4
& $py -m scripts.run_hypothesis_tests
