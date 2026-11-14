<#
.SYNOPSIS  Resume the study that stopped at run 31 of 40 (EXP-20261113-DESKTOPK9L37-0a2a). Keeps every finished run; runs only what is missing.

.EXAMPLE   .\scripts\resume_study.ps1
.EXAMPLE   .\scripts\resume_study.ps1 -ExperimentId EXP-20261113-DESKTOPK9L37-0a2a -BaseSeed 53 -Workers 4

           It refuses to start when (a) a run of the study is still alive (the same seed would run twice at once) or
           (b) poisoning.calibrated is false in the config (the last runs would use a different strength from the first 31).
           It never passes --new-experiment or --refreeze. A different seed plan under the same experiment id is refused by the launcher.
#>
param(
    [string]$ExperimentId = 'EXP-20261113-DESKTOPK9L37-0a2a',
    [int]$BaseSeed = 53,
    [int]$Workers = 4,
    [string]$Deviation = '14 Nov: report fixes and crash fix (progress-file read, dashboard errors ignored); no change to how runs are computed'
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# (a) a run left over from the crashed driver?
$alive = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
           Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'src\.experiments\.run_all' })
if ($alive.Count -gt 0) {
    Write-Host 'A study run is still alive:' -ForegroundColor Yellow
    $alive | ForEach-Object { Write-Host ('  PID {0}: {1}' -f $_.ProcessId, $_.CommandLine) }
    Write-Host 'Wait for it to finish (its row is kept), or end it (Stop-Process -Id <PID>), then run this script again.' -ForegroundColor Yellow
    exit 2
}

# (b) the calibration must be the one the first 31 runs used
$cfgPath = Join-Path $root 'config\default_config.yaml'
$text = [IO.File]::ReadAllText($cfgPath)
$blk = [regex]::Match($text, '(?m)^poisoning:\s*\r?\n((?:[ \t]+.*\r?\n?|\s*\r?\n)*)')
if (-not $blk.Success -or $blk.Groups[1].Value -notmatch '(?m)^[ \t]+calibrated:\s*(true|True)') {
    Write-Host 'poisoning.calibrated is not true in config\default_config.yaml.' -ForegroundColor Red
    Write-Host 'Resuming now would recalibrate and give the last runs a different poisoning strength from the first 31.' -ForegroundColor Red
    Write-Host 'Restore the config this experiment was started with (a .bak copy sits next to it if run_nov14.ps1 was used), then run this again.' -ForegroundColor Red
    exit 3
}

$studyArgs = @('--experiment-id', $ExperimentId, '--base-seed', "$BaseSeed", '--workers', "$Workers", '--record-deviation', $Deviation)
Write-Host ('[resume] .\scripts\run_full_study.ps1 ' + ($studyArgs -join ' ')) -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'run_full_study.ps1') @studyArgs
exit $LASTEXITCODE
