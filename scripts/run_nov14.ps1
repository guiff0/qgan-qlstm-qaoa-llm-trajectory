<#
.SYNOPSIS  14 Nov 2026: one command for everything that has to happen on the study machine, in the order that protects the study.

           1. checks that fix_nov14.zip was unzipped over the repo and runs its tests            (about 1 minute)
           2. runs the two pilot models on seed 42 and writes the pilot table                    (LSTM: 15-75 minutes; QAOA: about 3)
           3. recalibrates the poisoning strength under the new rule (every pilot seed must clear the floor)
           4. launches the full study:  .\scripts\run_full_study.ps1 --refreeze --new-experiment --base-seed 53 --workers 4

           Pilots run BEFORE the study, not beside it: both append to results\all_results.csv, and 4 workers already use the machine.
           A pilot that fails is reported and does NOT stop the study. A failed test run or a failed calibration DOES stop it.

.EXAMPLE   .\scripts\run_nov14.ps1
.EXAMPLE   .\scripts\run_nov14.ps1 -SkipPilots                    # study only
.EXAMPLE   .\scripts\run_nov14.ps1 -PilotsOnly                    # pilots and table only, no calibration, no study
.EXAMPLE   .\scripts\run_nov14.ps1 -AllowUncalibrated             # if step 3 finds no stable strength: run anyway, H4/H5 rows are flagged
.EXAMPLE   .\scripts\run_nov14.ps1 -NoRefreeze                    # keep the old frozen record; changed files are then listed as deviations
.EXAMPLE   .\scripts\run_nov14.ps1 -BaseSeed 53 -Workers 4 -Deviation "my own wording"

           Everything printed is also written to logs\run_nov14_<UTC>.log (pilots: logs\pilot_*_<UTC>.log).
#>
param(
    [int]$BaseSeed = 53,
    [int]$Workers = 4,
    [int]$PilotSeed = 42,
    [string]$Deviation = 'Amendment 3: dropped entanglement hypotheses, H4 split into H2/H3, H3 split into H4/H5, duplicate sweep level removed; 14 Nov: report fixes, poisoning calibration requires every pilot seed above the floor',
    [switch]$SkipPilots,
    [switch]$PilotsOnly,
    [switch]$AllowUncalibrated,
    [switch]$NoRefreeze
)
# 'Continue', not 'Stop': under Windows PowerShell 5.1 'Stop' turns any line python writes to stderr into a terminating error.
# Every step below checks its exit code explicitly instead.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. (Join-Path $PSScriptRoot '_python.ps1')
$py = Resolve-ProjectPython $root
$env:PYTHONUNBUFFERED = '1'
$utc = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
$mainLog = Join-Path $root "logs\run_nov14_$utc.log"

function Say([string]$msg, [string]$color = 'Cyan') {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $mainLog -Value $line
}
function Stop-Run([string]$msg, [int]$code = 1) {
    Say "STOPPED: $msg" 'Red'
    exit $code
}

Say "python: $py"
Say "log: $mainLog"

# ---- 1. the patch is in place and its tests pass -------------------------------------------------------------------------------
$needed = @('scripts\pilot_table.py', 'scripts\qaoa_solution_audit.py', 'tests\test_nov14_report_fixes.py',
            'tests\test_pilot_table.py', 'tests\test_delta_pilot_models.py')
$missing = @($needed | Where-Object { -not (Test-Path (Join-Path $root $_)) })
if ($missing.Count -gt 0) { Stop-Run ("fix_nov14.zip is not unzipped over this repo; missing: " + ($missing -join ', ')) 2 }

Say '[1/4] tests of the 14 Nov changes'
& $py -m pytest -q -p no:cacheprovider tests/test_nov14_report_fixes.py tests/test_pilot_table.py tests/test_delta_pilot_models.py 2>&1 |
    Tee-Object -FilePath $mainLog -Append | Select-Object -Last 3 | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { Stop-Run "the tests failed (exit $LASTEXITCODE); nothing else was run. Details in $mainLog" $LASTEXITCODE }

# ---- 2. pilots (seed 42), exact QAOA audit, pilot table -----------------------------------------------------------------------
if (-not $SkipPilots) {
    Say "[2/4] pilot: Classical LSTM, seed $PilotSeed (next-bar change, window-centred inputs; no --reuse-checkpoints)"
    $lstmLog = Join-Path $root "logs\pilot_classical_lstm_$utc.log"
    & $py -m src.experiments.run_all --only "Classical LSTM" --seed $PilotSeed 2>&1 | Tee-Object -FilePath $lstmLog | Out-Null
    if ($LASTEXITCODE -ne 0) { Say "WARNING: the Classical LSTM pilot failed (exit $LASTEXITCODE); see $lstmLog. Continuing." 'Yellow' }
    else { Say "Classical LSTM pilot done; log $lstmLog" }

    Say "[2/4] pilot: QAOA-Enhanced LLM, seed $PilotSeed (feature changes as inputs, ridge head chosen on validation)"
    $qaoaLog = Join-Path $root "logs\pilot_qaoa_enhanced_llm_$utc.log"
    $qaoaOut = & $py -m src.experiments.run_all --only "QAOA-Enhanced LLM" --seed $PilotSeed 2>&1 | Tee-Object -FilePath $qaoaLog
    if ($LASTEXITCODE -ne 0) { Say "WARNING: the QAOA-Enhanced LLM pilot failed (exit $LASTEXITCODE); see $qaoaLog. Continuing." 'Yellow' }
    else {
        Say "QAOA-Enhanced LLM pilot done; log $qaoaLog"
        $m = [regex]::Match(($qaoaOut -join "`n"), 'QAOA selected \d+ of \d+ features \(indices \[([0-9, ]+)\]\)')
        if ($m.Success) {
            $sel = $m.Groups[1].Value -replace '\s', ''
            Say "exact audit of the QAOA answer (all 65,536 possible answers): --selected $sel"
            & $py -m scripts.qaoa_solution_audit --selected $sel --seed $PilotSeed 2>&1 | Tee-Object -FilePath $mainLog -Append | ForEach-Object { Write-Host $_ }
            if ($LASTEXITCODE -ne 0) { Say "WARNING: the QAOA audit failed (exit $LASTEXITCODE). Continuing." 'Yellow' }
        } else {
            Say 'WARNING: could not read the selected feature indices from the pilot output; the QAOA audit was skipped.' 'Yellow'
        }
    }

    Say '[2/4] pilot table (descriptive, one seed, no inference): results\pilot_table.md'
    & $py -m scripts.pilot_table --all-runs 2>&1 | Tee-Object -FilePath $mainLog -Append | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { Say "WARNING: the pilot table was not written (exit $LASTEXITCODE). Continuing." 'Yellow' }
} else {
    Say '[2/4] pilots skipped (-SkipPilots)'
}
if ($PilotsOnly) { Say 'done (-PilotsOnly): no calibration, no study.'; exit 0 }

# ---- 3. poisoning strength: recalibrate under the new rule ---------------------------------------------------------------------
# The 00:46 calibration accepted a strength on the MEDIAN pilot seed and the study then logged a positive control of -15.8 %
# (check M8). The config still says calibrated: true, so a new study would reuse that strength. Reset the flag (inside the
# `poisoning:` block only, line endings kept) and let the calibrate stage choose again.
Say '[3/4] poisoning calibration (new rule: every pilot seed above the floor, at the chosen strength and every larger one)'
$cfgPath = Join-Path $root 'config\default_config.yaml'
$text = [IO.File]::ReadAllText($cfgPath)
$blk = [regex]::Match($text, '(?m)^poisoning:\s*\r?\n((?:[ \t]+.*\r?\n?|\s*\r?\n)*)')
if (-not $blk.Success) { Stop-Run "no 'poisoning:' section in $cfgPath" 2 }
$flag = New-Object System.Text.RegularExpressions.Regex('(?m)^([ \t]+calibrated:\s*)(true|True)')
$old = $blk.Groups[1].Value
$new = $flag.Replace($old, '${1}false', 1)
if ($new -ne $old) {
    Copy-Item $cfgPath "$cfgPath.before_nov14_$utc.bak"
    $text = $text.Substring(0, $blk.Groups[1].Index) + $new + $text.Substring($blk.Groups[1].Index + $blk.Groups[1].Length)
    [IO.File]::WriteAllText($cfgPath, $text, (New-Object System.Text.UTF8Encoding($false)))
    Say "poisoning.calibrated reset to false (backup: $cfgPath.before_nov14_$utc.bak)"
} else {
    Say 'poisoning.calibrated was already false'
}
$extra = @()
if ($AllowUncalibrated) {
    Say 'NOTE: -AllowUncalibrated: the study runs with the provisional strength and every row is flagged poisoning_calibrated=False.' 'Yellow'
    $extra += '--allow-uncalibrated'
} else {
    & $py -m scripts.run_study --only calibrate --force --apply-calibration 2>&1 | Tee-Object -FilePath $mainLog -Append | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        Say 'No stable poisoning strength was found (table above: the "ok" column). The positive control changes sign with the seed,' 'Yellow'
        Say 'so H4 (and H5 where it depends on it) would not be decidable. The study was NOT started. Your choices:' 'Yellow'
        Say '  a) run anyway, rows flagged as uncalibrated:   .\scripts\run_nov14.ps1 -SkipPilots -AllowUncalibrated' 'Yellow'
        Say '  b) fix the positive control first (which rows its head trains on) and record that as an amendment.' 'Yellow'
        Stop-Run "calibration exit code $LASTEXITCODE" $LASTEXITCODE
    }
}

# ---- 4. the full study ---------------------------------------------------------------------------------------------------------
$studyArgs = @('--new-experiment', '--base-seed', "$BaseSeed", '--workers', "$Workers", '--record-deviation', $Deviation) + $extra
if (-not $NoRefreeze) { $studyArgs = @('--refreeze') + $studyArgs }      # code changed since the old record: take a new one (the old is kept)
Say ('[4/4] .\scripts\run_full_study.ps1 ' + ($studyArgs -join ' '))
& (Join-Path $PSScriptRoot 'run_full_study.ps1') @studyArgs 2>&1 | Tee-Object -FilePath $mainLog -Append
$code = $LASTEXITCODE
if ($code -eq 0) { Say "study finished (exit 0). Full console output: $mainLog" } else { Say "study ended with exit code $code. Full console output: $mainLog" 'Yellow' }
exit $code
