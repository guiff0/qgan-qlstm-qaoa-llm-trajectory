<#
.SYNOPSIS  Launch the whole confirmatory study once (all conditions x seeds), resumable, then the hypothesis tests.
           FIRST it calibrates H3 (python -m scripts.run_study --only calibrate --apply-calibration) and checks the
           pre-registration freeze; add --record-deviation "<reason>" to record an intended Amendment.
.EXAMPLE   .\scripts\run_full_study.ps1 --workers 4          # DEFAULT: 4 conditions x 10 seeds = 40 runs; --quick = 16-run smoke test
.EXAMPLE   .\scripts\run_full_study.ps1 --full --workers 4   # same 40 runs (the sweep was dropped)
.EXAMPLE   .\scripts\run_full_study.ps1 --n-runs 10 --levels all   # optional exploratory sweep
.EXAMPLE   .\scripts\run_full_study.ps1 --status
.EXAMPLE   .\scripts\run_full_study.ps1 --skip-three-arm --workers 4   # study only: do not (re)run the two three-arm steps below
.EXAMPLE   .\scripts\run_full_study.ps1 --with-12q --workers 4          # Q1 at 12 qubits too, as PREREGISTRATION.md names (about 20x the cost of 8)
.EXAMPLE   .\scripts\run_full_study.ps1 --skip-freeze --workers 4        # do not freeze / check the pre-registration first
.EXAMPLE   .\scripts\run_full_study.ps1 --redo-three-arm --workers 4   # refit the walk-forward file and the naive baseline from scratch

           BEFORE the study (13 Nov 2026) it runs the three three-arm steps the hypothesis tests read at the end:
             1. python -m scripts.run_walk_forward      Arms 2 and 3 (classical / quantum features), skipped if the file is already
                                                        complete for this plan with the current ridge grid
             2. python -m scripts.run_naive_baseline    Arm 1 (inputs only), same windows, seeds, grid and attack rows; resumable
             3. python -m scripts.run_feature_map_experiment --seeds 0..9 --n-qubits 8
                                                        the paired 10-seed test behind the Q1 verdict (needs >= 10 seeds, else UNDERPOWERED);
                                                        skipped if the file is already complete. Hours of CPU: the quantum arm is simulated.
             With --with-12q the same three steps also run at 12 qubits: walk_forward_q12.json, naive_baseline\legacy_q12, and
             feature_map_advantage_q12.json (about 12 h + 29 h more). The hypothesis tests report each width separately.
           All three are skipped for --status, --dry-run, --help and --quick. None depends on the GAN runs.
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. (Join-Path $PSScriptRoot '_python.ps1')
$py = Resolve-ProjectPython $root
$env:PYTHONUNBUFFERED = '1'          # lines reach the timestamping filter as they are printed, not in blocks
$t0 = Get-Date

function Get-Stamp { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }
function Write-Stamp([string]$msg, [string]$color = 'Gray') { Write-Host ("[{0}] {1}" -f (Get-Stamp), $msg) -ForegroundColor $color }
# run `python -m <module> <args>` and put a timestamp in front of every line it prints; $LASTEXITCODE is the python exit code
function Invoke-Stamped {
    $mod = $args[0]; $rest = @($args | Select-Object -Skip 1)
    & $py -m $mod @rest 2>&1 | ForEach-Object { if ("$_" -match '^\[\d{4}-\d\d-\d\d ') { "$_" } else { "[{0}] {1}" -f (Get-Stamp), "$_" } }
}

# --- three-arm steps (own switches are removed before the rest is passed to run_full_study) -------------------------------------
$skipThree = $args -contains '--skip-three-arm'
$redoThree = $args -contains '--redo-three-arm'
$with12 = $args -contains '--with-12q'
$skipFreeze = $args -contains '--skip-freeze'
$studyArgs = @($args | Where-Object { $_ -ne '--skip-three-arm' -and $_ -ne '--redo-three-arm' -and $_ -ne '--with-12q' -and $_ -ne '--skip-freeze' })
$noRun = @('--status', '--dry-run', '--help', '-h', '--quick') | Where-Object { $studyArgs -contains $_ }

# --- pre-registration freeze: BEFORE any run, so margins and decision rules cannot drift (skipped with --skip-freeze) ------------
if (-not $skipFreeze -and -not $noRun) {
    $frozen = Get-ChildItem -ErrorAction SilentlyContinue (Join-Path $root 'results\preregistration_*.json')
    if (-not $frozen) {
        Write-Stamp '[freeze] no frozen record yet: python -m scripts.freeze_preregistration'
        Invoke-Stamped scripts.freeze_preregistration
        if ($LASTEXITCODE -ne 0) { Write-Error "freeze_preregistration failed (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
    } else {
        # a record exists: never overwrite it; list what changed since (record intended changes with --record-deviation)
        Write-Stamp '[freeze] frozen record exists; checking for changes since the freeze (python -m scripts.freeze_preregistration --check)'
        Invoke-Stamped scripts.freeze_preregistration --check
        if ($LASTEXITCODE -ne 0) { Write-Stamp 'WARNING: changes since the freeze are listed above. They are reported as deviations; add --record-deviation "<reason>" for an intended one.' 'Yellow' }
    }
}
if (-not $skipThree -and -not $noRun) {
    if ($redoThree) {
        Remove-Item -ErrorAction SilentlyContinue (Join-Path $root 'results\walk_forward_q8.json')
        Remove-Item -ErrorAction SilentlyContinue -Recurse (Join-Path $root 'results\naive_baseline\legacy')
        Remove-Item -ErrorAction SilentlyContinue (Join-Path $root 'results\feature_map_advantage.json')
        Remove-Item -ErrorAction SilentlyContinue (Join-Path $root 'results\feature_map_advantage_q12.json')
        Remove-Item -ErrorAction SilentlyContinue (Join-Path $root 'results\walk_forward_q12.json')
        Remove-Item -ErrorAction SilentlyContinue -Recurse (Join-Path $root 'results\naive_baseline\legacy_q12')
    }
    if (-not $with12) {
        Write-Stamp 'WARNING: Q1 runs at 8 qubits only. PREREGISTRATION.md also names 12 qubits (about 20x the simulation cost per row); skipping it is a deviation to record. Add --with-12q to run it.' 'Yellow'
    }
    # preflight (seconds): a plan that cannot run fails here, not after hours of simulation
    Write-Stamp '[three-arm] preflight: windows and data of all three steps'
    Invoke-Stamped scripts.run_walk_forward --dry-run
    if ($LASTEXITCODE -ne 0) { Write-Error "preflight: the walk-forward plan is not valid (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
    Invoke-Stamped scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8 --dry-run
    if ($LASTEXITCODE -ne 0) { Write-Error "preflight: the feature-map plan is not valid (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
    if ($with12) {
        Invoke-Stamped scripts.run_walk_forward --n-qubits 12 --dry-run
        if ($LASTEXITCODE -ne 0) { Write-Error "preflight: the 12-qubit walk-forward plan is not valid (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
        Invoke-Stamped scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 12 --dry-run
        if ($LASTEXITCODE -ne 0) { Write-Error "preflight: the 12-qubit feature-map plan is not valid (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
    }
    Write-Stamp '[three-arm] 1/3  python -m scripts.run_walk_forward'
    Invoke-Stamped scripts.run_walk_forward --skip-if-current
    if ($LASTEXITCODE -ne 0) { Write-Error "run_walk_forward failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    Write-Stamp '[three-arm] 2/3  python -m scripts.run_naive_baseline'
    # --experiment-id legacy: Arm 1 is a deterministic ridge on the processed data, so ONE file serves every experiment; the
    # hypothesis tests look in the experiment's own folder first and then in results\naive_baseline\legacy.
    Invoke-Stamped scripts.run_naive_baseline --experiment-id legacy
    if ($LASTEXITCODE -ne 0) { Write-Error "run_naive_baseline failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    Write-Stamp '[three-arm] 3/3  python -m scripts.run_feature_map_experiment   (Q1: 10 paired seeds, 8 qubits)'
    Invoke-Stamped scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8 --skip-if-current
    if ($LASTEXITCODE -ne 0) { Write-Error "run_feature_map_experiment failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    if ($with12) {
        Write-Stamp '[three-arm] 3b   python -m scripts.run_feature_map_experiment --n-qubits 12   (Q1, 12 qubits)'
        Invoke-Stamped scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 12 --out results/feature_map_advantage_q12.json --skip-if-current
        if ($LASTEXITCODE -ne 0) { Write-Error "run_feature_map_experiment (12 qubits) failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
        Write-Stamp '[three-arm] 1b   python -m scripts.run_walk_forward --n-qubits 12   (about 12 h: 25 quantum fits)'
        Invoke-Stamped scripts.run_walk_forward --n-qubits 12 --skip-if-current
        if ($LASTEXITCODE -ne 0) { Write-Error "run_walk_forward (12 qubits) failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
        Write-Stamp '[three-arm] 2b   python -m scripts.run_naive_baseline --n-qubits 12   (own folder legacy_q12)'
        Invoke-Stamped scripts.run_naive_baseline --like results/walk_forward_q12.json --n-qubits 12 --experiment-id legacy_q12
        if ($LASTEXITCODE -ne 0) { Write-Error "run_naive_baseline (12 qubits) failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    }
}

Write-Stamp ('[study] start: python -m scripts.run_full_study ' + ($studyArgs -join ' ')) 'Cyan'
& $py -m scripts.run_full_study @studyArgs
$code = $LASTEXITCODE
$el = (Get-Date) - $t0
Write-Stamp ('[study] finished with exit code {0}; total elapsed {1:dd\.hh\:mm\:ss}' -f $code, $el) 'Cyan'
exit $code
