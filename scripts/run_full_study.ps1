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
.EXAMPLE   .\scripts\run_full_study.ps1 --redo-three-arm --workers 4   # refit the walk-forward file and the naive baseline from scratch

           BEFORE the study (13 Nov 2026) it runs the three three-arm steps the hypothesis tests read at the end:
             1. python -m scripts.run_walk_forward      Arms 2 and 3 (classical / quantum features), skipped if the file is already
                                                        complete for this plan with the current ridge grid
             2. python -m scripts.run_naive_baseline    Arm 1 (inputs only), same windows, seeds, grid and attack rows; resumable
             3. python -m scripts.run_feature_map_experiment --seeds 0..9 --n-qubits 8
                                                        the paired 10-seed test behind the Q1 verdict (needs >= 10 seeds, else UNDERPOWERED);
                                                        skipped if the file is already complete. Hours of CPU: the quantum arm is simulated.
           All three are skipped for --status, --dry-run, --help and --quick. None depends on the GAN runs.
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. (Join-Path $PSScriptRoot '_python.ps1')
$py = Resolve-ProjectPython $root

# --- three-arm steps (own switches are removed before the rest is passed to run_full_study) -------------------------------------
$skipThree = $args -contains '--skip-three-arm'
$redoThree = $args -contains '--redo-three-arm'
$with12 = $args -contains '--with-12q'
$studyArgs = @($args | Where-Object { $_ -ne '--skip-three-arm' -and $_ -ne '--redo-three-arm' -and $_ -ne '--with-12q' })
$noRun = @('--status', '--dry-run', '--help', '-h', '--quick') | Where-Object { $studyArgs -contains $_ }
if (-not $skipThree -and -not $noRun) {
    if ($redoThree) {
        Remove-Item -ErrorAction SilentlyContinue (Join-Path $root 'results\walk_forward_q8.json')
        Remove-Item -ErrorAction SilentlyContinue -Recurse (Join-Path $root 'results\naive_baseline\legacy')
        Remove-Item -ErrorAction SilentlyContinue (Join-Path $root 'results\feature_map_advantage.json')
        Remove-Item -ErrorAction SilentlyContinue (Join-Path $root 'results\feature_map_advantage_q12.json')
    }
    if (-not $with12) {
        Write-Warning 'Q1 runs at 8 qubits only. PREREGISTRATION.md also names 12 qubits (about 20x the simulation cost per row); skipping it is a deviation to record. Add --with-12q to run it.'
    }
    # preflight (seconds): a plan that cannot run fails here, not after hours of simulation
    Write-Host '[three-arm] preflight: windows and data of all three steps'
    & $py -m scripts.run_walk_forward --dry-run
    if ($LASTEXITCODE -ne 0) { Write-Error "preflight: the walk-forward plan is not valid (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
    & $py -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8 --dry-run
    if ($LASTEXITCODE -ne 0) { Write-Error "preflight: the feature-map plan is not valid (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
    if ($with12) {
        & $py -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 12 --dry-run
        if ($LASTEXITCODE -ne 0) { Write-Error "preflight: the 12-qubit feature-map plan is not valid (exit $LASTEXITCODE); nothing was run."; exit $LASTEXITCODE }
    }
    Write-Host '[three-arm] 1/3  python -m scripts.run_walk_forward'
    & $py -m scripts.run_walk_forward --skip-if-current
    if ($LASTEXITCODE -ne 0) { Write-Error "run_walk_forward failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    Write-Host '[three-arm] 2/3  python -m scripts.run_naive_baseline'
    # --experiment-id legacy: Arm 1 is a deterministic ridge on the processed data, so ONE file serves every experiment; the
    # hypothesis tests look in the experiment's own folder first and then in results\naive_baseline\legacy.
    & $py -m scripts.run_naive_baseline --experiment-id legacy
    if ($LASTEXITCODE -ne 0) { Write-Error "run_naive_baseline failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    Write-Host '[three-arm] 3/3  python -m scripts.run_feature_map_experiment   (Q1: 10 paired seeds, 8 qubits)'
    & $py -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8 --skip-if-current
    if ($LASTEXITCODE -ne 0) { Write-Error "run_feature_map_experiment failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    if ($with12) {
        Write-Host '[three-arm] 3b   python -m scripts.run_feature_map_experiment --n-qubits 12   (Q1, 12 qubits)'
        & $py -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 12 --out results/feature_map_advantage_q12.json --skip-if-current
        if ($LASTEXITCODE -ne 0) { Write-Error "run_feature_map_experiment (12 qubits) failed (exit $LASTEXITCODE); the study was not started."; exit $LASTEXITCODE }
    }
}

& $py -m scripts.run_full_study @studyArgs
exit $LASTEXITCODE
