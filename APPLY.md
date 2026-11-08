# Applying the Nov-07 update (two phases) and running the second experiment

Everything is an overlay: copy a phase folder onto the repo root (same relative paths). Nothing is deleted.

## Phase A — safe now, while the current 80-run study is still running
`phase_A_now/` contains analysis, launcher and tooling files. None of them is imported by a run in progress
(the training runs import only the files in phase B), so the running study is not affected. If the launcher
itself is ever restarted it will use the new version, which refuses to guess (see "Resuming" below).

After copying:
```
python -m pytest tests -q          # new tests; the torch/pennylane ones need your venv
python -m scripts.freeze_preregistration --record-deviation "analysis tooling added after the freeze (see PREREGISTRATION_ADDENDUM.md)"
python -m scripts.check_balance    # must print PASS
python -m scripts.experiments list # shows the current rows as 'unassigned'
```

## Phase B — apply AFTER the last run finishes (or on the second machine before its first launch)
`phase_B_after_study/` changes the run path: experiment id and machine are stamped into every results row, training and
evaluation seconds are logged, and the per-row forecasts are saved (`results/predictions/<experiment>/`). Applying it while
runs are being launched would mix two code versions inside one experiment.

Then, on the machine that ran the first study:
```
python -m scripts.experiments adopt --since 2026-11-03            # dry run: lists what would be stamped
python -m scripts.experiments adopt --since 2026-11-03 --apply     # stamps those rows with a new experiment id
python -m scripts.experiments verify <EXP-id>
python -m scripts.run_hypothesis_tests --experiment-id <EXP-id>
```
`--since 2026-11-03` is the first day of the 80-run study in your results file; the 23 earlier pilot rows (27-31 Oct, including
the seed-42 runs of QGAN-LLM and QGAN-LLM (no noise) that the earlier H1/H3 output counted as n = 2) stay 'unassigned' and
are excluded from the study's analysis. Check the dry run against your own file first.

## The second 80-run experiment (other machine)
1. Copy the repo with BOTH phases applied. Do not copy `results/` or `logs/` (a fresh folder avoids any confusion).
2. `.\scripts\run_full_study.ps1 --new-experiment --label "machine B" --base-seed 143 --n-runs 10 --workers 4`
   * a new id such as `EXP-20261108-LAPTOP-9c1d` is created with a manifest (plan, config hash, code hashes, git commit, machine);
   * `--base-seed 143` gives seeds that cannot collide with 43-52. Use `--base-seed 43` only if you want an exact replication;
     it is still a separate experiment and is never merged silently.
3. When it finishes, zip the `results` folder and on machine A: `python -m scripts.experiments import D:\machine_b.zip` (dry run), then `--apply`.
   Import is idempotent, refuses rows without an experiment id, and refuses a run id that already exists under another experiment.
4. Analyse one experiment at a time: `run_hypothesis_tests --experiment-id <id>`. Several need `--pool-experiments`, which turns the
   paired-by-seed analyses off and lists seeds that appear in both.

## Resuming
Re-launching with no flags resumes the ACTIVE experiment (`results/experiments/ACTIVE`) if its plan is the same; it refuses when the
results file holds rows that belong to no experiment (it prints `adopt`, `--new-experiment` and `--legacy` as the choices).
`--status` and `--dry-run` never refuse.

## Claims 4 and 6 after the study
* Temporal stability and the row-level Diebold-Mariano rule: `python -m scripts.run_temporal_validation --experiment-id <id>`.
  It needs the per-row forecasts, which exist for runs made with phase B. Runs made before it have none; whether re-running a finished
  run's evaluation from its checkpoint (resume) reproduces the logged RMSE can be checked on ONE run first, because the saved file records
  the recomputed and the logged RMSE side by side.
* Preprocessing sensitivity: `python -m scripts.run_sensitivity_study --parent <id> --write-configs`, then run the printed commands
  (about 30 h with 4 workers, a lower-cost proxy), then `python -m scripts.compare_sensitivity --parent <id>`.
  Never run `prepare_data` without `--processed-dir` for a variant: the baseline data folder is what the main study reads.

## Not verified here
The sandbox has no torch, PennyLane or your data. New code was tested with synthetic data and fake models (all new tests pass); the
17 test modules that import torch/pennylane could not be collected and must be run in your venv. Nothing was run against a real
80-run experiment. The manuscript changes are tracked changes.
