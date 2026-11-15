# 14 Nov 2026 - fixes from the EXP-20261114-SERVTEN-8564 console log

The 12 runs themselves finished cleanly (exit 0). What was wrong was the report around them, plus one real instrument problem (M8).

## Reporting bugs (fixed)
| Seen in the log | Cause | Fix |
|---|---|---|
| `hypothesis=failed` for the whole study | `results/study_driver_state.json` still held the record of a 10 Nov experiment | `study_dashboard.stage_marks`: a record of another experiment reads `pending` |
| `main=pending` after `ALL RUNS COMPLETE` | only `scripts.run_study` wrote stage records; `run_full_study(.ps1)` wrote none | `run_full_study.mark_stage` records `main` and `hypothesis` (running / ok / failed / stopped_at_gate) |
| `three-arm feature-map: not run yet` although `feature_map_advantage.json` had 10 seeds x 3 arms | the dashboard only globbed `feature_map_advantage_q*.json` | `study_dashboard.feature_map_files` finds both names; `feature_map=ok (file on disk)` |
| `ETA n/a (no run finished yet)` at 12/12 | fall-through text | `all runs done` |
| thousands of JSON lines in the console, cut mid-string | `run_hypothesis_tests` printed the whole record | prints the saved path; `--print-json` restores the dump |
| `H1 ... CONTRADICTED (exploratory, interim)` under a table row `SUGGESTIVE_AGAINST`; `NOT_SUPPORTED` under `INCONCLUSIVE` | interim downgrade changed the verdict but not the headline word | `verdicts.apply_interim_gate` rewrites the headline too |
| `-0.0000` skills in the walk-forward table | sign of a value below the printed precision | printed as `0.0000` |

## M8 poisoning positive control (-15.8 % vs floor 5 %): calibration rule fixed, instrument NOT yet valid
`calibrate_shift` accepted the smallest shift whose MEDIAN pilot seed reached the target; check M8 fails when ANY run is under
the floor. The two rules did not match, so a launch could be "calibrated" and still fail M8. Now a shift is accepted only if the
median reaches the target AND every pilot seed clears `pc_floor_pct`, at that shift and every larger one (`--floor`, default
`poisoning.pc_floor_pct`). If none qualifies, nothing is frozen and the note says the control is unstable.

Why the control is unstable (from the 13 Nov rows in all_results.csv; the 14 Nov rows were not in the zip): the real-only
control head is trained on the OLDEST 100,000 rows of the training half and scored on the later half. Its clean RMSE is
0.000358, against 0.000156 for the main head and 0.000144 for persistence. A head that far behind persistence is
extrapolating, so a small spurious slope moves its error up or down depending on the seed's random attack direction:
-2.3 %, +18.6 %, +13.5 %, -1.2 % on seeds 48-51 at the same strength.

Not changed here (a protocol decision, to be recorded as an amendment if taken): which rows the positive-control head trains on
(e.g. the training rows nearest the evaluation window instead of the oldest ones).

## To do on the study machine
1. `pytest tests/test_nov14_report_fixes.py`
2. Set `poisoning.calibrated: false`, then `python -m scripts.calibrate_poisoning` and read the `ok` column.
3. Tracked files changed: record it (`--record-deviation "14 Nov: calibration floor on every pilot seed; report fixes"`).

## Addendum - pilot table (descriptive only)
`python -m scripts.pilot_table [--all-runs]` writes `results/pilot_table.md` / `.json`: RMSE over persistence for the October
one-seed runs (Classical LSTM, QLSTM Forecaster, QLSTM-LLM, QAOA-Enhanced LLM, QAOA-LLM, Classical-Features-LLM, LLM Forecaster).
No test, no interval, no verdict. The script imports nothing from `src` or `scripts`, is not read by `run_hypothesis_tests`, and is
not in the freeze's TRACKED list. Three-arm code (`run_walk_forward`, `run_naive_baseline`, `run_feature_map_experiment`,
`walk_forward.py`, `feature_map_forecaster.py`, `feature_map.py`, `arm_metrics.py`, `three_arm.py`, `protocol_compliance.py`) is
byte-identical to the uploaded tree; its 61 tests pass before and after.

## Addendum 2 - Classical LSTM and QAOA-Enhanced LLM forecast the next-bar change (two pilot reruns)
Purpose: presentational only. The pilot table's two level-frame rows (38x and 687x persistence) become like-for-like with the
other pilots. Still one seed, still no inference, still outside H1-H5 and outside the three-arm study.

- `forecast_target: delta` is now the default of both models (set in the model's own defaults, NOT in config/default_config.yaml,
  which is a frozen file). `classical_lstm.forecast_target: level` / `qaoa_llm.forecast_target: level` give the October behaviour.
- Classical LSTM: target = close after the window minus last close in the window, standardised by the training SD; forecast =
  network output x that SD. Delta runs use `models/classical_lstm_delta_best.pt`; the October `models/classical_lstm_best.pt` is
  not overwritten, and a checkpoint of the other frame is refused with a clear error.
- QAOA-Enhanced LLM: features at row k -> change to row k+1. QAOA scores candidates against that same target, so the selected
  features can differ from October's. Correlations with a next-bar change are small, so the budget and redundancy terms will
  weigh more in the QUBO than they did against the price level.
- Both report `target_frame` and a persistence RMSE on exactly the rows they were scored on; the attack reference is 0.

Run (do NOT pass --reuse-checkpoints; rows carry no experiment id, so the study never reads them):

    python -m src.experiments.run_all --only "Classical LSTM" --seed 42
    python -m src.experiments.run_all --only "QAOA-Enhanced LLM" --seed 42
    python -m scripts.pilot_table --all-runs

Tests: tests/test_delta_pilot_models.py (7), tests/test_pilot_table.py (4).

## Addendum 3 - pilot models: drift-free inputs, validation-gated heads, exact QAOA audit (14 Nov, afternoon)
Measured on training and validation rows only: four PCA components sit 2.6-4.8 training SDs away in the validation years, and a
linear next-bar model on the raw components is worse than no-change on validation (1.0004) while the same model on drift-free
inputs is not (0.9999). The lag-1 autocorrelation of the next-bar change is about -0.01 to -0.02, so the most any model can gain
over no-change on this target is a few hundredths of a percent.

- Classical LSTM (`input_centering: window`, default in the delta frame): each 60-row window has its own mean subtracted. The
  output layer starts at zero (= the no-change forecast) and the epoch-0 validation loss is the bar every epoch must beat;
  `best_epoch: -1` in the row means none did and the saved model is the no-change forecast.
- QAOA-Enhanced LLM (`feature_frame: diff`, `head_fit: ridge`, `ridge: auto`, defaults in the delta frame): inputs are the
  change of each feature since the previous bar; the head is a closed-form ridge whose penalty is chosen on the validation rows,
  with "no head" (no-change) as a candidate. `forecast_target: level` restores the October behaviour of both models.
- `python -m scripts.qaoa_solution_audit --selected <indices>`: rebuilds the model's QUBO and lists all 65,536 answers to
  show where QAOA's answer ranks. Descriptive; reads no test data.

Seed 42, run in the assistant's sandbox on the uploaded data (X_train rebuilt from the raw Dukascopy file; y and VIX arrays
bit-identical to the uploads, X_val/X_test within 1e-6):
- QAOA-Enhanced LLM: 687.113 (October, level) -> 1.00107 (diff inputs, fixed ridge 1.0) -> 1.00002 (validation-chosen ridge).
  QAOA picked 8 features; its QUBO value 3.884 ranks 2,566th of 65,536 (optimum 1.839).
- Classical LSTM: NOT completed there (process stopped at epoch 7 of training). Run it on the study machine.

## Run script
`.\scripts\run_nov14.ps1` does, in order: tests of these changes -> the two pilots (seed 42), the exact QAOA audit and the pilot
table -> poisoning recalibration under the new rule -> `run_full_study.ps1 --refreeze --new-experiment --base-seed 53 --workers 4
--record-deviation "..."`. Switches: `-SkipPilots`, `-PilotsOnly`, `-AllowUncalibrated`, `-NoRefreeze`. Not executed on Windows
by the assistant (no PowerShell available there); read the first lines it prints.

## Addendum 4 - selector comparison (exploratory; no LLM call; not the three-arm methodology)
`python -m scripts.qaoa_selector_comparison` (about 10 minutes, CPU only): the same 16 candidates and the same head, with the
selector swapped: QAOA (10 seeds), top-8 by correlation, and the exact QUBO optimum found by listing all 65,536 answers.
Run in the assistant's sandbox on the uploaded data, seeds 42-51:
- QAOA found the exact optimum in 7 of 10 seeds (ranks 1, 2,566 and 3,694 of 65,536); all 10 returned exactly 8 features.
- Forecast error is the same for every selector to five decimals: RMSE / persistence between 0.999995 and 1.000022.
  The classical top-8 set was rejected by the validation gate (its head is the no-change forecast, ratio exactly 1).
It reads and writes nothing of `run_walk_forward`, `run_naive_baseline` or the feature-map experiment; those files are unchanged.

## Addendum 5 - study driver stopped at run 31 of 40 (EXP-20261113-DESKTOPK9L37-0a2a, 14 Nov 15:34)
Cause: after `QGAN-LLM seed 58` finished, the dashboard re-read the per-condition progress file
(`logs\experiments\<id>\repeated_experiment_*.json`) at the moment a worker was replacing it, got half a file, and the
JSONDecodeError went up through `DASH.run_finished` and ended the driver. No run failed; the 31 rows are in the results CSV.
- `run_repeated_experiment._load_state` never raises: it retries, then treats an unreadable file as empty with one warning
  (completed seeds also come from the results CSV, so nothing is rerun). `_save_state` uses a temp file unique to the process
  and thread and retries the replace while a reader holds the file open (Windows).
- `run_full_study._run_one`: a dashboard error is printed and ignored; it can no longer stop a study.
Resume: relaunch with the SAME arguments as the original launch, WITHOUT `--new-experiment` and without `--refreeze`.
Do not run `run_nov14.ps1` on that machine until this experiment is finished: it resets the calibration and starts a new study.

## Addendum 6 (15 Nov) - H2/H3 headline after the Holm correction; M8 positive control for FUTURE studies
**H2/H3 text.** `verdicts.compute_verdicts`: when the Holm correction downgrades a verdict to INCONCLUSIVE, the headline now says
so ("INCONCLUSIVE (leans for) ... Holm-adjusted p=0.126, not significant after correction") instead of keeping "SUPPORTED" /
"CONTRADICTED". Replayed on the saved confirmatory results of EXP-20261113-DESKTOPK9L37-0a2a: every verdict, rule, tier and
adjusted p is unchanged; only those two lines of text differ. `verdicts.py` is a frozen file: record the deviation. Do not
rerun the confirmatory tests for this.

**M8, cause.** The positive-control head trained on the OLDEST 100,000 rows of the training half. On the test split the price
level moves from about 1.085 there to about 1.130 in the scored rows; the clean head's error on the scored rows is 133x the
no-change error (0.42x the spread of the scored prices), so it had failed before any poison was added and its reaction to the
poison changed sign with the seed.

**M8, fix for the next study only.** `poisoning.pc_max_rows: null` (or "all") = the control trains on the whole training half,
the same rows as the main head. A number behaves exactly as before, so nothing changes until the config is changed.
    python -m scripts.set_positive_control --rows all --apply     # writes pc_max_rows: null and calibrated: false (+ backup)
The next launch then recalibrates. On the uploaded test data: clean head error 0.005x the spread; calibration picks
shift_sd 0.1 (pilot seeds: median 100 %, worst 6.7 %; at 0.2 the worst is 124 %). `calibrate_poisoning` now prints the clean
head's error and warns when it is above 0.1. Rows record `poisoning_pc_rows`.

**This experiment.** H4 of EXP-20261113-DESKTOPK9L37-0a2a stays as it is: not decidable, because the positive control failed.
