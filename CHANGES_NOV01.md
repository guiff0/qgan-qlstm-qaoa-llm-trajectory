# 2026-11-01 -- fixes combined into this code base

## Dukascopy acquisition: HTTP 429 on every year (scripts/acquire_all_data.py)
Symptom: all 14 years "failed after 3 attempts: 429 Too Many Requests", 25 minutes, nothing kept.
Cause: the direct datafeed client sent one un-throttled request per calendar day from 8 threads
(~5,100 requests) and cached nothing per day. (`dukascopy_python` itself is broken -- NameError on
`requests` -- so the direct client is the only path.)
Now:
- one shared throttle (`DUKASCOPY_MIN_INTERVAL`, default 0.35 s) and 2 workers (`DUKASCOPY_WORKERS`);
- 429/5xx/timeouts back off exponentially (honouring `Retry-After`) and pause ALL workers;
- per-day cache in `data/raw/dukascopy_cache/days/` (`.bi5` body, or `.none` for 404/empty days), so a
  re-run only requests days still missing; corrupt bodies are never cached;
- Saturdays (no FX data) are not requested;
- circuit breaker: after 4 consecutive exhausted days the step stops at once with "wait 15-60 min and
  re-run" instead of failing through every remaining year; rate limiting is not retried at year level;
- the "dukascopy_python unusable" warning prints once per run, not once per year.
If it still gets refused (limits are per IP): `DUKASCOPY_WORKERS=1 DUKASCOPY_MIN_INTERVAL=1.0`.
Tests: tests/test_dukascopy_rate_limit.py (no network).

## Earlier today (see the chat): H1/H3 config pooling, QLSTM persistence rows, zero-entanglement sweep level 0
Files: scripts/run_hypothesis_tests.py, src/baselines/qlstm_forecaster.py, src/quantum/*.py,
src/attacks/quantum_attacks.py, src/evaluation/{entanglement_design,mediation_pipelines}.py,
config/default_config.yaml, tests/test_no_config_pooling.py, tests/test_entanglement_sweep.py.

## Classical LSTM compute budget (src/baselines/classical_lstm.py, config/default_config.yaml)
58,061 batches/epoch (3.7M overlapping windows / 64) was ~2 h/epoch on CPU, up to 50 epochs.
New `classical_lstm.max_batches_per_epoch: 2000` (shuffled -> fresh random subsample each epoch;
null = old full pass) and `val_samples: 20000` (fixed-seed validation subsample). Not run here (no torch).

## Dukascopy: one HTTP 503 day aborted the step (2017-08-27) -- scripts/acquire_all_data.py
Years 2012-2016 downloaded fine (cached); 2017 stopped because a single day answered 503 six times.
A day that fails only with 5xx (never 429/network errors) is now skipped, logged in
data/raw/dukascopy_cache/missing_days.csv, not cached, and the year continues (tolerance
`DUKASCOPY_MAX_GAP_DAYS`, default 5/year; more than that = server unhealthy = fatal). 429s still trip
the breaker. To retry skipped days later: delete that year's eurusd_1min_<year>.csv and the master, re-run.

## Equivalence (TOST) + revised hypotheses
- `src/evaluation/statistical_tests.py`: `tost_equivalence` (Welch TOST on a mean difference) and
  `tost_correlation` (Fisher-z, |r| < r0).
- `scripts/run_hypothesis_tests.py`: `EQUIV_MARGINS` (+ `--equiv-margin KEY=VALUE`); `equivalence` blocks for
  H1 (FID), H3 (poisoning %), H4 (RMSE vs persistence scale, ASR), `equivalence_r` for H2/H5.
- `src/evaluation/verdicts.py`: equivalence is reported beside the superiority verdict, never rewrites it.
- `HYPOTHESES_REVISED.md`: draft statements, margins to freeze, power table, disclosure text.
- tests/test_equivalence_tost.py

## One-launch study runner (scripts/run_full_study.py, scripts/run_full_study.ps1)
`.\scripts\run_full_study.ps1` (or `python -m scripts.run_full_study`) runs 7 conditions x 5 seeds (primary
QGAN-LLM, Classical GAN-LLM, QGAN-LLM (no noise), sweep levels 0/2/4/6), repeating passes until every
(condition, seed) is in results/all_results.csv, then runs the hypothesis tests. Resumable; retries failed
runs with waiting; gives up only after 5 no-progress passes (`--max-stalled-passes 0` = never). Refuses to
start if forecast_path is not `generator`. Options: --n-runs, --levels, --status, --dry-run. Tests: tests/test_run_full_study.py.

## `--workers N` for run_full_study
N (condition, seed) runs at a time, each its own process with logical_cpus // N BLAS/OpenMP threads
(OMP/MKL/OPENBLAS/NUMEXPR); per-run output in logs/study/<condition>_seed<k>.log. Two runs of the same
condition never overlap (run_ids are claimed check-then-create per second). Default 1 = sequential.
