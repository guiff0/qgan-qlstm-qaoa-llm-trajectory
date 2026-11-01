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
