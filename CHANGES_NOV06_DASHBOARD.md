# 2026-11-06 -- study console: per-run banners, progress bar, subtasks, failure diagnosis

Extract over the repo root (replaces scripts/run_full_study.py; adds three files). No other file changes.

- scripts/run_full_study.py  patched: dashboard + diagnosis wired into _run_one / main (old output via --no-dashboard)
- scripts/study_dashboard.py header "[study] 8 conditions x 10 seeds = 80 runs; N remaining", overall bar + ETA,
  per-condition bars, START/END banners, live subtask block every --status-interval s (default 120),
  "retry queue" of failed attempts
- scripts/study_diagnose.py  one-sentence "why: ... -> fix: ..." for every [worker] FAIL (fixed ordered rules; reads
  only the latest attempt in logs/study/<condition>_seed<k>.log; also handles Windows exit codes and "exit 0 but no row")
- tests/test_study_dashboard.py (17 tests)

Every failure is also appended to logs/study/failures.log. Each attempt writes a "===== attempt started" marker into
the run log. --workers 1 now also goes through the dashboard path (use --no-dashboard for the old sequential pass).

Verified: new tests + tests/test_run_full_study.py pass (30), a simulated failing worker prints the expected block.
NOT verified: a real multi-hour run (no torch/PennyLane/data here). Subtasks show only for stages wrapped by
src/utils/stage_log.py; the rule list covers the failure types seen in this code base, anything else prints the last
exception line and "read the log".
