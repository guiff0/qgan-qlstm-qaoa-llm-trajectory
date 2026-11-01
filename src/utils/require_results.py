"""
Order guard: results/all_results.csv is produced ONLY by src/experiments/run_all.py (one row per
model run). scripts/verify_chapter4.py and scripts/run_hypothesis_tests.py are analysis steps that
read it, so running either before run_all gives an empty or misleading report.

    1. python -m src.experiments.run_all          (trains + evaluates every model; writes the CSV)
       -- or run_pipeline.py / scripts/run_repeated_experiment.py, which call run_all --
    2. python -m scripts.verify_chapter4
    3. python -m scripts.run_hypothesis_tests

Both analysis scripts call require_results() first and stop with exit code 2 and this message if the CSV is
missing or has no data rows.
"""
from __future__ import annotations

import os
import sys

ORDER_HINT = ("Order: src.experiments.run_all  ->  scripts.verify_chapter4  ->  scripts.run_hypothesis_tests")


def count_result_rows(csv_path: str) -> int:
    """Data rows in the CSV (header excluded); 0 if the file is missing or unreadable."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
    except OSError:
        return 0
    return max(n - 1, 0)


def require_results(csv_path: str, caller: str) -> int:
    """Return the number of result rows, or exit(2) telling the user to run run_all first."""
    n = count_result_rows(csv_path)
    if n == 0:
        state = "does not exist" if not os.path.isfile(csv_path) else "has no data rows"
        print(
            f"[{caller}] {csv_path} {state}.\n"
            f"Run the models first:  python -m src.experiments.run_all\n"
            f"  (a subset:  python -m src.experiments.run_all --only \"QGAN-LLM\"; or  python run_pipeline.py)\n"
            f"then re-run {caller}.\n{ORDER_HINT}",
            file=sys.stderr,
        )
        raise SystemExit(2)      # SystemExit("text") would exit 1; 2 = "precondition not met"
    return n
