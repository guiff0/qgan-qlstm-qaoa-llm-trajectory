"""Quantum-vs-classical balance audit.

    python -m scripts.check_balance                          # audit config/default_config.yaml, exit 1 on FAIL
    python -m scripts.check_balance --experiment-id EXP-...  # also saves results/experiments/<id>/balance_audit.json
                                                             # and prints median compute minutes per condition
run_full_study runs this before launching (skip with --skip-balance-check).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.balance_audit import audit, compute_budget, format_report  # noqa: E402
from src.utils import experiment as exp  # noqa: E402


def run(config_path: str, experiment_id: str | None = None, tol: float = 0.05, root: str = exp.EXPERIMENT_ROOT) -> dict:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rep = audit(cfg, tol=tol)
    if experiment_id:
        rep["compute_budget"] = compute_budget(exp.read_runs(experiment_id, root))
        rep["config_path"], rep["experiment_id"] = config_path, experiment_id
        d = exp.experiment_dir(experiment_id, root)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "balance_audit.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--tol", type=float, default=0.05, help="allowed parameter-count gap (default 5%%)")
    a = ap.parse_args()
    rep = run(a.config, a.experiment_id, a.tol)
    print(format_report(rep))
    if rep.get("compute_budget", {}).get("median_minutes_per_condition"):
        print("  compute (median wall minutes per run):", json.dumps(rep["compute_budget"]["median_minutes_per_condition"]))
        if "quantum_over_matched_classical" in rep["compute_budget"]:
            print(f"  quantum / matched classical compute: x{rep['compute_budget']['quantum_over_matched_classical']}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
