"""EXPLORATORY mediator screen on the entanglement sweep (discovery half / replication half). Never a verdict.

    python -m scripts.run_mediation_screen [--csv results/all_results.csv] [--experiment EXP-...]

Writes results/mediation_screen_<UTC>.json. Read src/evaluation/mediation_screen.py first: a survivor is a hypothesis for a
new pre-registered run, not support for M2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.entanglement_design import sweep_subset  # noqa: E402
from src.evaluation.mediation_screen import screen  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "all_results.csv"))
    ap.add_argument("--experiment", default=None, help="restrict to one experiment_id (never mix experiments)")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if args.experiment and "experiment_id" in df.columns:
        df = df[df["experiment_id"] == args.experiment]
    sw = sweep_subset(df, {"asr", "entanglement_entropy"})
    if sw is None or len(sw) < 24:
        print("not enough sweep runs for a discovery/replication split (need >= 24)")
        return 2
    res = screen(sw, n_bootstrap=args.n_bootstrap)
    res["n_runs"] = int(len(sw))
    path = os.path.join(ROOT, "results", f"mediation_screen_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    with open(path, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(res["label"])
    print(f"{res['k_candidates_tested']} candidates tested; ~{res['expected_false_hits_at_alpha']} false hits expected at alpha .05")
    for r in res["candidates"]:
        if r["status"] == "tested":
            print(f"  {r['candidate']:34s} disc p(Holm)={r['discovery_p_holm']:.3f}  survives={r['survives_discovery']!s:5}  replicates={r['replicates']}")
    print("survivors:", res["survivors"] or "none")
    print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
