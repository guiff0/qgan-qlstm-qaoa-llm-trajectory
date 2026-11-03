"""Freeze the analysis plan BEFORE the confirmatory runs, so margins and decision rules cannot drift afterwards.

  python -m scripts.freeze_preregistration            # write results/preregistration_<UTC>.json (never overwrites)
  python -m scripts.freeze_preregistration --check    # compare the working tree with the newest frozen record

The record holds the equivalence margins, the decision rules (PREREGISTRATION.md, hashed), the config, the hashes
of the files that define the experiment, the seeds, and the git commit when available. `--check` exits 1 and names
every item that changed since the freeze, so any post-freeze change is visible and has to be reported as such.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRACKED = [
    "config/default_config.yaml", "PREREGISTRATION.md",
    "scripts/run_hypothesis_tests.py", "scripts/run_feature_map_experiment.py",
    "src/baselines/feature_map_forecaster.py", "src/quantum/feature_map.py",
    "src/utils/generator_path.py", "src/utils/capacity_match.py",
    "src/baselines/qgan_llm.py", "src/baselines/classical_gan_llm.py",
]
SEEDS = list(range(10))          # confirmatory seeds for both studies (10 gives power for ~1.3 sigma paired differences)
FEATURE_MAP_RUNS = [{"n_qubits": 8, "n_layers": 3}, {"n_qubits": 12, "n_layers": 3}]


def _sha(path: str) -> str | None:
    p = os.path.join(ROOT, path)
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _margins() -> dict:
    from scripts.run_hypothesis_tests import EQUIV_ALPHA, EQUIV_MARGINS
    return {"alpha": EQUIV_ALPHA, "margins": {k: list(v) for k, v in EQUIV_MARGINS.items()}}


def build_record() -> dict:
    return {"frozen_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "git_commit": _commit(),
            "equivalence": _margins(), "seeds": SEEDS, "feature_map_runs": FEATURE_MAP_RUNS,
            "file_sha256": {p: _sha(p) for p in TRACKED}}


def _latest() -> str | None:
    files = sorted(glob.glob(os.path.join(ROOT, "results", "preregistration_*.json")))
    return files[-1] if files else None


def check() -> int:
    path = _latest()
    if not path:
        print("no frozen record found: run without --check first")
        return 2
    frozen, now = json.load(open(path)), build_record()
    changed = [f"{p}: file changed" for p, h in frozen["file_sha256"].items() if now["file_sha256"].get(p) != h]
    if frozen["equivalence"] != now["equivalence"]:
        changed.append("equivalence margins / alpha changed")
    if frozen["seeds"] != now["seeds"]:
        changed.append("seed list changed")
    print(f"frozen record: {os.path.basename(path)} (commit {frozen.get('git_commit')})")
    print("\n".join(changed) if changed else "nothing changed since the freeze")
    return 1 if changed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    rec = build_record()
    missing = [p for p, h in rec["file_sha256"].items() if h is None]
    if missing:
        print(f"refusing to freeze: tracked files missing: {missing}")
        return 1
    out = os.path.join(ROOT, "results", f"preregistration_{rec['frozen_utc']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x") as fh:                       # 'x': never overwrite a frozen record
        json.dump(rec, fh, indent=2)
    print(f"frozen -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
