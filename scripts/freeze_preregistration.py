"""Freeze the analysis plan BEFORE the confirmatory runs, so margins and decision rules cannot drift afterwards.

  python -m scripts.freeze_preregistration            # write results/preregistration_<UTC>.json (never overwrites)
  python -m scripts.freeze_preregistration --check    # compare the working tree with the newest frozen record
  python -m scripts.freeze_preregistration --record-deviation "reason"   # log a post-freeze change of a tracked file

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


def _deviations() -> list:
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "deviations_*.json"))):
        try:
            out.append(json.load(open(f)))
        except (OSError, ValueError):
            pass
    return out


def check() -> int:
    path = _latest()
    if not path:
        print("no frozen record found: run without --check first")
        return 2
    frozen, now = json.load(open(path)), build_record()
    recorded = {(p, v["sha256_now"]): d["reason"] for d in _deviations() if d.get("frozen_record") == os.path.basename(path)
                for p, v in d.get("changed_files", {}).items()}
    changed, noted = [], []
    for p, h in frozen["file_sha256"].items():
        if now["file_sha256"].get(p) != h:
            why = recorded.get((p, now["file_sha256"].get(p)))
            (noted if why else changed).append(f"{p}: file changed" + (f"  [recorded deviation: {why}]" if why else ""))
    if frozen["equivalence"] != now["equivalence"]:
        changed.append("equivalence margins / alpha changed (decision rules: never a quiet deviation)")
    if frozen["seeds"] != now["seeds"]:
        changed.append("seed list changed")
    print(f"frozen record: {os.path.basename(path)} (commit {frozen.get('git_commit')})")
    if noted:
        print("changes recorded as deviations:\n  " + "\n  ".join(noted))
    print("UNRECORDED changes:\n  " + "\n  ".join(changed) if changed else
          ("no unrecorded changes since the freeze" if noted else "nothing changed since the freeze"))
    return 1 if changed else 0


def record_deviation(reason: str) -> int:
    """Log the current state of every tracked file that differs from the freeze, with a reason. The margins, alpha and
    seeds must be unchanged: those are decision rules and cannot be recorded away."""
    path = _latest()
    if not path:
        print("no frozen record found: nothing to deviate from")
        return 2
    frozen, now = json.load(open(path)), build_record()
    files = {p: {"sha256_frozen": h, "sha256_now": now["file_sha256"].get(p)}
             for p, h in frozen["file_sha256"].items() if now["file_sha256"].get(p) != h}
    rules_changed = frozen["equivalence"] != now["equivalence"] or frozen["seeds"] != now["seeds"]
    rec = {"recorded_utc": now["frozen_utc"], "reason": reason, "frozen_record": os.path.basename(path),
           "git_commit": now["git_commit"], "changed_files": files, "decision_rules_changed": rules_changed}
    out = os.path.join(ROOT, "results", f"deviations_{now['frozen_utc']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x") as fh:
        json.dump(rec, fh, indent=2)
    print(f"recorded {len(files)} changed file(s) -> {out}")
    if rules_changed:
        print("WARNING: margins / alpha / seeds differ from the freeze. That is a change of decision rules, not a "
              "tooling deviation; it must be reported as such.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--record-deviation", metavar="REASON", default=None,
                    help="log tracked files that changed since the freeze, with this reason")
    args = ap.parse_args()
    if args.record_deviation:
        return record_deviation(args.record_deviation)
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
