"""ONE entry point for the whole study: calibrate -> freeze check -> main runs -> three-arm feature-map -> matrix -> hypothesis tests
-> temporal validation -> exploratory screen. Resumable: every stage records its outcome in results/study_driver_state.json.

    python -m scripts.run_study --plan                       # what would run, in order, with the exact commands
    python -m scripts.run_study --n-runs 10 --base-seed 53 --workers 4 --label "Amendment 2" --new-experiment \\
                                --record-deviation "Amendment 2: H3 positive control, H5 calibrated detector" --apply-calibration
    python -m scripts.run_study                              # re-launch: finished stages are skipped, the main stage resumes
    python -m scripts.run_study --from hypothesis            # re-run the analysis only
    python -m scripts.run_study --status

Stages (in this order)
  calibrate    H3 poisoning strength: scripts.calibrate_poisoning on PILOT seeds; --apply-calibration writes the value into
               the config (or use --allow-uncalibrated to run with the provisional value, flagged in every row)
  freeze       pre-registration drift check AFTER calibration (the config edit must be covered); --record-deviation "<reason>"
               records the Amendment once. Margins, alpha and seeds can never be recorded away.
  main         scripts.run_full_study (balance audit, runs, experiment id); idempotent, so always safe to re-run
  feature_map  scripts.run_feature_map_experiment at each --feature-map-qubits (three arms, paired seeds)
  matrix       scripts.run_three_arm_matrix (matched and bracketed resources)
  hypothesis   scripts.run_hypothesis_tests on the experiment, with the naive baseline and the three-arm files
  temporal     scripts.run_temporal_validation (optional: needs saved per-row forecasts)
  screen       scripts.run_mediation_screen (EXPLORATORY, optional, never a verdict)
Not included on purpose: the entanglement-dose calibration (scripts.calibrate_entanglement_dose) until the circuit wiring is
verified under PennyLane; run it by hand after its prerequisites.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STAGES = ["calibrate", "freeze", "main", "feature_map", "matrix", "hypothesis", "temporal", "screen"]
OPTIONAL = {"temporal", "screen"}
STATE_PATH = os.path.join(ROOT, "results", "study_driver_state.json")
PY = sys.executable


# ------------------------------------------------------------------ config helpers (pure, tested) ----
def poisoning_section(cfg_text: str) -> dict:
    """{'protocol':..., 'calibrated': bool, 'shift_sd': float|None} from raw YAML text, without needing PyYAML."""
    m = re.search(r"(?m)^poisoning:\s*\r?\n((?:[ \t]+.*\r?\n?|\s*\r?\n)*)", cfg_text)
    block = m.group(1) if m else ""
    def grab(key):
        mm = re.search(rf"(?m)^[ \t]+{key}:\s*\"?([^\s#\"]+)", block)
        return mm.group(1) if mm else None
    return {"present": bool(m), "protocol": grab("protocol"),
            "calibrated": (grab("calibrated") or "false").lower() == "true",
            "shift_sd": float(grab("shift_sd")) if grab("shift_sd") else None}


def needs_calibration(cfg_text: str) -> bool:
    s = poisoning_section(cfg_text)
    return bool(s["present"] and (s["protocol"] or "").startswith("v2") and not s["calibrated"])


def apply_calibration(cfg_text: str, shift_sd: float) -> str:
    """Set poisoning.shift_sd and calibrated: true inside the `poisoning:` block only; line endings preserved."""
    m = re.search(r"(?m)^poisoning:\s*\r?\n((?:[ \t]+.*\r?\n?|\s*\r?\n)*)", cfg_text)
    if not m:
        raise ValueError("no `poisoning:` section in the config")
    block = m.group(1)
    new, n1 = re.subn(r"(?m)^([ \t]+shift_sd:\s*)[0-9.eE+-]+", lambda x: f"{x.group(1)}{shift_sd!r}", block, count=1)
    new, n2 = re.subn(r"(?m)^([ \t]+calibrated:\s*)(false|False)", r"\1true", new, count=1)
    if n1 != 1 or n2 != 1:
        raise ValueError("could not find both `shift_sd:` and `calibrated: false` in the poisoning block")
    return cfg_text[: m.start(1)] + new + cfg_text[m.end(1):]


def latest_calibration(results_dir: str) -> dict | None:
    files = sorted(glob.glob(os.path.join(results_dir, "poisoning_calibration_*.json")))
    return json.load(open(files[-1], encoding="utf8")) if files else None


# ------------------------------------------------------------------ driver ----
class Driver:
    def __init__(self, args, runner=subprocess.run, state_path=STATE_PATH, config_path=None, results_dir=None):
        self.a, self.run, self.state_path = args, runner, state_path
        self.config_path = config_path or os.path.join(ROOT, args.config)
        self.results_dir = results_dir or os.path.join(ROOT, "results")
        self.state = self._load()
        self.experiment_id = args.experiment_id

    # state ------------------------------------------------------------
    def _load(self) -> dict:
        try:
            return json.load(open(self.state_path, encoding="utf8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf8") as fh:
            json.dump(self.state, fh, indent=2)
        os.replace(tmp, self.state_path)

    def _mark(self, stage, status, rc=None, cmds=None, note="", seconds=0.0):
        self.state[stage] = {"status": status, "rc": rc, "utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                             "seconds": round(seconds, 1), "commands": cmds or [], "note": note,
                             "experiment_id": self.experiment_id}
        self._save()

    # commands ---------------------------------------------------------
    def sh(self, cmd: list[str]) -> int:
        print("[study] $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        return self.run(cmd, cwd=ROOT).returncode

    def seeds(self):
        return [str(s) for s in range(self.a.fm_base_seed, self.a.fm_base_seed + self.a.fm_n_seeds)]

    def fm_files(self):
        return [os.path.join("results", f"feature_map_advantage_q{q}.json") for q in self.a.feature_map_qubits]

    def commands(self, stage: str) -> list[list[str]]:
        a = self.a
        if stage == "freeze":
            return [[PY, "-m", "scripts.freeze_preregistration", "--check"]]
        if stage == "calibrate":
            return [[PY, "-m", "scripts.calibrate_poisoning"]]
        if stage == "main":
            c = [PY, "-m", "scripts.run_full_study", "--no-hypothesis-tests", "--n-runs", str(a.n_runs),
                 "--base-seed", str(a.base_seed), "--workers", str(a.workers), "--config", a.config,
                 "--no-prepare"]      # calibrate and freeze are separate stages of this driver
            if a.label: c += ["--label", a.label]
            if a.new_experiment: c += ["--new-experiment"]
            if a.experiment_id: c += ["--experiment-id", a.experiment_id]
            if a.conditions: c += ["--conditions", a.conditions]
            else: c += ["--full"]          # run_study is the full pre-registered driver; run_full_study alone defaults to the reduced plan
            if a.no_auto_adopt: c += ["--no-auto-adopt"]
            if a.legacy: c += ["--legacy"]
            if a.allow_uncalibrated: c += ["--allow-uncalibrated"]
            return [c + shlex.split(a.main_extra or "")]
        if stage == "feature_map":
            return [[PY, "-m", "scripts.run_feature_map_experiment", "--seeds", *self.seeds(), "--n-qubits", str(q),
                     "--n-layers", "3", "--out", out] for q, out in zip(a.feature_map_qubits, self.fm_files())]
        if stage == "matrix":
            return [[PY, "-m", "scripts.run_three_arm_matrix", "--seeds", *self.seeds(), "--n-qubits",
                     str(a.feature_map_qubits[0]), "--out", os.path.join("results", "three_arm_matrix.json")]]
        if stage == "hypothesis":
            c = [PY, "-m", "scripts.run_hypothesis_tests", "--feature-map-json", *self.fm_files()]
            if self.experiment_id: c += ["--experiment-id", self.experiment_id]
            if a.pool_experiments: c += ["--pool-experiments"]
            if a.run_confirmatory: c += ["--run-confirmatory"]
            if a.rerun_reason: c += ["--rerun-reason", a.rerun_reason]
            return [c]
        if stage == "temporal":
            if not self.experiment_id:
                return []          # needs an experiment id and saved per-row forecasts; nothing to run without them
            return [[PY, "-m", "scripts.run_temporal_validation", "--experiment-id", self.experiment_id]]
        if stage == "screen":
            c = [PY, "-m", "scripts.run_mediation_screen"]
            return [c + (["--experiment", self.experiment_id] if self.experiment_id else [])]
        raise KeyError(stage)

    # stages -----------------------------------------------------------
    def do_freeze(self) -> tuple[int, str]:
        rc = self.sh(self.commands("freeze")[0])
        if rc != 0:
            if not self.a.record_deviation:
                return 2, ("tracked files differ from the pre-registration freeze. If that is the intended Amendment, rerun with "
                           "--record-deviation \"<reason>\" (margins, alpha and seeds cannot be recorded away).")
            rc2 = self.sh([PY, "-m", "scripts.freeze_preregistration", "--record-deviation", self.a.record_deviation])
            if rc2 != 0:
                return rc2, "recording the deviation failed (decision rules changed, or no frozen record exists)"
        return 0, "ok"

    def do_calibrate(self) -> tuple[int, str]:
        text = open(self.config_path, encoding="utf8", newline="").read()
        if not needs_calibration(text):
            return 0, "not needed (already calibrated, or the legacy protocol is configured)"
        if self.a.allow_uncalibrated:
            return 0, "skipped by --allow-uncalibrated: rows carry poisoning_calibrated=False and shift_sd is the provisional value"
        self.sh(self.commands("calibrate")[0])
        cal = latest_calibration(self.results_dir)
        if not cal or cal.get("chosen_shift_sd") is None:
            return 1, ("no shift in the grid reaches the target on the real-only positive control: the head has nothing to damage on "
                       "these features, so H3 is NOT decidable. Set poisoning.protocol: legacy or report H3 as not decidable.")
        if not self.a.apply_calibration:
            return 3, (f"calibration chose shift_sd = {cal['chosen_shift_sd']}. Rerun with --apply-calibration to write it (and "
                       f"calibrated: true) into {self.a.config}, or paste it yourself.")
        new = apply_calibration(text, float(cal["chosen_shift_sd"]))
        with open(self.config_path, "w", encoding="utf8", newline="") as fh:
            fh.write(new)
        return 0, f"wrote shift_sd = {cal['chosen_shift_sd']} and calibrated: true to {self.a.config}"

    def _generic(self, stage: str) -> tuple[int, str]:
        cmds = self.commands(stage)
        if not cmds:
            return 0, f"{stage}: nothing to run (no experiment id)"
        rcs = [self.sh(c) for c in cmds]
        return max(rcs), ""

    def after_main(self):
        if not self.experiment_id:
            try:
                from src.utils import experiment as exp
                self.experiment_id = exp.get_active()
            except Exception:  # noqa: BLE001 - the analysis stages still work without an id (legacy mode)
                self.experiment_id = None

    def selected(self) -> list[str]:
        a = self.a
        stages = STAGES[:]
        if a.only:
            stages = [s for s in stages if s in a.only]
        if a.from_stage:
            stages = stages[stages.index(a.from_stage):] if a.from_stage in stages else stages
        return [s for s in stages if s not in set(a.skip or [])]

    def execute(self) -> int:
        for stage in self.selected():
            prev = self.state.get(stage, {})
            if prev.get("status") == "ok" and stage != "main" and not self.a.force:
                print(f"[study] {stage}: already ok ({prev.get('utc')}), skipping (--force or --from {stage} to redo)")
                if stage == "main":
                    self.after_main()
                continue
            print(f"\n[study] ===== {stage} =====", flush=True)
            t0 = time.time()
            if stage == "hypothesis" or stage in ("temporal", "screen"):
                self.after_main()
            fn = {"freeze": self.do_freeze, "calibrate": self.do_calibrate}.get(stage, lambda s=stage: self._generic(s))
            rc, note = fn()
            if stage == "main":
                self.after_main()
            cmds = [" ".join(c) for c in self.commands(stage)] if stage not in ("freeze", "calibrate") else []
            if rc == 0:
                self._mark(stage, "ok", 0, cmds, note, time.time() - t0)
                if note: print(f"[study] {stage}: {note}")
                continue
            if stage in OPTIONAL:
                self._mark(stage, "failed_optional", rc, cmds, note or "optional stage failed; the study continues", time.time() - t0)
                print(f"[study] {stage}: failed (rc={rc}) but it is optional, continuing")
                continue
            self._mark(stage, "failed", rc, cmds, note, time.time() - t0)
            print(f"[study] STOP at {stage} (rc={rc}). {note}")
            return rc or 1
        print("\n[study] all selected stages finished")
        return 0

    def print_plan(self):
        print("Study plan (nothing is executed):")
        for st in self.selected():
            print(f"\n  [{st}]" + ("  (optional)" if st in OPTIONAL else ""))
            if st == "calibrate":
                print("    only if poisoning.protocol is v2 and calibrated is false; " + " ".join(self.commands(st)[0]))
            elif st == "freeze":
                print("    " + " ".join(self.commands(st)[0]) + "   (+ --record-deviation when you pass it)")
            else:
                for c in self.commands(st):
                    print("    " + " ".join(shlex.quote(x) for x in c))

    def print_status(self):
        if not self.state:
            print("no stage has run yet")
        for st in STAGES:
            s = self.state.get(st)
            print(f"  {st:<12} {s['status'] if s else '-':<16} {s['utc'] if s else '':<18} {s.get('note','')[:70] if s else ''}")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("main runs (passed to scripts.run_full_study)")
    g.add_argument("--n-runs", type=int, default=10)
    g.add_argument("--base-seed", type=int, default=43)
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--config", default="config/default_config.yaml")
    g.add_argument("--label", default="")
    g.add_argument("--new-experiment", action="store_true")
    g.add_argument("--experiment-id", default=None)
    g.add_argument("--conditions", default=None)
    g.add_argument("--no-auto-adopt", action="store_true")
    g.add_argument("--legacy", action="store_true")
    g.add_argument("--main-extra", default="", help="extra raw arguments for run_full_study, e.g. '--levels 0,2,4,6'")
    f = ap.add_argument_group("three-arm feature-map study")
    f.add_argument("--feature-map-qubits", type=int, nargs="+", default=[8, 12])
    f.add_argument("--fm-n-seeds", type=int, default=10)
    f.add_argument("--fm-base-seed", type=int, default=0)
    p = ap.add_argument_group("pre-registration / calibration")
    p.add_argument("--record-deviation", default=None, metavar="REASON")
    p.add_argument("--apply-calibration", action="store_true")
    p.add_argument("--allow-uncalibrated", action="store_true")
    c = ap.add_argument_group("control")
    c.add_argument("--only", nargs="+", choices=STAGES)
    c.add_argument("--from", dest="from_stage", choices=STAGES)
    c.add_argument("--skip", nargs="+", choices=STAGES)
    c.add_argument("--force", action="store_true", help="re-run stages already recorded as ok")
    c.add_argument("--pool-experiments", action="store_true")
    c.add_argument("--run-confirmatory", action="store_true",
                   help="hypothesis stage: opt in to the confirmatory analysis (otherwise it STOPS at the gate)")
    c.add_argument("--rerun-reason", default=None, help="hypothesis stage: disclosure text for a repeated confirmatory run")
    c.add_argument("--plan", action="store_true")
    c.add_argument("--status", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    d = Driver(a)
    if a.status:
        d.print_status(); return 0
    if a.plan:
        d.print_plan(); return 0
    return d.execute()


if __name__ == "__main__":
    raise SystemExit(main())
