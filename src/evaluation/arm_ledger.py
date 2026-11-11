"""THREE-ARM LEDGER and CHRONOLOGY: which of the three arms (naive/linear baseline, classical, quantum) have results,
which of them have a checkpoint, and in what order things happened.

WHY (user + reviewer: "I do not see the baseline checkpoint / the three-arm method / the chronology")
  The three arms of the methodology figure are
      baseline   y = w0.x + b0                 ridge on the inputs only          -> feature_map arm "linear"  (+ persistence)
      classical  y = wc.phi_classical(x) + bc  random features                   -> feature_map arm "classical_rf"; Classical GAN-LLM
      quantum    y = wq.phi_quantum(x) + bq    PQC features                      -> feature_map arm "quantum"; QGAN-LLM
  The baseline has NO checkpoint by construction: persistence is computed from the data and the linear head is a closed-form
  ridge fit in a fraction of a second, so there is nothing to resume. Only the GAN conditions train for hours and write
  models/*.ckpt. The ledger says so explicitly, per arm, instead of leaving the absence to be misread as "not run".

Pure functions over the results frame and the folders; torch-free (src/utils/ckpt_inspect.py).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from src.utils import ckpt_inspect as ci

_RUN_UTC = re.compile(r"_(\d{8}T\d{6}Z)$")
_FM_ARMS = {"linear": "baseline", "classical_rf": "classical", "quantum": "quantum"}


def run_utc(run_id: str) -> str | None:
    m = _RUN_UTC.search(str(run_id))
    if not m:
        return None
    t = m.group(1)
    return f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:15]}Z"


def _config_of(run_id: str) -> str:
    return _RUN_UTC.sub("", str(run_id))


def checkpoint_inventory(models_dir: str) -> dict:
    """Every checkpoint under models_dir, grouped by prefix, with lock state."""
    recs = []
    for p in sorted(glob.glob(os.path.join(models_dir, "*.ckpt"))):
        r = ci.describe(p)
        r["lock"] = ci.lock_info(p)
        r["synth_cache"] = ci.synth_cache(p)
        recs.append(r)
    by = {}
    for r in recs:
        b = by.setdefault(r["prefix"], {"complete": 0, "incomplete": 0, "unreadable": 0, "live_lock": 0, "stale_lock": 0})
        b["complete" if r.get("completed") else "unreadable" if r["state"] == "UNREADABLE" else "incomplete"] += 1
        if r.get("lock"):
            b["live_lock" if r["lock"]["alive"] else "stale_lock"] += 1
    return {"models_dir": models_dir, "checkpoints": recs, "by_prefix": by}


def feature_map_status(paths) -> dict:
    """Seeds per arm found in results/feature_map_advantage_q*.json (the closed-form three-arm experiment)."""
    out = {}
    for p in paths:
        try:
            with open(p, encoding="utf8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            out[os.path.basename(p)] = {"error": "unreadable"}
            continue
        rows = d.get("per_run", [])
        arms = {a: sorted({r.get("seed") for r in rows if r.get("arm") == a}) for a in _FM_ARMS}
        out[os.path.basename(p)] = {"arms": {a: len(s) for a, s in arms.items()}, "seeds": sorted({r.get("seed") for r in rows}),
                                    "n_qubits": (d.get("args") or {}).get("n_qubits"),
                                    "walk_forward_windows": (d.get("args") or {}).get("n_windows")}
    return out


def walk_forward_status(results_dir: str, paths=None) -> dict:
    """results/walk_forward_q*.json written by scripts.run_walk_forward: the figure's Window 1..K (train -> test) for the
    three arms with the same seeds. Versioned copies (name__<UTC>.json) are skipped."""
    files = [p for p in (paths or []) if os.path.isfile(p)] or [
        p for p in sorted(glob.glob(os.path.join(results_dir, "walk_forward_q*.json"))) if "__" not in os.path.basename(p)]
    out = {}
    for p in files:
        try:
            with open(p, encoding="utf8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            out[os.path.basename(p)] = {"error": "unreadable"}
            continue
        a = d.get("args") or {}
        out[os.path.basename(p)] = {"mode": a.get("mode"), "gap": a.get("gap"), "n_qubits": a.get("n_qubits"),
                                    "seeds": a.get("seeds"), "windows": (d.get("summary") or {}).get("windows", []),
                                    "plan": d.get("windows", []), "summary": {k: v for k, v in (d.get("summary") or {}).items()
                                                                              if k != "windows"}}
    return out


def temporal_blocks_status(results_dir: str, experiment_id: str | None = None) -> dict:
    """results/temporal_validation__<experiment>.json: 10 contiguous BLOCKS of the one fixed test period (within-period
    stability from saved predictions). It is NOT walk-forward: the models are not re-fitted per block."""
    out = {}
    pat = f"temporal_validation__{experiment_id}.json" if experiment_id else "temporal_validation__*.json"
    for p in sorted(glob.glob(os.path.join(results_dir, pat))):
        try:
            with open(p, encoding="utf8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        out[os.path.basename(p)] = {c: {"n_seeds": v.get("n_seeds"), "mean_ratio": v.get("mean_ratio"),
                                        "blocks_beating_persistence": v.get("share_of_blocks_beating_persistence")}
                                    for c, v in (d.get("conditions") or {}).items()}
    return out


def arm_table(df, inv: dict, fm: dict) -> list[dict]:
    """One row per arm: what it is, where its result comes from, how many runs, and whether it has a checkpoint."""
    cfg = df["run_id"].map(_config_of) if len(df) and "run_id" in df.columns else None
    n_cfg = (lambda *names: int(cfg.isin(names).sum())) if cfg is not None else (lambda *names: 0)
    persist = int(df["persistence_rmse"].notna().sum()) if "persistence_rmse" in df.columns else 0
    fm_n = {a: sum(v.get("arms", {}).get(a, 0) for v in fm.values()) for a in _FM_ARMS}
    bp = inv["by_prefix"]
    def ck(prefix):
        b = bp.get(prefix)
        return f"{b['complete']} complete / {b['incomplete']} incomplete" if b else "none found in models/"
    return [
        {"arm": "baseline", "what": "persistence (naive) + ridge on the inputs only", "kind": "closed form, no training",
         "checkpoint": "none by design (nothing to resume)", "runs_in_all_results": persist,   # rows that carry a persistence_rmse (the naive benchmark)
         "feature_map_seeds": fm_n["linear"],
         "where": "persistence_rmse column of all_results.csv; feature_map_advantage_q*.json arm 'linear'"},
        {"arm": "classical", "what": "Classical GAN-LLM (large + parameter-matched) and random-feature ridge", "kind": "GAN trained for hours",
         "checkpoint": ck("classical_gan_llm"),
         "runs_in_all_results": n_cfg("classical_gan_llm", "classical_gan_llm_matched"), "feature_map_seeds": fm_n["classical_rf"],
         "where": "all_results.csv rows; models/classical_gan_llm_*.ckpt; feature_map arm 'classical_rf'"},
        {"arm": "quantum", "what": "QGAN-LLM (PQC generator) and PQC-feature ridge", "kind": "GAN trained for hours",
         "checkpoint": ck("qgan_llm"),
         "runs_in_all_results": int(cfg.str.startswith("qgan_llm").sum()) if cfg is not None else 0,
         "feature_map_seeds": fm_n["quantum"],
         "where": "all_results.csv rows; models/qgan_llm_*.ckpt; feature_map arm 'quantum'"},
    ]


def chronology(df, results_dir: str, snapshots_dir: str | None = None) -> list[dict]:
    """Everything with a UTC time, in order: runs, calibration, pre-registration freezes, deviations, snapshots,
    confirmatory runs. Lets you see which runs predate which decision (the SME's 'old rows' problem)."""
    ev = []
    if len(df) and "run_id" in df.columns:
        for _, r in df.iterrows():
            t = run_utc(r["run_id"])
            if t:
                ev.append({"utc": t, "kind": "run", "label": f"{_config_of(r['run_id'])} seed {r.get('seed')}",
                           "detail": {"forecast_path": r.get("forecast_path"), "poisoning_protocol": r.get("poisoning_protocol"),
                                      "experiment_id": r.get("experiment_id")}})
    for pat, kind in (("preregistration_*.json", "freeze"), ("deviations_*.json", "deviation"),
                      ("poisoning_calibration_*.json", "calibration"), ("poisoning_gap_diagnostic_*.json", "diagnostic")):
        for f in sorted(glob.glob(os.path.join(results_dir, pat))):
            m = re.search(r"_(\d{8}T\d{6}Z)\.json$", f)
            if m:
                t = m.group(1)
                ev.append({"utc": f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:15]}Z", "kind": kind,
                           "label": os.path.basename(f), "detail": {}})
    sd = snapshots_dir or os.path.join(results_dir, "snapshots")
    try:
        with open(os.path.join(sd, "manifest.json"), encoding="utf8") as fh:
            for e in json.load(fh):
                t = e["utc"]
                ev.append({"utc": f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:15]}Z", "kind": "snapshot",
                           "label": f"{e['snapshot']} sha256 {e['sha256'][:12]}...", "detail": {"rows": e.get("rows")}})
    except (OSError, ValueError):
        pass
    return sorted(ev, key=lambda e: (e["utc"], e["kind"] != "run"))


def label_runs_by_decisions(events: list[dict]) -> dict:
    """How many runs came BEFORE the first freeze / the calibration / the last freeze, from the event list."""
    def first(kind):
        ts = [e["utc"] for e in events if e["kind"] == kind]
        return (min(ts), max(ts)) if ts else (None, None)
    runs = [e for e in events if e["kind"] == "run"]
    f0, f1 = first("freeze")
    c0, c1 = first("calibration")
    cnt = lambda pred: int(sum(1 for r in runs if pred(r["utc"])))  # noqa: E731
    return {"n_runs": len(runs),
            "first_run": runs[0]["utc"] if runs else None, "last_run": runs[-1]["utc"] if runs else None,
            "first_freeze": f0, "last_freeze": f1, "calibration": c1,
            "runs_before_first_freeze": cnt(lambda t: f0 is not None and t < f0) if f0 else None,
            "runs_before_calibration": cnt(lambda t: t < c1) if c1 else None,
            "runs_after_last_freeze": cnt(lambda t: t >= f1) if f1 else None}


def build_ledger(df, models_dir="models", results_dir="results", feature_map_paths=None, experiment_id=None,
                 walk_forward_paths=None) -> dict:
    fm_paths = [p for p in (feature_map_paths or []) if os.path.isfile(p)]
    if not fm_paths:       # versioned copies (name__<UTC>.json) are skipped so a seed is never counted twice
        fm_paths = [p for p in sorted(glob.glob(os.path.join(results_dir, "feature_map_advantage*.json")))
                    if "__" not in os.path.basename(p)]
    inv = checkpoint_inventory(models_dir)
    fm = feature_map_status(fm_paths)
    ev = chronology(df, results_dir)
    wfs = walk_forward_status(results_dir, walk_forward_paths)
    return {"arms": arm_table(df, inv, fm), "checkpoints": inv, "feature_map": fm, "walk_forward": wfs,
            "temporal_blocks": temporal_blocks_status(results_dir, experiment_id if isinstance(experiment_id, str) else None),
            "chronology": ev,
            "chronology_summary": label_runs_by_decisions(ev),
            "three_arm_ran": bool(fm) and all(sum(v.get("arms", {}).values()) > 0 for v in fm.values() if "arms" in v)}


def _sk(v):
    return "   n/a " if v is None else f"{v:+.4f}"


def format_walk_forward(led: dict) -> list[str]:
    L = ["WALK-FORWARD WINDOWS (rolling origin: every window re-fits each arm on EARLIER rows only, then scores the next window; "
         "same seeds for every arm)"]
    wfs = led.get("walk_forward") or {}
    if not wfs:
        L.append("  NOT RUN: no results/walk_forward_q*.json. Plan it with   python -m scripts.run_walk_forward --dry-run   "
                 "then run   python -m scripts.run_walk_forward --windows 5 --test-len 20000 --seeds 0 1 2 3 4")
    for name, v in wfs.items():
        if "error" in v:
            L.append(f"  {name}: {v['error']}")
            continue
        sm = v["summary"]
        L.append(f"  {name}: {len(v['windows'])} window(s), seeds {v.get('seeds')}, {v.get('mode')}, embargo {v.get('gap')} row(s)")
        L.append(f"  {'win':>3} {'train rows':>21} {'test rows':>21} {'seeds':>5}   skill vs persistence: baseline / classical / quantum"
                 f"   quantum beats linear | classical (seeds)")
        plan = {p["window"]: p for p in v.get("plan", [])}
        for w in v["windows"]:
            pl = plan.get(w["window"], {})
            tr = f"{pl.get('train_start', 0):,}-{pl.get('train_end', 0):,}" if pl else f"n={w['n_train']:,}"
            te = f"{w['test_start']:,}-{w['test_end']:,}"
            sk = w["skill_median"]
            qb = w["quantum_beats"]
            L.append(f"  {w['window']:>3} {tr:>21} {te:>21} {w['n_seeds']:>5}   "
                     f"{_sk(sk.get('linear'))} / {_sk(sk.get('classical_rf'))} / {_sk(sk.get('quantum'))}"
                     f"            {qb.get('linear', 0)}/{qb.get('linear_of', 0)} | {qb.get('classical_rf', 0)}/{qb.get('classical_rf_of', 0)}")
        L.append(f"  -> quantum beats BOTH controls in every seed in {sm.get('windows_quantum_beats_both')} of {sm.get('n_windows_run')} window(s); "
                 f"quantum skill > 0 in {sm.get('windows_quantum_skill_positive')}; baseline (inputs-only) skill > 0 in "
                 f"{sm.get('windows_baseline_skill_positive')}. A window that disagrees is a regime effect, not noise to average away.")
    tb = led.get("temporal_blocks") or {}
    if tb:
        for name, conds in tb.items():
            L.append(f"  within-test-period blocks ({name}; NOT walk-forward, models are not re-fitted): "
                     + "; ".join(f"{c} {v['n_seeds']} seeds, blocks<persistence "
                                 f"{(v['blocks_beating_persistence'] if v['blocks_beating_persistence'] is not None else float('nan')):.0%}"
                                 for c, v in conds.items()))
    else:
        L.append("  within-test-period blocks: no results/temporal_validation__<experiment>.json "
                 "(python -m scripts.run_temporal_validation --experiment-id <id>)")
    return L


def format_ledger(led: dict, max_events: int = 14) -> str:
    L = ["THREE-ARM LEDGER (baseline / classical / quantum; same seeds fed to every arm)"]
    L.append(f"{'arm':<10}{'runs (csv)':>11}{'feature-map seeds':>19}   checkpoint")
    for a in led["arms"]:
        L.append(f"{a['arm']:<10}{a['runs_in_all_results']:>11}{a['feature_map_seeds']:>19}   {a['checkpoint']}")
        L.append(f"{'':<10}  {a['what']} [{a['kind']}]")
    if not led["feature_map"]:
        L.append("  three-arm feature-map comparison: NO results/feature_map_advantage*.json found -> the paired three-arm test "
                 "has not been run (python -m scripts.run_study --only feature_map). The baseline arm has no checkpoint by design.")
    else:
        for name, v in led["feature_map"].items():
            L.append(f"  {name}: arms {v.get('arms')} seeds {v.get('seeds')}" if "arms" in v else f"  {name}: {v.get('error')}")
    L.extend(format_walk_forward(led))
    inv = led["checkpoints"]
    L.append(f"CHECKPOINTS in {inv['models_dir']}: " + ("; ".join(
        f"{p}: {b['complete']} complete, {b['incomplete']} incomplete, {b['stale_lock']} stale lock(s), {b['live_lock']} live lock(s)"
        for p, b in inv["by_prefix"].items()) or "none"))
    s = led["chronology_summary"]
    L.append(f"CHRONOLOGY: {s['n_runs']} run(s) {s['first_run']} -> {s['last_run']}; first freeze {s['first_freeze']}, last freeze "
             f"{s['last_freeze']}, calibration {s['calibration']}")
    if s["runs_before_calibration"] is not None:
        L.append(f"  {s['runs_before_calibration']} run(s) PREDATE the calibration, "
                 f"{s['runs_before_first_freeze'] if s['runs_before_first_freeze'] is not None else '?'} predate the first freeze record")
    ev = led["chronology"]
    show = ev if len(ev) <= max_events else ev[:5] + [{"utc": "...", "kind": "", "label": f"({len(ev) - 10} more; see JSON)"}] + ev[-5:]
    for e in show:
        L.append(f"  {e['utc']:<21}{e['kind']:<12}{e['label']}")
    return "\n".join(L)
