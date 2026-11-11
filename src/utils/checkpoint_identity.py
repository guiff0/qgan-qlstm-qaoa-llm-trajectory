"""Who does a models/*.ckpt belong to?  A checkpoint is named by a hash of (model config, seed), which says nothing about the
condition. Three independent sources identify one, strongest first, and scripts/list_checkpoints.py reports which it used:

  1. SIDECAR   models/<ckpt>.id.json, written the first time the run saves the checkpoint (condition, seed, UTC). From the
               Nov-11 code on, every new checkpoint has one. `list_checkpoints --write-sidecars` backfills the old ones.
  2. ARCHIVE   run_all copies a finished checkpoint to models/history/<stem>__<run_id>.ckpt. The run_id is
               "<condition slug>_<UTC>" and the results row with that run_id carries the seed and experiment. Needs no torch.
  3. FINGERPRINT recompute hash(config, seed) for every condition x seed and look for a match (needs torch: it imports the
               models). If nothing matches, the code or config changed after the file was written: such a checkpoint will NOT
               resume (the run retrains), which is exactly what you want to know.
Torch-free, so it can run anywhere. Never deletes or modifies a checkpoint.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

_RUN_ID = re.compile(r"^(?P<slug>.+)_(?P<utc>\d{8}T\d{6}Z)(?:_\d+)?$")


def sidecar_path(ckpt_path: str) -> str:
    return ckpt_path + ".id.json"


def write_identity_once(ckpt_path: str, env=None) -> bool:
    """Called from save_checkpoint. Records PIPELINE_CONDITION / PIPELINE_SEED (set by run_all) next to the checkpoint.
    First writer wins; a different condition using the same file is appended to `also_used_by`. Returns True if it wrote.
    Never raises: identification must not be able to break a 12-hour training run."""
    try:
        env = os.environ if env is None else env
        cond, seed = env.get("PIPELINE_CONDITION"), env.get("PIPELINE_SEED")
        if not cond:
            return False
        sp = sidecar_path(ckpt_path)
        rec = {"condition": cond, "seed": int(seed) if seed not in (None, "") else None,
               "first_saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "run_all"}
        if os.path.isfile(sp):
            with open(sp, encoding="utf8") as fh:
                old = json.load(fh)
            if old.get("condition") == cond and old.get("seed") == rec["seed"]:
                return False
            also = old.setdefault("also_used_by", [])
            entry = {"condition": cond, "seed": rec["seed"]}
            if entry in also:
                return False
            also.append(entry)
            rec = old
        tmp = sp + ".tmp"
        with open(tmp, "w", encoding="utf8") as fh:
            json.dump(rec, fh, indent=2)
        os.replace(tmp, sp)
        return True
    except Exception:  # noqa: BLE001
        return False


def read_sidecar(ckpt_path: str) -> dict | None:
    try:
        with open(sidecar_path(ckpt_path), encoding="utf8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def parse_run_id(run_id: str) -> dict | None:
    m = _RUN_ID.match(run_id)
    return {"run_id": run_id, "condition_slug": m["slug"], "utc": m["utc"]} if m else None


def history_index(models_dir: str) -> dict:
    """{live stem (e.g. qgan_llm_046dbbdca0): [ {run_id, condition_slug, utc, file}, ... ]} from models/history/."""
    idx = {}
    for p in sorted(glob.glob(os.path.join(models_dir, "history", "*__*.ckpt"))):
        stem, _, tag = os.path.basename(p)[:-len(".ckpt")].partition("__")
        rid = parse_run_id(tag)
        if rid:
            idx.setdefault(stem, []).append({**rid, "file": os.path.basename(p)})
    return idx


def load_run_table(csv_path: str | None) -> dict:
    """{run_id: {seed, experiment_id, condition_slug}} from results/all_results.csv (optional)."""
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return {}
    if "run_id" not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        rid = str(r["run_id"])
        out[rid] = {"seed": (int(r["seed"]) if "seed" in df.columns and r["seed"] == r["seed"] else None),
                    "experiment_id": (r["experiment_id"] if "experiment_id" in df.columns and r["experiment_id"] == r["experiment_id"] else None)}
    return out


def identify_record(rec: dict, ckpt_path: str, hist: dict, runs: dict, fp_map: dict | None = None) -> dict:
    """Attach {identity: {condition, seed, by, ...}} to one describe() record using the three sources."""
    stem = rec["file"][:-len(".ckpt")]
    sc = read_sidecar(ckpt_path)
    if sc:
        return {"condition": sc.get("condition"), "seed": sc.get("seed"), "by": "sidecar", "also_used_by": sc.get("also_used_by", [])}
    arch = hist.get(stem)
    if arch:
        seeds = sorted({runs.get(a["run_id"], {}).get("seed") for a in arch if runs.get(a["run_id"], {}).get("seed") is not None})
        return {"condition": arch[-1]["condition_slug"], "seed": seeds[-1] if seeds else None, "by": "archive (run_id)",
                "run_ids": [a["run_id"] for a in arch],
                "note": None if seeds else "seed not found: pass --csv results/all_results.csv"}
    if fp_map:
        hit = fp_map.get(rec.get("fingerprint") or rec.get("fingerprint_in_name"))
        if hit:
            return {"condition": hit[0], "seed": hit[1], "by": "fingerprint (recomputed)"}
        return {"condition": None, "seed": None, "by": None,
                "note": "no condition x seed in the plan reproduces this fingerprint: the model config or code changed "
                        "after it was written, so it will not resume (the run retrains)"}
    return {"condition": None, "seed": None, "by": None,
            "note": "no sidecar and no archived run_id; run with --identify (needs torch) or --write-sidecars after identifying"}
