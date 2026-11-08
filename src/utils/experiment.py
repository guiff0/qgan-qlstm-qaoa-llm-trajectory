"""Experiment identity: ONE id that groups every run (run_id) of one study, so two studies are never confused.

    experiment_id   EXP-<YYYYMMDD>-<MACHINE>-<4 hex>      one per study (e.g. one 80-run launch, on one machine)
      pipeline_run_id  P<UTC>-<git sha7>                  one per launch/relaunch of a script (already existed)
        run_id         <condition slug>_<UTC second>      one per training run (already existed; format unchanged)

Why it exists. Before this, "which seeds are done" was answered from ALL rows of results/all_results.csv and from
state files shared per condition, and the analysis kept the newest row per (condition, seed). A second 80-run study
(same conditions, same seeds, another machine or a later day) would therefore be seen as "already finished", or one
of the two rows per (condition, seed) would be silently dropped. With an experiment id:
  * every results row carries experiment_id and machine (logging_utils);
  * resume/remaining counts only that experiment's rows; state and logs live in per-experiment folders;
  * analysis refuses to mix experiments unless told to, and de-duplicates within an experiment only;
  * results/experiments/<id>/manifest.json holds the plan (conditions, seeds, config hash, code hashes, machine,
    git commit) and runs.jsonl lists every run of that experiment, with its run_id once known.

Rows written before this existed have no experiment_id. They are treated as the experiment "unassigned" until
`python -m scripts.experiments adopt` stamps them.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import socket
import subprocess
from collections import Counter
from datetime import datetime, timezone

ENV_EXPERIMENT_ID = "EXPERIMENT_ID"
UNASSIGNED = "unassigned"
EXPERIMENT_ROOT = os.path.join("results", "experiments")
ID_RE = re.compile(r"^EXP-\d{8}-[A-Z0-9]{1,12}-[0-9a-f]{4}$")
CODE_FILES = ["config/default_config.yaml", "PREREGISTRATION.md", "scripts/run_hypothesis_tests.py",
              "src/utils/generator_path.py", "src/utils/capacity_match.py", "src/baselines/qgan_llm.py",
              "src/baselines/classical_gan_llm.py", "src/experiments/run_all.py", "src/utils/logging_utils.py"]


class ExperimentError(RuntimeError):
    """A request that could silently mix or overwrite experiments."""


# ------------------------------------------------------------------ ids ----
def machine_slug(name: str | None = None) -> str:
    return (re.sub(r"[^A-Za-z0-9]", "", name or socket.gethostname()).upper()[:12]) or "HOST"


def new_experiment_id(machine: str | None = None, now: datetime | None = None, rand: str | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"EXP-{stamp}-{machine_slug(machine)}-{rand or secrets.token_hex(2)}"


def is_valid_id(eid: str) -> bool:
    return bool(ID_RE.match(str(eid)))


def current_experiment_id() -> str:
    """The id of the running study (env EXPERIMENT_ID), or 'unassigned' when none was set."""
    return os.environ.get(ENV_EXPERIMENT_ID, "").strip() or UNASSIGNED


def experiment_dir(eid: str, root: str = EXPERIMENT_ROOT) -> str:
    return os.path.join(root, eid)


def _sha(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def plan_hash(conditions, seeds, config_sha: str | None = None) -> str:
    """Same plan -> same hash on any machine, so a replication is recognisable as one (ids still differ)."""
    return hashlib.sha256(json.dumps([sorted(conditions), sorted(int(s) for s in seeds), config_sha],
                                     sort_keys=True).encode()).hexdigest()[:16]


def _git() -> dict:
    out = {"commit": None, "dirty": None}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        out["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:  # noqa: BLE001
        pass
    return out


# ------------------------------------------------------------- manifest ----
def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def create_experiment(eid: str, *, conditions, seeds, base_seed: int, label: str = "", config_path: str | None = None,
                      machine: str | None = None, parent: str | None = None, root: str = EXPERIMENT_ROOT) -> dict:
    """Create results/experiments/<eid>/manifest.json, or return the existing one when the plan is identical.
    Re-using an id for a DIFFERENT plan raises: that is exactly the confusion this module prevents."""
    if not is_valid_id(eid):
        raise ExperimentError(f"{eid!r} is not an experiment id (expected EXP-YYYYMMDD-MACHINE-xxxx)")
    d = experiment_dir(eid, root)
    mpath = os.path.join(d, "manifest.json")
    conditions, seeds = list(conditions), [int(s) for s in seeds]
    cfg_sha = _sha(config_path) if config_path else None
    if os.path.isfile(mpath):
        old = load_manifest(eid, root)
        if sorted(old["conditions"]) != sorted(conditions) or sorted(old["seeds"]) != sorted(seeds):
            raise ExperimentError(
                f"{eid} already exists with a different plan ({len(old['conditions'])} conditions, seeds "
                f"{min(old['seeds'])}-{max(old['seeds'])}). Use --new-experiment for a different study.")
        return old
    os.makedirs(d, exist_ok=True)
    man = {"schema": 1, "experiment_id": eid, "label": label, "parent_experiment": parent,
           "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
           "machine": {"name": machine or socket.gethostname(), "platform": platform.platform(),
                       "python": platform.python_version()},
           "git": _git(), "conditions": conditions, "seeds": seeds, "base_seed": int(base_seed),
           "n_runs": len(seeds), "planned_runs": len(conditions) * len(seeds),
           "config_path": config_path, "config_sha256": cfg_sha,
           "plan_hash": plan_hash(conditions, seeds, cfg_sha),
           "code_sha256": {p: _sha(p) for p in CODE_FILES}}
    pre = sorted(f for f in os.listdir("results") if f.startswith("preregistration_")) if os.path.isdir("results") else []
    if pre:
        man["preregistration_record"] = {"file": pre[-1], "sha256": _sha(os.path.join("results", pre[-1]))}
    _atomic_write(mpath, json.dumps(man, indent=2))
    return man


def load_manifest(eid: str, root: str = EXPERIMENT_ROOT) -> dict:
    try:
        with open(os.path.join(experiment_dir(eid, root), "manifest.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ExperimentError(f"no manifest for {eid} under {root}") from None


def list_experiments(root: str = EXPERIMENT_ROOT) -> list[dict]:
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, name, "manifest.json")):
                out.append(load_manifest(name, root))
    return out


def set_active(eid: str, root: str = EXPERIMENT_ROOT) -> None:
    os.makedirs(root, exist_ok=True)
    _atomic_write(os.path.join(root, "ACTIVE"), eid + "\n")


def get_active(root: str = EXPERIMENT_ROOT) -> str | None:
    try:
        with open(os.path.join(root, "ACTIVE"), encoding="utf-8") as f:
            eid = f.read().strip()
        return eid if is_valid_id(eid) and os.path.isfile(os.path.join(root, eid, "manifest.json")) else None
    except OSError:
        return None


# ----------------------------------------------------------------- runs ----
def register_run(eid: str, record: dict, root: str = EXPERIMENT_ROOT) -> None:
    """Append one JSON line to results/experiments/<eid>/runs.jsonl (process-safe)."""
    from src.utils.io import file_lock
    d = experiment_dir(eid, root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "runs.jsonl")
    rec = {"experiment_id": eid, "logged_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), **record}
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def read_runs(eid: str, root: str = EXPERIMENT_ROOT) -> list[dict]:
    path = os.path.join(experiment_dir(eid, root), "runs.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def find_run_id(csv_path: str, eid: str, slug: str, seed: int) -> str | None:
    """Newest run_id of (condition slug, seed) among this experiment's rows, or None."""
    import csv
    best = None
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("experiment_id") or UNASSIGNED) != eid or not row.get("run_id"):
                continue
            if re.sub(r"_\d{8}T\d{6}Z$", "", row["run_id"]) != slug:
                continue
            try:
                if int(float(row.get("seed", ""))) != int(seed):
                    continue
            except ValueError:
                continue
            if best is None or row["run_id"] > best:
                best = row["run_id"]
    return best


# --------------------------------------------------------- analysis side ----
def row_experiments(df):
    """Series of experiment ids for a results DataFrame; rows with none are 'unassigned'."""
    if "experiment_id" not in df.columns:
        return df.index.to_series().map(lambda _: UNASSIGNED)
    return df["experiment_id"].fillna("").astype(str).str.strip().replace("", UNASSIGNED)


def experiments_in(df) -> dict:
    return dict(Counter(row_experiments(df)))


def select_experiments(df, requested=None, pool: bool = False):
    """Rows of the requested experiment(s). With no request, the file must contain exactly one experiment.
    Returns (filtered_df, ids). Raises ExperimentError rather than silently mixing studies."""
    present = experiments_in(df)
    listing = ", ".join(f"{k} ({v} rows)" for k, v in sorted(present.items())) or "no rows"
    requested = [r for r in (requested or []) if r]
    if not requested:
        if len(present) > 1:
            raise ExperimentError(
                f"results contain {len(present)} experiments: {listing}. Pass --experiment-id <one of them> "
                f"(or several with --pool-experiments) so two studies are never analysed as one.")
        return df, list(present)
    missing = [r for r in requested if r not in present]
    if missing:
        raise ExperimentError(f"experiment(s) {missing} not found in results; present: {listing}")
    if len(requested) > 1 and not pool:
        raise ExperimentError(f"{len(requested)} experiments requested; add --pool-experiments to analyse them together")
    keep = row_experiments(df).isin(requested)
    return df[keep].reset_index(drop=True), requested


def seed_overlap(df, slug_of=lambda rid: re.sub(r"_\d{8}T\d{6}Z$", "", str(rid))) -> dict:
    """{condition: [seeds present in more than one experiment]} -- paired-by-seed tests are ambiguous for these."""
    if "seed" not in df.columns or "run_id" not in df.columns:
        return {}
    t = df.assign(_e=row_experiments(df), _c=df["run_id"].map(slug_of))
    out = {}
    for (c, s), g in t.dropna(subset=["seed"]).groupby(["_c", "seed"]):
        if g["_e"].nunique() > 1:
            out.setdefault(c, []).append(int(s))
    return out


def supports_stamping(repo_root: str = ".") -> bool:
    """True when src/utils/logging_utils.py writes experiment_id (checked in the source, nothing is imported)."""
    try:
        with open(os.path.join(repo_root, "src", "utils", "logging_utils.py"), encoding="utf-8") as f:
            return "experiment_id" in f.read()
    except OSError:
        return False
