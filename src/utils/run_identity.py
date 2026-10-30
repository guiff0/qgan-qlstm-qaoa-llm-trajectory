"""Run-identity helpers shared by the pipeline, collectors and tests.

run_id = "<slug of model name>_<UTC timestamp>" (see logging_utils.make_run_id).
All QGAN-LLM ablation configurations are written to results/all_results.csv
with model_type == "QGAN-LLM", so the configuration can only be recovered from
the run_id slug. Matching on model_type alone made run_pipeline.py report the
three ablation configs as failed on every run and re-run them each time.
"""
from __future__ import annotations

import re

_TS = re.compile(r"_\d{8}T\d{6}Z$")


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def config_of_run_id(run_id: str) -> str:
    return _TS.sub("", str(run_id))


def latest_per_config(df):
    """One row per configuration (the newest run_id), so repeated evaluations of
    the same checkpoint are not counted as independent observations."""
    out = df.copy()
    out["config"] = out["run_id"].map(config_of_run_id)
    out["_ts"] = out["run_id"].astype(str).str.extract(r"(\d{8}T\d{6}Z)$")[0]
    out = out.sort_values("_ts").drop_duplicates(subset="config", keep="last")
    return out.drop(columns="_ts").reset_index(drop=True)


def latest_per_config_seed(df):
    """One row per (configuration, seed): the newest run_id of each. Re-evaluations of the same
    checkpoint collapse (same config AND seed), but a different seed is an independent
    observation and is kept. Rows without a seed (legacy) are treated as one seed (-1), so they
    behave exactly like latest_per_config."""
    out = df.copy()
    out["config"] = out["run_id"].map(config_of_run_id)
    out["_ts"] = out["run_id"].astype(str).str.extract(r"(\d{8}T\d{6}Z)$")[0]
    out["_seed"] = out["seed"].fillna(-1) if "seed" in out.columns else -1
    out = out.sort_values("_ts").drop_duplicates(subset=["config", "_seed"], keep="last")
    return out.drop(columns=["_ts", "_seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Run history: every run is dated, has a unique id, and nothing a later run
# would overwrite is lost.
#
#   run_id           "<model slug>_<UTC timestamp>"  one per model run. Unique: if the id is
#                    already taken (two runs in the same second, or an existing log) the
#                    timestamp is bumped by one second, so a log is never overwritten.
#   pipeline_run_id  "P<UTC timestamp>-<git sha7>"   one per `python run_pipeline.py` call
#                    (or per standalone script call). Exported through the PIPELINE_RUN_ID
#                    environment variable so every subprocess of one pipeline shares it.
#
# Files that are rewritten in place by design (all_results.csv, chapter4_numbers.json,
# hypothesis_test_results.json, data/pipeline_state.json, models/classical_lstm_best.pt) are
# copied to a `history/` folder next to them, with the stamp in the file name, BEFORE they are
# changed. results/history/all_results_ledger.csv is an append-only copy of every result row.
# ---------------------------------------------------------------------------
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

_TS_FMT = "%Y%m%dT%H%M%SZ"
ENV_PIPELINE_ID = "PIPELINE_RUN_ID"


def utc_stamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now(timezone.utc)).strftime(_TS_FMT)


def _git_sha7() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def pipeline_run_id() -> str:
    """One id per pipeline invocation; subprocesses inherit it via the environment."""
    pid = os.environ.get(ENV_PIPELINE_ID)
    if not pid:
        pid = f"P{utc_stamp()}-{_git_sha7()}"
        os.environ[ENV_PIPELINE_ID] = pid
    return pid


def unique_run_id(model_name: str, log_dir: str = "logs") -> str:
    """model slug + UTC timestamp, bumped one second at a time until no artifact with that id exists."""
    slug = model_slug(model_name)
    now = datetime.now(timezone.utc)
    for k in range(600):
        rid = f"{slug}_{utc_stamp(now + timedelta(seconds=k))}"
        if not any(os.path.exists(os.path.join(log_dir, rid + ext)) for ext in (".log", ".json", ".trace.jsonl")):
            return rid
    raise RuntimeError(f"could not find a free run_id for {model_name} in {log_dir}")


def config_sha256(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_file(path, tag: str | None = None, max_mb: float = 500.0):
    """Copy `path` to <dir>/history/<stem>__<tag><ext> (tag defaults to the current UTC stamp).
    Never overwrites: an existing target gets a numeric suffix. Skips the copy when the newest
    archived copy of the same file is byte-identical. Returns the archive path, or None."""
    src = Path(path)
    if not src.is_file() or src.stat().st_size > max_mb * 1024 * 1024:
        return None
    hist = src.parent / "history"
    hist.mkdir(parents=True, exist_ok=True)
    prior = sorted(hist.glob(f"{src.stem}__*{src.suffix}"), key=lambda p: p.stat().st_mtime)
    if prior and prior[-1].stat().st_size == src.stat().st_size and file_sha256(prior[-1]) == file_sha256(src):
        return str(prior[-1])
    tag = tag or utc_stamp()
    dst = hist / f"{src.stem}__{tag}{src.suffix}"
    n = 1
    while dst.exists():
        dst = hist / f"{src.stem}__{tag}_{n}{src.suffix}"; n += 1
    shutil.copy2(src, dst)
    return str(dst)


def save_json_versioned(path, obj, tag: str | None = None) -> str:
    """Archive the current file (if any), then write the new one atomically. Returns the archive of the OLD file."""
    from .io import save_json
    old = archive_file(path, tag)
    save_json(path, obj)
    return old


def append_ledger(results_dir: str, row: dict) -> str:
    """Append-only copy of every results row: results/history/all_results_ledger.csv.
    Rows are only ever added; the header is widened (atomically) when a new column appears."""
    import csv
    from .io import file_lock, atomic_write_text
    import io as _io
    hist = os.path.join(results_dir, "history"); os.makedirs(hist, exist_ok=True)
    path = os.path.join(hist, "all_results_ledger.csv")
    with file_lock(path):
        rows, fields = [], []
        if os.path.isfile(path):
            with open(path, newline="") as f:
                rd = csv.DictReader(f); fields = list(rd.fieldnames or []); rows = list(rd)
        for k in row:
            if k not in fields:
                fields.append(k)
        buf = _io.StringIO(); w = csv.DictWriter(buf, fieldnames=fields); w.writeheader()
        for r in rows + [row]:
            w.writerow(r)
        atomic_write_text(path, buf.getvalue())
    return path
