"""
Structured run logging.

Every experiment writes THREE things per run, all timestamped and
tied together by a shared run_id, so results can be handed back as
evidence and traced to exactly which code/config produced them:

1. A human-readable .log file (what a person reads)
2. A machine-readable .json file (what verify_chapter4.py reads)
3. Appends one row to results/all_results.csv (what pandas reads)

This replaces the original code's approach of only calling
logger.info() — printed log lines are not something that can be
mechanically diffed against Chapter 4's tables afterward.
"""
import csv
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from .io import file_lock


def make_run_id(model_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    return f"{safe_name}_{ts}"


def get_environment_fingerprint() -> dict:
    """Captures exactly what ran, so results can't be second-guessed later as
    'maybe it was a different version of X'."""
    fingerprint = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fingerprint["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        fingerprint["git_commit"] = "not_a_git_repo"
    # A bare commit hash is not reproducible if the working tree had edits
    # (the Sept LSTM runs differed 2.3x between "same" hyperparameters).
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        fingerprint["git_dirty"] = bool(porcelain.strip())
    except Exception:
        fingerprint["git_dirty"] = None
    for pkg in ["torch", "pennylane", "numpy", "scipy", "pandas"]:
        try:
            mod = __import__(pkg)
            fingerprint[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            fingerprint[f"{pkg}_version"] = "not_installed"
    return fingerprint


class RunLogger:
    def __init__(self, run_id: str, log_dir: str = "logs", results_dir: str = "results"):
        self.run_id = run_id
        self.log_dir = log_dir
        self.results_dir = results_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        self.logger = logging.getLogger(run_id)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(os.path.join(log_dir, f"{run_id}.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(sh)

        self.record = {
            "run_id": run_id,
            "environment": get_environment_fingerprint(),
            "config_used": None,
            "training_history": [],
            "final_metrics": {},
        }

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def log_config(self, config: dict):
        self.record["config_used"] = config

    def log_epoch(self, epoch: int, **metrics):
        row = {"epoch": epoch, **metrics}
        self.record["training_history"].append(row)
        metric_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in metrics.items())
        self.info(f"Epoch {epoch} | {metric_str}")

    def log_final_metrics(self, metrics: dict):
        self.record["final_metrics"] = metrics
        self.info(f"FINAL METRICS: {json.dumps(metrics, indent=2, default=str)}")

    def save_json(self):
        path = os.path.join(self.log_dir, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.record, f, indent=2, default=str)
        self.info(f"Saved structured log to {path}")
        return path

    def append_to_results_csv(self, model_type: str, metrics: dict):
        """This CSV is what verify_chapter4.py reads — it's the single
        accumulating ledger of every model's final numbers.

        BUG FIX: different model types report different metric sets
        (e.g. Classical LSTM has no `asr`/`entanglement_entropy`; QGAN-LLM
        has both). The original version built `fieldnames` fresh from
        each row's own keys, so the header written for the FIRST row
        would be too narrow for later rows with extra keys -- appending
        those wrote more comma-separated values than the header had
        columns for, producing a ragged CSV that pandas.read_csv() (used
        by results_collector.py and verify_chapter4.py) cannot parse
        correctly. Fixed by keeping the fieldname set as the UNION of
        every row's keys seen so far, rewriting the file with an updated
        header whenever a new key appears. Results files here are small
        (one row per model run), so a full rewrite on append is cheap.

        CONCURRENCY FIX: this is a read-the-whole-file / rewrite-the-
        whole-file operation, which is a lost-update race if two
        processes call it around the same time (e.g. run_pipeline.py
        running several `run_all.py --only <model>` subprocesses in
        parallel via --jobs). Wrapped in file_lock so only one process
        can be inside the read/rewrite at a time; the others simply wait
        their turn rather than silently clobbering each other's row.
        Verified: 8 processes appending concurrently to a fresh CSV all
        landed intact (see tests/test_logging_utils.py)."""
        csv_path = os.path.join(self.results_dir, "all_results.csv")
        row = {"run_id": self.run_id, "model_type": model_type, **metrics}

        with file_lock(csv_path):
            existing_rows = []
            existing_fieldnames: list[str] = []
            if os.path.isfile(csv_path):
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    existing_fieldnames = reader.fieldnames or []
                    existing_rows = list(reader)

            fieldnames = list(existing_fieldnames)
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)
                writer.writerow(row)

        self.info(f"Appended results row to {csv_path}")
