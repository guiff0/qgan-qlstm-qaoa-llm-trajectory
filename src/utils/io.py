"""
Atomic file I/O -- ported and adapted from an external repository audit
(one of the genuinely real files found there; most of that repository's
measurement layer was fabricated, but its utils/ layer was solid).

WHY THIS MATTERS HERE SPECIFICALLY: RUNNING.md describes multi-hour-to-
multi-day training runs. A crash, OOM kill, or disconnect partway
through a plain `f.write(json.dumps(...))` can leave a truncated,
corrupt results file on disk -- which is worse than losing the run
entirely, because a partially-written JSON file can look like a
legitimate (if oddly incomplete) result rather than an obvious failure.
atomic_write_text() writes to a temp file in the same directory and
only replaces the target file with a single atomic os.replace() once
the write has fully succeeded, so a result file on disk is always
either the previous complete version or the new complete version --
never a partial one.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Streams the file in chunks rather than reading it fully into
    memory -- relevant here since the raw Dukascopy/ForexSB CSVs this
    project works with run into the hundreds of MB to low GB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic on POSIX and Windows (same filesystem)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_json(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Not atomic (pandas/pyarrow's parquet writer doesn't expose a
    temp-file-then-replace hook the way plain text writes do) -- for
    genuinely crash-safe large-array writes, prefer save_json for
    metadata/results and let the .npy arrays in data/processed/ be
    regenerated from scripts/prepare_data.py if a run is interrupted,
    rather than relying on this function's atomicity."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


class file_lock:
    """Portable cross-PROCESS lock via an exclusively-created sibling
    `.lock` file (works the same on Windows and POSIX, unlike fcntl.flock
    or msvcrt.locking, so it's usable from either the PowerShell/Windows
    environment this project has been run in, or Linux/Mac).

    Used to protect read-modify-write file operations (like
    RunLogger.append_to_results_csv's read-whole-CSV / rewrite-whole-CSV)
    from a lost-update race when multiple processes touch the same file
    concurrently -- e.g. run_pipeline.py's `--jobs N` running several
    `python -m src.experiments.run_all --only <model>` subprocesses at
    once, each appending its own row to the same results/all_results.csv.
    Without this, two processes finishing close together can each read
    the same "existing rows" snapshot, and whichever rewrites the file
    second silently discards the other's row.
    """

    def __init__(self, path: str | Path, timeout: float = 60.0, poll_interval: float = 0.05):
        self.lock_path = f"{path}.lock"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._acquired = False

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                # O_CREAT|O_EXCL is atomic on both Windows and POSIX: the
                # open fails if the file already exists, so exactly one
                # concurrent caller can ever succeed at a time.
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._acquired = True
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Could not acquire lock {self.lock_path} within {self.timeout}s "
                        f"-- if a previous run crashed while holding it, delete this file "
                        f"manually and retry."
                    )
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired and os.path.exists(self.lock_path):
            os.remove(self.lock_path)
        return False


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check that works on POSIX and Windows."""
    if pid <= 0:
        return False
    try:
        import psutil  # optional
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class training_lock:
    """Exclusive, stale-aware lock around TRAINING of one checkpoint.

    Why: on Oct 25 two pipeline processes trained/evaluated the same
    no-noise QGAN-LLM checkpoint concurrently (run ids ...T085629Z and
    ...T124139Z share one config fingerprint and one .pt file), so both
    wrote the same checkpoint and the duplicate evaluation cost ~6 h.
    file_lock only protected the results CSV, not training.

    Lock file `<path>.train.lock` holds "pid\nUTC-timestamp". If it exists
    and its pid is alive, acquisition fails fast with RuntimeError (we do not
    wait hours). If the pid is dead the lock is stale and is taken over.
    """

    def __init__(self, path: str | Path):
        self.lock_path = Path(str(path) + ".train.lock")

    def _read_owner(self):
        try:
            return int(self.lock_path.read_text().split()[0])
        except Exception:
            return -1

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(f"{os.getpid()}\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
                return self
            except FileExistsError:
                owner = self._read_owner()
                if owner != os.getpid() and _pid_alive(owner):
                    raise RuntimeError(
                        f"Another process (pid {owner}) is already training this checkpoint "
                        f"({self.lock_path}). Refusing to run a concurrent duplicate."
                    )
                try:  # stale -> take over
                    os.remove(self.lock_path)
                except FileNotFoundError:
                    pass
        raise RuntimeError(f"Could not acquire {self.lock_path}")

    def __exit__(self, *exc):
        try:
            if self._read_owner() == os.getpid():
                os.remove(self.lock_path)
        except FileNotFoundError:
            pass
        return False
