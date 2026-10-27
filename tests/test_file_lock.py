"""
Tests for src/utils/io.py's file_lock, and for the concurrency fix it
enables in RunLogger.append_to_results_csv (used when run_pipeline.py's
--jobs > 1 runs several `run_all.py --only <model>` processes at once,
each appending a row to the same results/all_results.csv).
"""
from __future__ import annotations

import concurrent.futures
import csv
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import file_lock  # noqa: E402
from src.utils.logging_utils import RunLogger  # noqa: E402


def test_second_acquire_blocks_until_first_releases(tmp_path):
    target = str(tmp_path / "f.txt")
    order = []

    def holder():
        with file_lock(target, timeout=5):
            order.append("first-acquired")
            time.sleep(0.2)
            order.append("first-released")

    def waiter():
        time.sleep(0.05)  # ensure `holder` acquires first
        with file_lock(target, timeout=5):
            order.append("second-acquired")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(holder)
        f2 = ex.submit(waiter)
        f1.result()
        f2.result()

    assert order == ["first-acquired", "first-released", "second-acquired"]


def test_lock_file_removed_after_release(tmp_path):
    target = str(tmp_path / "f.txt")
    with file_lock(target):
        assert os.path.exists(f"{target}.lock")
    assert not os.path.exists(f"{target}.lock")


def test_lock_file_removed_even_on_exception(tmp_path):
    target = str(tmp_path / "f.txt")
    with pytest.raises(ValueError):
        with file_lock(target):
            raise ValueError("boom")
    assert not os.path.exists(f"{target}.lock")


def test_timeout_raises_if_never_released(tmp_path):
    target = str(tmp_path / "f.txt")
    with file_lock(target):  # held and never released within this block
        with pytest.raises(TimeoutError):
            with file_lock(target, timeout=0.2, poll_interval=0.02):
                pass


def test_forced_race_window_loses_rows_without_lock(tmp_path):
    """Reproduces the exact read-all/sleep/rewrite-all shape that
    RunLogger.append_to_results_csv used before the fix, with an
    artificially widened window standing in for real cross-process
    disk-I/O/scheduling variance. Without a lock, concurrent writers lose
    rows; this pins down WHY the lock is necessary, not just that it
    doesn't break anything."""
    csv_path = str(tmp_path / "results.csv")

    def append_unlocked(name):
        rows = []
        if os.path.isfile(csv_path):
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
        time.sleep(0.05)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model_type"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
            w.writerow({"model_type": name})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(append_unlocked, [f"m{i}" for i in range(8)]))

    with open(csv_path, newline="") as f:
        n_rows = len(list(csv.DictReader(f)))
    assert n_rows < 8, "expected the unprotected version to lose at least one row"


def test_forced_race_window_loses_nothing_with_lock(tmp_path):
    """Same shape and same widened window as the test above, but wrapped
    in file_lock -- must not lose any rows."""
    csv_path = str(tmp_path / "results.csv")

    def append_locked(name):
        with file_lock(csv_path):
            rows = []
            if os.path.isfile(csv_path):
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
            time.sleep(0.05)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model_type"])
                w.writeheader()
                for r in rows:
                    w.writerow(r)
                w.writerow({"model_type": name})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(append_locked, [f"m{i}" for i in range(8)]))

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert sorted(r["model_type"] for r in rows) == sorted(f"m{i}" for i in range(8))


def test_append_to_results_csv_concurrent_via_run_logger(tmp_path):
    """End-to-end through the real RunLogger API (not the re-implemented
    shape above), with enough concurrent callers to make a race likely if
    the lock were ever removed."""
    results_dir = str(tmp_path / "results")
    log_dir = str(tmp_path / "logs")
    os.makedirs(results_dir, exist_ok=True)

    def worker(i):
        logger = RunLogger(run_id=f"worker_{i}", log_dir=log_dir, results_dir=results_dir)
        logger.append_to_results_csv(f"Model_{i}", {"rmse": float(i) / 10.0, "asr": float(i)})

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, range(16)))

    import pandas as pd
    df = pd.read_csv(os.path.join(results_dir, "all_results.csv"))
    assert len(df) == 16
    assert sorted(df["model_type"]) == sorted(f"Model_{i}" for i in range(16))


def test_training_lock_blocks_live_owner_and_takes_over_stale(tmp_path):
    import os
    import pytest
    from src.utils.io import training_lock
    ckpt = str(tmp_path / "m.pt")
    lock = ckpt + ".train.lock"
    open(lock, "w").write("1\nx\n")  # pid 1 is alive
    with pytest.raises(RuntimeError):
        with training_lock(ckpt):
            pass
    open(lock, "w").write("99999999\nx\n")  # dead pid -> stale
    with training_lock(ckpt):
        assert os.path.exists(lock)
    assert not os.path.exists(lock)
