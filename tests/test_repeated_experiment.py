"""
Tests the harness's own orchestration logic (resumability, state
tracking) WITHOUT running real training -- this sandbox has no
data/processed/*.npy (the user's real market data lives on their own
machine), so subprocess.run is monkeypatched to a trivial stand-in
rather than the real `python -m src.experiments.run_all`.
"""
import json
import os
import subprocess
import sys

import pytest

import scripts.run_repeated_experiment as rre


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("logs", exist_ok=True)
    yield tmp_path


def test_state_round_trips(isolated_logs):
    state = {"completed_seeds": [1, 2, 3], "failed_seeds": [4]}
    rre._save_state("mytag", state)
    loaded = rre._load_state("mytag")
    assert loaded == state


def test_missing_state_file_returns_empty_defaults(isolated_logs):
    state = rre._load_state("never_seen_tag")
    assert state == {"completed_seeds": [], "failed_seeds": []}


def test_all_seeds_run_and_recorded_on_success(isolated_logs, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_repeated_experiment.py", "--model", "Classical LSTM", "--n-runs", "3", "--tag", "t1",
    ])
    rre.main()

    state = rre._load_state("t1")
    assert state["completed_seeds"] == [0, 1, 2]
    assert state["failed_seeds"] == []
    assert len(calls) == 3
    assert "--seed" in calls[0] and "0" in calls[0]


def test_resumability_skips_already_completed_seeds(isolated_logs, monkeypatch):
    rre._save_state("t2", {"completed_seeds": [0, 1], "failed_seeds": []})
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_repeated_experiment.py", "--model", "Classical LSTM", "--n-runs", "3", "--tag", "t2",
    ])
    rre.main()

    # Only seed 2 should have actually been run -- 0 and 1 were already done.
    assert len(calls) == 1
    assert "2" in calls[0]
    state = rre._load_state("t2")
    assert sorted(state["completed_seeds"]) == [0, 1, 2]


def test_failed_seed_is_recorded_and_not_marked_complete(isolated_logs, monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_repeated_experiment.py", "--model", "Classical LSTM", "--n-runs", "1", "--tag", "t3",
    ])
    rre.main()

    state = rre._load_state("t3")
    assert state["completed_seeds"] == []
    assert state["failed_seeds"] == [0]
