"""Answering 'n' (or --no-skip / --reset) must mean: start from scratch, resume nothing."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_pipeline as rp  # noqa: E402


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rp, "STATE_PATH", os.path.join("data", "pipeline_state.json"))
    monkeypatch.setattr(rp, "_fresh_requested", False)
    monkeypatch.delenv("PIPELINE_FRESH", raising=False)
    monkeypatch.delenv("PIPELINE_RUN_ID", raising=False)


def _steps():
    return [{"id": "train_eval:A", "artifact_ok": lambda: True, "cmd": []},
            {"id": "train_eval:B", "artifact_ok": lambda: True, "cmd": []}]


def _state():
    return {"steps": {"train_eval:A": {"status": "completed", "duration_seconds": 61860.0, "completed_utc": "x"},
                      "train_eval:B": {"status": "running", "started_utc": "2026-10-28T00:00:00Z"}}}


def _args(**kw):
    d = dict(reset=False, yes=False, no_skip=False, rerun_failed=False, keep_llm_cache=False)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_answer_n_requests_fresh(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert rp.decide_skip_policy(_state(), _steps(), _args()) is False
    assert rp._fresh_requested is True


def test_answer_y_resumes(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert rp.decide_skip_policy(_state(), _steps(), _args()) is True
    assert rp._fresh_requested is False


def test_no_skip_flag_requests_fresh():
    rp.decide_skip_policy(_state(), _steps(), _args(no_skip=True))
    assert rp._fresh_requested is True


def test_begin_fresh_run_resets_state_and_cache(monkeypatch):
    os.makedirs("data"); os.makedirs("results/llm_cache")
    open("results/llm_cache/k.json", "w").write("{}")
    st = _state()
    json.dump(st, open(rp.STATE_PATH, "w"))
    rp.begin_fresh_run(st, _steps())
    assert os.environ["PIPELINE_FRESH"] == "1"
    assert all(st["steps"][s["id"]] == {"status": "pending"} for s in _steps())   # no stale DONE / RUN
    assert not os.path.isdir("results/llm_cache")
    moved = [d for d in os.listdir("results/history") if d.startswith("llm_cache__")]
    assert moved and os.path.isfile(os.path.join("results/history", moved[0], "k.json"))


def test_keep_llm_cache(monkeypatch):
    os.makedirs("data"); os.makedirs("results/llm_cache")
    open("results/llm_cache/k.json", "w").write("{}")
    st = _state(); json.dump(st, open(rp.STATE_PATH, "w"))
    rp.begin_fresh_run(st, _steps(), keep_llm_cache=True)
    assert os.path.isfile("results/llm_cache/k.json")


def test_stale_running_becomes_interrupted():
    st = _state()
    assert rp._mark_stale_running(st) == ["train_eval:B"]
    assert st["steps"]["train_eval:B"]["status"] == "interrupted"
    assert st["steps"]["train_eval:A"]["status"] == "completed"


def test_load_checkpoint_ignored_in_fresh_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.utils.gan_training import load_checkpoint, save_checkpoint
    p = str(tmp_path / "m.ckpt")
    save_checkpoint(p, {"fingerprint": "abc", "epoch": 7})
    assert load_checkpoint(p, "abc")["epoch"] == 7          # normal run: resumes
    monkeypatch.setenv("PIPELINE_FRESH", "1")
    assert load_checkpoint(p, "abc") is None                # fresh run: starts from scratch
