"""Orchestration of scripts.run_full_study: passes until complete, retries, stall detection, resumability.
No training: the pass runner is replaced by a fake that appends rows to results/all_results.csv."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.getcwd())
from scripts import run_full_study as fs
from scripts import run_repeated_experiment as rre


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


def _append(model, seed):
    new = not os.path.isfile(rre.RESULTS_CSV)
    with open(rre.RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "model_type", "seed"])
        w.writerow([f"{rre._model_slug(model)}_20261101T000000Z", "QGAN-LLM", seed])


MODELS = ["A", "B (x)"]


def test_conditions_default_and_all():
    names = fs.build_conditions(fs.DEFAULT_LEVELS)
    assert names[:3] == fs.PRIMARY and len(names) == 3 + 4
    assert "QGAN-LLM (ent-L0 none-d4)" in names
    assert len(fs.build_conditions("all")) == 3 + 7
    with pytest.raises(SystemExit):
        fs.build_conditions((9,))


def test_runs_passes_until_everything_is_done_and_retries_failures(cwd):
    flaky = {("B (x)", 44): 2}                  # fails twice, then succeeds

    def fake_pass(models, n_runs, base_seed, config):
        for s in range(base_seed, base_seed + n_runs):
            for m in models:
                if s in rre._csv_seeds(m):
                    continue
                if flaky.get((m, s), 0) > 0:
                    flaky[(m, s)] -= 1
                    continue
                _append(m, s)
        return 0

    sleeps = []
    rc = fs.run_study(MODELS, 3, 43, None, max_stalled=5, wait_s=1, sleep=sleeps.append, run_pass=fake_pass)
    assert rc == 0 and fs.remaining(MODELS, [43, 44, 45]) == []
    assert len(sleeps) == 2                      # two retry waits, then done
    import json
    assert json.load(open(fs.STATUS_FILE))["state"] == "complete"


def test_gives_up_only_after_the_stall_limit_and_names_the_missing_runs(cwd, capsys):
    calls = []

    def always_fails(models, n_runs, base_seed, config):
        calls.append(1)
        return 1

    rc = fs.run_study(MODELS, 2, 43, None, max_stalled=3, wait_s=1, sleep=lambda s: None, run_pass=always_fails)
    assert rc == 1 and len(calls) == 3
    assert "STOPPED" in capsys.readouterr().out
    import json
    st = json.load(open(fs.STATUS_FILE))
    assert st["state"] == "stalled" and st["completed"] == 0 and len(st["remaining"]) == 4


def test_unlimited_mode_keeps_going_until_it_succeeds(cwd):
    n = {"c": 0}

    def eventually(models, n_runs, base_seed, config):
        n["c"] += 1
        if n["c"] >= 8:
            for s in range(base_seed, base_seed + n_runs):
                for m in models:
                    if s not in rre._csv_seeds(m):
                        _append(m, s)
        return 0

    assert fs.run_study(MODELS, 2, 43, None, max_stalled=0, wait_s=1, sleep=lambda s: None, run_pass=eventually) == 0
    assert n["c"] == 8


def test_relaunch_after_interruption_does_not_repeat_finished_runs(cwd):
    _append("A", 43)
    _append("B (x)", 43)                        # first seed finished before the 'crash'
    seen = []

    def fake_pass(models, n_runs, base_seed, config):
        for s in range(base_seed, base_seed + n_runs):
            for m in models:
                if s not in rre._csv_seeds(m):
                    seen.append((m, s))
                    _append(m, s)
        return 0

    assert fs.run_study(MODELS, 2, 43, None, 5, 1, sleep=lambda s: None, run_pass=fake_pass) == 0
    assert seen == [("A", 44), ("B (x)", 44)]


def test_partial_progress_resets_the_stall_counter(cwd):
    order = iter([True, False, False, True, True, True, True])

    def slow(models, n_runs, base_seed, config):
        if next(order, True):                    # one more run completes on 'True' passes
            for s in range(base_seed, base_seed + n_runs):
                for m in models:
                    if s not in rre._csv_seeds(m):
                        _append(m, s)
                        return 0
        return 1

    # 4 runs needed; 2 consecutive no-progress passes are tolerated because limit is 3
    assert fs.run_study(MODELS, 2, 43, None, max_stalled=3, wait_s=1, sleep=lambda s: None, run_pass=slow) == 0


def test_head_only_forecast_path_is_detected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("qgan_llm:\n  forecast_path: head_only\nclassical_gan_llm:\n  forecast_path: generator\n")
    assert fs.check_forecast_path(str(p)) == ["qgan_llm.forecast_path = 'head_only'"]
    p.write_text("qgan_llm:\n  forecast_path: generator\nclassical_gan_llm:\n  forecast_path: generator\n")
    assert fs.check_forecast_path(str(p)) == []


def test_repo_config_passes_the_forecast_path_check():
    assert fs.check_forecast_path("config/default_config.yaml") == []
