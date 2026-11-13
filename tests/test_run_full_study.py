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
    assert names == fs.PRIMARY and len(names) == 4        # Amendment 3: no entanglement sweep in the default plan
    assert "Classical GAN-LLM (matched)" in fs.PRIMARY
    assert not any("ent-L" in n for n in names)
    assert "QGAN-LLM (ent-L0 none-d4)" in fs.build_conditions((0,))       # the sweep stays available as an opt-in
    assert len(fs.build_conditions("all")) == len(fs.PRIMARY) + 6         # six levels after the duplicate L4 was removed
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


# ------------------------------------------------------------------ --workers
def test_worker_env_splits_logical_cpus():
    assert fs.worker_env(2, 8)["OMP_NUM_THREADS"] == "4"
    assert fs.worker_env(4, 8)["MKL_NUM_THREADS"] == "2"
    assert fs.worker_env(3, 8)["OPENBLAS_NUM_THREADS"] == "2"
    assert fs.worker_env(16, 8)["OMP_NUM_THREADS"] == "1"          # never 0
    assert fs.worker_env(1, 8)["NUMEXPR_NUM_THREADS"] == "8"


def test_parallel_pass_runs_each_missing_pair_once_with_bounded_concurrency(cwd, monkeypatch):
    import threading
    import time as _t
    active, peak, ran = {"n": 0}, {"n": 0}, []
    lock = threading.Lock()

    class _P:
        def __init__(self, rc): self.returncode = rc

    def fake_subprocess_run(cmd, stdout=None, stderr=None, env=None):
        model, seed = cmd[cmd.index("--only") + 1], int(cmd[cmd.index("--seed") + 1])
        with lock:
            active["n"] += 1
            peak["n"] = max(peak["n"], active["n"])
            ran.append((model, seed, env["OMP_NUM_THREADS"]))
        _t.sleep(0.05)
        _append(model, seed)
        with lock:
            active["n"] -= 1
        return _P(0)

    monkeypatch.setattr(fs.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(fs.os, "cpu_count", lambda: 8)
    models = ["A", "B (x)", "C", "D"]
    _append("A", 43)                                               # already done: must not run again
    rc = fs._run_pass_parallel(models, 2, 43, None, workers=2)
    assert rc == 0 and fs.remaining(models, [43, 44]) == []
    assert sorted((m, s) for m, s, _ in ran) == sorted((m, s) for s in (43, 44) for m in models if (m, s) != ("A", 43))
    assert peak["n"] == 2 and {t for *_, t in ran} == {"4"}
    assert os.path.isfile(os.path.join("logs", "study", "b_x_seed43.log"))
    # state files updated
    assert 44 in rre._load_state(rre._default_tag("C"))["completed_seeds"]


def test_parallel_pass_reports_failure_and_does_not_mark_it_done(cwd, monkeypatch):
    class _P:
        def __init__(self, rc): self.returncode = rc

    def fake(cmd, stdout=None, stderr=None, env=None):
        model, seed = cmd[cmd.index("--only") + 1], int(cmd[cmd.index("--seed") + 1])
        if model == "B (x)":
            return _P(1)
        _append(model, seed)
        return _P(0)

    monkeypatch.setattr(fs.subprocess, "run", fake)
    assert fs._run_pass_parallel(["A", "B (x)"], 1, 43, None, workers=2) == 1
    assert fs.remaining(["A", "B (x)"], [43]) == [("B (x)", 43)]
    assert 43 in rre._load_state(rre._default_tag("B (x)"))["failed_seeds"]


def test_parallel_pass_never_runs_the_same_condition_twice_at_once(cwd, monkeypatch):
    import threading
    import time as _t
    running, violations, peak = set(), [], {"n": 0}
    lock = threading.Lock()

    class _P:
        def __init__(self, rc): self.returncode = rc

    def fake(cmd, stdout=None, stderr=None, env=None):
        model, seed = cmd[cmd.index("--only") + 1], int(cmd[cmd.index("--seed") + 1])
        with lock:
            if model in running:
                violations.append(model)
            running.add(model)
            peak["n"] = max(peak["n"], len(running))
        _t.sleep(0.03)
        _append(model, seed)
        with lock:
            running.discard(model)
        return _P(0)

    monkeypatch.setattr(fs.subprocess, "run", fake)
    assert fs._run_pass_parallel(["A", "B (x)", "C"], 4, 43, None, workers=3) == 0
    assert violations == [] and peak["n"] == 3
    assert fs.remaining(["A", "B (x)", "C"], [43, 44, 45, 46]) == []


def test_tail_of_one_condition_runs_sequentially_not_in_parallel(cwd, monkeypatch):
    import threading
    import time as _t
    active, peak, lock = {"n": 0}, {"n": 0}, threading.Lock()

    class _P:
        returncode = 0

    def fake(cmd, stdout=None, stderr=None, env=None):
        model, seed = cmd[cmd.index("--only") + 1], int(cmd[cmd.index("--seed") + 1])
        with lock:
            active["n"] += 1
            peak["n"] = max(peak["n"], active["n"])
        _t.sleep(0.02)
        _append(model, seed)
        with lock:
            active["n"] -= 1
        return _P()

    monkeypatch.setattr(fs.subprocess, "run", fake)
    assert fs._run_pass_parallel(["A"], 4, 43, None, workers=4) == 0
    assert peak["n"] == 1
