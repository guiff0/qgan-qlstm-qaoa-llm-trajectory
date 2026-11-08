import csv
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import experiments as cli  # noqa: E402
from scripts import run_full_study as rfs  # noqa: E402
from scripts import run_repeated_experiment as rre  # noqa: E402
from src.utils import experiment as exp  # noqa: E402
from src.utils.run_identity import latest_per_config_seed  # noqa: E402

A, B = "EXP-20261107-WINBOX-aaaa", "EXP-20261108-LAPTOP-bbbb"
MODELS, SEEDS = ["QGAN-LLM", "Classical GAN-LLM"], [43, 44]


def _write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def row(slug, seed, eid="", ts="20261107T100000Z", utc="2026-11-07T10:00:00+00:00", fid=100.0):
    r = {"run_id": f"{slug}_{ts}", "model_type": "QGAN-LLM" if "qgan" in slug else "Classical GAN-LLM",
         "seed": seed, "run_utc": utc, "fid": fid}
    if eid is not None:
        r["experiment_id"] = eid
    return r


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)
    rre.set_experiment_id(None)
    yield tmp_path
    rre.set_experiment_id(None)
    os.environ.pop("EXPERIMENT_ID", None)


def test_id_format_and_machine_slug():
    eid = exp.new_experiment_id(machine="Win-Box 01!", rand="ab12")
    assert exp.is_valid_id(eid) and eid.endswith("-WINBOX01-ab12")
    assert not exp.is_valid_id("EXP-1") and not exp.is_valid_id("run_1")
    assert exp.new_experiment_id() != exp.new_experiment_id()


def test_current_id_defaults_to_unassigned(repo, monkeypatch):
    assert exp.current_experiment_id() == "unassigned"
    monkeypatch.setenv("EXPERIMENT_ID", A)
    assert exp.current_experiment_id() == A


def test_create_is_idempotent_and_refuses_a_different_plan(repo):
    m1 = exp.create_experiment(A, conditions=MODELS, seeds=SEEDS, base_seed=43, label="a")
    m2 = exp.create_experiment(A, conditions=MODELS, seeds=SEEDS, base_seed=43)
    assert m1["created_utc"] == m2["created_utc"] and m1["plan_hash"] == m2["plan_hash"]
    with pytest.raises(exp.ExperimentError):
        exp.create_experiment(A, conditions=MODELS, seeds=[50, 51], base_seed=50)
    # the same plan under another id has the same plan_hash: recognisable as a replication
    m3 = exp.create_experiment(B, conditions=MODELS, seeds=SEEDS, base_seed=43)
    assert m3["plan_hash"] == m1["plan_hash"] and m3["experiment_id"] != m1["experiment_id"]


def test_runs_log_and_find_run_id(repo):
    _write_csv("results/all_results.csv", [row("qgan_llm", 43, A, ts="20261107T100000Z"),
                                           row("qgan_llm", 43, A, ts="20261107T120000Z"),
                                           row("qgan_llm", 43, B, ts="20261109T120000Z")])
    assert exp.find_run_id("results/all_results.csv", A, "qgan_llm", 43) == "qgan_llm_20261107T120000Z"
    assert exp.find_run_id("results/all_results.csv", B, "qgan_llm", 43) == "qgan_llm_20261109T120000Z"
    assert exp.find_run_id("results/all_results.csv", A, "qgan_llm", 99) is None
    exp.register_run(A, {"slug": "qgan_llm", "seed": 43, "status": "ok", "run_id": "x"})
    assert exp.read_runs(A)[0]["run_id"] == "x"


def test_select_refuses_to_mix_and_pools_only_on_request():
    df = pd.DataFrame([row("qgan_llm", 43, A), row("qgan_llm", 43, B), row("qgan_llm", 44, "")])
    with pytest.raises(exp.ExperimentError, match="3 experiments"):
        exp.select_experiments(df)
    one, ids = exp.select_experiments(df, [A])
    assert ids == [A] and len(one) == 1
    with pytest.raises(exp.ExperimentError, match="pool"):
        exp.select_experiments(df, [A, B])
    both, ids = exp.select_experiments(df, [A, B], pool=True)
    assert len(both) == 2
    with pytest.raises(exp.ExperimentError, match="not found"):
        exp.select_experiments(df, ["EXP-20200101-X-0000"])
    legacy_only = df[df["experiment_id"] == ""]
    assert exp.select_experiments(legacy_only)[1] == ["unassigned"]


def test_seed_overlap_flags_shared_seeds():
    df = pd.DataFrame([row("qgan_llm", 43, A), row("qgan_llm", 43, B), row("qgan_llm", 44, A)])
    assert exp.seed_overlap(df) == {"qgan_llm": [43]}


def test_dedupe_collapses_within_but_not_across_experiments():
    df = pd.DataFrame([row("qgan_llm", 43, A, ts="20261107T100000Z"), row("qgan_llm", 43, A, ts="20261107T110000Z"),
                       row("qgan_llm", 43, B, ts="20261108T100000Z")])
    out = latest_per_config_seed(df)
    assert len(out) == 2 and set(out["experiment_id"]) == {A, B}      # same-experiment re-evaluation collapsed
    assert latest_per_config_seed(df.drop(columns="experiment_id")).shape[0] == 1    # legacy behaviour unchanged


def test_resume_counts_only_this_experiments_rows(repo):
    _write_csv("results/all_results.csv", [row("qgan_llm", 43, A), row("qgan_llm", 44, B), row("qgan_llm", 45, "")])
    assert rre._csv_seeds("QGAN-LLM") == {43, 44, 45}                   # legacy: everything counts
    assert rre._csv_seeds("QGAN-LLM", experiment_id=A) == {43}
    assert rre._csv_seeds("QGAN-LLM", experiment_id="unassigned") == {45}
    rre.set_experiment_id(B)
    assert rre._csv_seeds("QGAN-LLM") == {44}
    assert "experiments" in rre._state_path("tag") and B in rre._state_path("tag")
    assert os.environ["EXPERIMENT_ID"] == B                              # inherited by run_all subprocesses
    rre._save_state("tag", {"completed_seeds": [44], "failed_seeds": []})
    rre.set_experiment_id(A)
    assert rre._load_state("tag")["completed_seeds"] == []               # a second study never inherits "done"


def test_study_remaining_is_per_experiment(repo):
    models = ["QGAN-LLM", "Classical GAN-LLM"]
    _write_csv("results/all_results.csv", [row("qgan_llm", 43, A), row("classical_gan_llm", 43, A)])
    rre.set_experiment_id(A)
    assert rfs.remaining(models, [43]) == []
    rre.set_experiment_id(B)
    assert rfs.remaining(models, [43]) == [("QGAN-LLM", 43), ("Classical GAN-LLM", 43)]


def test_log_dir_is_per_experiment(repo, monkeypatch):
    monkeypatch.setattr(rfs, "EXPERIMENT", A)
    assert A in rfs._log_path("QGAN-LLM", 43)
    monkeypatch.setattr(rfs, "EXPERIMENT", None)
    assert rfs._log_path("QGAN-LLM", 43) == os.path.join("logs", "study", "qgan_llm_seed43.log")


def test_resolve_experiment_rules(repo):
    kw = dict(models=MODELS, seeds=SEEDS, base_seed=43, config=None)
    assert rfs.resolve_experiment(legacy=True, **kw) is None
    # clean folder, no flags: starts a new experiment and makes it active
    first = rfs.resolve_experiment(**kw)
    assert exp.is_valid_id(first) and exp.get_active() == first
    # re-launch with no flags resumes it
    assert rfs.resolve_experiment(**kw) == first
    # a different plan must not silently reuse the active experiment
    with pytest.raises(SystemExit, match="different plan"):
        rfs.resolve_experiment(models=MODELS, seeds=[100, 101], base_seed=100, config=None)
    # explicit new experiment: different id, becomes active
    second = rfs.resolve_experiment(new=True, **kw)
    assert second != first and exp.get_active() == second
    # explicit id of an existing experiment resumes it
    assert rfs.resolve_experiment(experiment_id=first, **kw) == first


def test_resolve_refuses_when_unlabelled_rows_exist(repo):
    _write_csv("results/all_results.csv", [row("qgan_llm", 43, "")])
    with pytest.raises(SystemExit, match="adopt"):
        rfs.resolve_experiment(models=MODELS, seeds=SEEDS, base_seed=43, config=None)
    assert rfs.resolve_experiment(models=MODELS, seeds=SEEDS, base_seed=43, config=None, legacy=True) is None
    assert exp.is_valid_id(rfs.resolve_experiment(models=MODELS, seeds=SEEDS, base_seed=43, config=None, new=True))


def test_adopt_dry_run_then_apply(repo):
    rows = [row("qgan_llm", 43, "", utc="2026-11-05T10:00:00+00:00"),
            row("classical_gan_llm", 43, "", utc="2026-11-05T11:00:00+00:00"),
            row("qgan_llm", 42, "", utc="2026-11-05T12:00:00+00:00"),       # seed outside the plan: left alone
            row("qgan_llm", 43, "", utc="2026-10-01T10:00:00+00:00")]       # before --since: left alone
    _write_csv("results/all_results.csv", rows)
    dry = cli.adopt(A, "2026-11-03", conditions=MODELS, seeds=SEEDS, base_seed=43)
    assert dry["rows_to_stamp"] == 2 and not dry["applied"]
    assert not exp.experiments_in(pd.read_csv("results/all_results.csv")).get(A)
    rep = cli.adopt(A, "2026-11-03", conditions=MODELS, seeds=SEEDS, base_seed=43, apply=True)
    assert rep["applied"] and rep["csv_archive"] and os.path.isfile(rep["csv_archive"])
    counts = exp.experiments_in(pd.read_csv("results/all_results.csv"))
    assert counts == {A: 2, "unassigned": 2}
    assert len(exp.read_runs(A)) == 2 and exp.get_active() == A
    # the launcher now resumes this experiment (same plan) instead of refusing
    assert rfs.resolve_experiment(models=MODELS, seeds=SEEDS, base_seed=43, config=None) == A


def _machine_b(tmp_path, rows, with_manifest=True):
    d = tmp_path / "machine_b" / "results"
    _write_csv(str(d / "all_results.csv"), rows)
    if with_manifest:
        mdir = d / "experiments" / B
        mdir.mkdir(parents=True)
        man = {"experiment_id": B, "plan_hash": exp.plan_hash(MODELS, SEEDS, None), "conditions": MODELS,
               "seeds": SEEDS}
        (mdir / "manifest.json").write_text(json.dumps(man))
        (mdir / "runs.jsonl").write_text(json.dumps({"experiment_id": B, "status": "ok", "run_id": "r1"}) + "\n")
    return str(tmp_path / "machine_b")


def test_import_is_idempotent_and_keeps_experiments_apart(repo):
    _write_csv("results/all_results.csv", [row("qgan_llm", 43, A)])
    src = _machine_b(repo, [row("qgan_llm", 43, B, ts="20261108T100000Z"), row("qgan_llm", 44, B, ts="20261108T110000Z")])
    dry = cli.import_experiment(src)
    assert dry["experiments"][B]["new_rows"] == 2 and not dry["applied"]
    assert len(pd.read_csv("results/all_results.csv")) == 1
    cli.import_experiment(src, apply=True)
    df = pd.read_csv("results/all_results.csv")
    assert exp.experiments_in(df) == {A: 1, B: 2}
    again = cli.import_experiment(src, apply=True)
    assert again["experiments"][B]["new_rows"] == 0 and len(pd.read_csv("results/all_results.csv")) == 3
    assert os.path.isfile(os.path.join("results", "experiments", B, "manifest.json"))
    assert len(exp.read_runs(B)) == 1


def test_import_refuses_unlabelled_rows_and_conflicts(repo):
    _write_csv("results/all_results.csv", [row("qgan_llm", 43, A, ts="20261107T100000Z")])
    src = _machine_b(repo, [row("qgan_llm", 43, "", ts="20261108T100000Z")], with_manifest=False)
    with pytest.raises(cli.ImportConflict, match="no rows with an experiment_id"):
        cli.import_experiment(src)
    # same run_id claimed by another experiment
    src2 = _machine_b(repo, [row("qgan_llm", 43, B, ts="20261107T100000Z")], with_manifest=False)
    with pytest.raises(cli.ImportConflict, match="exists locally under experiment"):
        cli.import_experiment(src2)


def test_verify_reports_missing_and_outside_plan(repo):
    exp.create_experiment(A, conditions=MODELS, seeds=SEEDS, base_seed=43)
    _write_csv("results/all_results.csv", [row("qgan_llm", 43, A), row("qgan_llm", 77, A)])
    rep = cli.verify(A)
    assert rep["done"] == 1 and rep["planned"] == 4 and len(rep["missing"]) == 3
    assert rep["outside_plan"] == ["qgan_llm seed 77"] and not rep["clean"]


def test_hypothesis_tests_refuse_a_mixed_results_file(repo):
    rows = [row("classical_gan_llm", s, e, fid=100 + s) for e in (A, B) for s in (43, 44)]
    _write_csv("results/all_results.csv", rows)
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-m", "scripts.run_hypothesis_tests"], capture_output=True, text=True,
                       cwd=str(repo), env=env)
    assert r.returncode != 0 and "2 experiments" in (r.stdout + r.stderr)
    ok = subprocess.run([sys.executable, "-m", "scripts.run_hypothesis_tests", "--experiment-id", A],
                        capture_output=True, text=True, cwd=str(repo), env=env)
    assert ok.returncode == 0, ok.stderr[-600:]
    out = json.load(open(repo / "results" / "hypothesis_test_results.json"))
    assert out["experiment_id"] == A and out["n_rows_raw"] == 2 and out["n_rows_in_file"] == 4


def test_supports_stamping_detects_the_logging_patch(repo):
    has_patch = "experiment_id" in open(os.path.join(ROOT, "src", "utils", "logging_utils.py"), encoding="utf-8").read()
    assert exp.supports_stamping(ROOT) == has_patch        # True once the run-path update (APPLY.md, phase B) is in place
    os.makedirs("src/utils")
    open("src/utils/logging_utils.py", "w").write("# old version\n")
    assert not exp.supports_stamping(".")
