"""14 Nov 2026: fixes for what the EXP-20261114 console log showed.

1. stages line: a record left by ANOTHER experiment (hypothesis=failed from 10 Nov) was shown for the whole new study, `main`
   stayed pending although all runs finished, and feature_map_advantage.json (the .ps1 launcher's file name) was not found.
2. "ETA n/a (no run finished yet)" printed at 12/12.
3. interim headlines said CONTRADICTED / NOT_SUPPORTED under a table reading SUGGESTIVE_AGAINST / INCONCLUSIVE.
4. the poisoning calibration accepted a strength on the MEDIAN pilot seed although check M8 fails on ANY run under the floor.
5. "-0.0000" in the walk-forward table.
"""
import json
import os

import numpy as np

from scripts import study_dashboard as sd
from src.evaluation import poisoning_protocol as pp
from src.evaluation import verdicts as vd
from src.evaluation.arm_ledger import _sk


def _state(tmp_path, state):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "study_driver_state.json").write_text(json.dumps(state), encoding="utf8")


def test_stage_record_of_another_experiment_is_not_shown_for_this_one(tmp_path):
    _state(tmp_path, {"calibrate": {"status": "ok", "experiment_id": None},
                      "hypothesis": {"status": "failed", "experiment_id": "EXP-20261109-OLD-5b6f"},
                      "main": {"status": "ok", "experiment_id": "EXP-20261114-NEW-8564"}})
    line = sd.pipeline_lines(str(tmp_path), experiment_id="EXP-20261114-NEW-8564")[0]
    assert "calibrate=ok" in line and "main=ok" in line and "hypothesis=pending" in line and "failed" not in line
    # without an experiment id (legacy) the record is shown as before
    assert "hypothesis=failed" in sd.pipeline_lines(str(tmp_path))[0]


def test_unsuffixed_feature_map_file_is_found_and_marks_the_stage(tmp_path):
    _state(tmp_path, {})
    (tmp_path / "results" / "feature_map_advantage.json").write_text(
        json.dumps({"per_run": [{"seed": s, "arm": a} for s in range(10) for a in ("linear", "classical_rf", "quantum")]}))
    (tmp_path / "results" / "feature_map_advantage__20261110T122733Z.json").write_text("{}")     # history copy: ignored
    lines = sd.pipeline_lines(str(tmp_path), experiment_id="EXP-X")
    assert "feature_map=ok (file on disk)" in lines[0]
    assert any("10 seed(s) x 3 arm(s) = 30 rows" in l for l in lines) and not any("not run yet (stage feature_map" in l for l in lines)
    assert len(sd.feature_map_files(str(tmp_path))) == 1


def test_eta_says_done_when_nothing_remains():
    dash = sd.StudyDashboard(["A"], [1, 2], lambda: [], lambda m, s: "x.log", extra_lines_fn=lambda: [])
    text = "\n".join(dash.summary_lines())
    assert "all runs done" in text and "no run finished yet" not in text


def test_main_and_hypothesis_stages_are_recorded_by_run_full_study(tmp_path, monkeypatch):
    from scripts import run_full_study as rfs
    monkeypatch.setattr(rfs, "EXPERIMENT", "EXP-20261114-NEW-8564")
    path = str(tmp_path / "results" / "study_driver_state.json")
    rfs.mark_stage("main", "running", path=path)
    rfs.mark_stage("main", "ok", 0, path=path)
    rfs.mark_stage("hypothesis", "ok", 0, commands=["python -m scripts.run_hypothesis_tests"], path=path)
    st = json.load(open(path, encoding="utf8"))
    assert st["main"]["status"] == "ok" and st["hypothesis"]["experiment_id"] == "EXP-20261114-NEW-8564"
    line = sd.pipeline_lines(str(tmp_path), experiment_id="EXP-20261114-NEW-8564")[0]
    assert "main=ok" in line and "hypothesis=ok" in line


def test_interim_headline_uses_the_downgraded_verdict_word():
    status = {"complete": False, "label": "INTERIM: 12 of 40 planned runs", "seeds_per_condition": 10}
    for old, new in (("CONTRADICTED", "SUGGESTIVE_AGAINST"), ("NOT_SUPPORTED", "INCONCLUSIVE"), ("SUPPORTED", "SUGGESTIVE_FOR")):
        d = {"verdict": old, "tier": "CONFIRMATORY", "rule": "R", "lean": None, "blockers": [],
             "headline": f"H1 (QGAN FID lower): {old} (confirmatory) -- observed QGAN-LLM > Classical GAN-LLM, p=0.038."}
        vd.apply_interim_gate(d, status)
        assert d["verdict"] == new and d["headline"].startswith(f"H1 (QGAN FID lower): {new} (exploratory, interim) -- ")
        assert f": {old} " not in d["headline"]


def _unstable_data(n=4000, k=6, seed=0):
    """A control head whose clean error is dominated by a feature drift between the training and the evaluation rows, so the
    poison's effect changes SIGN with the attack direction (what the 14 Nov positive control did: -15.8 % .. +18.6 %)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=k)
    Xt = rng.normal(size=(n, k))
    yt = Xt @ w * 0.05 + rng.normal(size=n)
    Xe = rng.normal(size=(n, k)) + 6.0            # evaluation rows far outside the training range
    ye = rng.normal(size=n) + 2.0                 # and the head's extrapolation there is wrong
    return Xt, yt, Xe, ye


def test_calibration_refuses_a_strength_whose_sign_depends_on_the_seed():
    Xt, yt, Xe, ye = _unstable_data()
    seeds = list(range(100, 110))
    old = pp.calibrate_shift(Xt, yt, Xe, ye, seeds=seeds, grid=(0.05, 0.1, 0.2), target_pct=0.0, floor_pct=None)
    row = {r["shift_sd"]: r for r in old["curve"]}
    assert any(r["min_degradation_pct"] < 0 < max(r["per_seed"]) for r in row.values())      # the instability is real here
    new = pp.calibrate_shift(Xt, yt, Xe, ye, seeds=seeds, grid=(0.05, 0.1, 0.2), target_pct=0.0, floor_pct=5.0)
    assert new["chosen_shift_sd"] is None and "not decidable" in new["note"]
    assert "UNSTABLE" in new["note"] or "no shift in the grid" in new["note"]


def test_calibration_choice_holds_for_every_pilot_seed_and_every_larger_shift():
    rng = np.random.default_rng(1)
    Xt, Xe = rng.normal(size=(3000, 5)), rng.normal(size=(3000, 5))
    w = rng.normal(size=5)
    yt, ye = Xt @ w + 0.1 * rng.normal(size=3000), Xe @ w + 0.1 * rng.normal(size=3000)
    out = pp.calibrate_shift(Xt, yt, Xe, ye, seeds=[100, 101, 102, 103, 104], target_pct=10.0, floor_pct=5.0)
    c = out["chosen_shift_sd"]
    assert c is not None and out["floor_pct"] == 5.0
    for r in out["curve"]:
        if r["shift_sd"] >= c:
            assert r["meets_criterion"] and r["min_degradation_pct"] >= 5.0 and r["median_degradation_pct"] >= 10.0


def test_walk_forward_skill_has_no_negative_zero():
    assert _sk(-0.00001).strip() == "0.0000" and _sk(0.00002).strip() == "0.0000"
    assert _sk(0.0003) == "+0.0003" and _sk(-0.0002) == "-0.0002" and _sk(None).strip() == "n/a"


# ---- 14 Nov, 15:34: the study driver died at run 31 of 40 on a half-read progress file ------------------------------------

def test_progress_file_that_is_half_written_never_raises(tmp_path, monkeypatch, capsys):
    from scripts import run_repeated_experiment as rre
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rre, "EXPERIMENT_ID", "EXP-T")
    monkeypatch.setattr(rre, "_STATE_WARNED", set())
    rre._save_state("QGAN-LLM", {"completed_seeds": [53, 54], "failed_seeds": []})
    assert rre._load_state("QGAN-LLM")["completed_seeds"] == [53, 54]
    path = rre._state_path("QGAN-LLM")
    assert not [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp")]          # no temp file left behind
    with open(path, "w") as fh:
        fh.write("{\n")                                                                       # what the dashboard read
    st = rre._load_state("QGAN-LLM", retries=2, wait=0.0)
    assert st == {"completed_seeds": [], "failed_seeds": []} and "could not be read" in capsys.readouterr().out
    rre._load_state("QGAN-LLM", retries=1, wait=0.0)
    assert capsys.readouterr().out == ""                                                      # warned once, not every second
    with open(path, "w") as fh:
        json.dump({"completed_seeds": [58]}, fh)                                              # older file without failed_seeds
    assert rre._load_state("QGAN-LLM") == {"completed_seeds": [58], "failed_seeds": []}


def test_a_dashboard_error_does_not_stop_the_run_bookkeeping(tmp_path, monkeypatch, capsys):
    from scripts import run_full_study as rfs
    import threading

    class Boom:
        def run_started(self, *a): raise ValueError("half-read file")
        def run_finished(self, *a): raise ValueError("half-read file")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rfs, "DASH", Boom())
    monkeypatch.setattr(rfs, "EXPERIMENT", None)
    monkeypatch.setattr(rfs, "_study_dir", lambda: str(tmp_path / "logs" / "study"))
    monkeypatch.setattr(rfs, "_log_path", lambda m, s: str(tmp_path / "logs" / "study" / "x.log"))
    monkeypatch.setattr(rfs.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(rfs.rre, "_csv_seeds", lambda model, *a, **k: {58})
    assert rfs._run_one("QGAN-LLM", 58, None, {}, threading.Lock()) is True
    out = capsys.readouterr().out
    assert "display error after 'QGAN-LLM' seed 58 (ignored; the run itself was recorded" in out


# ---- 15 Nov: H2/H3 headline after the Holm correction; M8 positive control on the whole training half --------------------

def test_headline_follows_the_holm_downgrade():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "evaluation", "verdicts.py"),
               encoding="utf8").read()
    assert "not significant after correction" in src
    # the rewrite itself, on the two lines the 15 Nov output printed
    for old, lean, adj, line in (("SUPPORTED", "for", 0.126, "H2 (QGAN RMSE lower than classical): SUPPORTED (confirmatory) -- observed QGAN-LLM < Classical GAN-LLM, p=0.042."),
                                 ("CONTRADICTED", "against", 0.085, "H3 (QGAN ASR lower than classical): CONTRADICTED (confirmatory) -- observed QGAN-LLM > Classical GAN-LLM, p=0.021.")):
        word = f"INCONCLUSIVE (leans {lean})"
        hl = line.replace(f": {old} (", f": {word} (", 1).rstrip()
        hl = (hl[:-1] if hl.endswith(".") else hl) + f"; Holm-adjusted p={adj:.3f}, not significant after correction."
        assert old not in hl and word in hl and hl.endswith("not significant after correction.") and f"p={adj:.3f}" in hl


def test_positive_control_rows_number_is_unchanged_and_all_means_the_whole_half():
    assert pp.pc_rows(100_000, 372_393) == 100_000 and pp.pc_rows(100_000, 500) == 500
    for v in (None, "all", "ALL", "null", 0, -1):
        assert pp.pc_rows(v, 372_393) == 372_393
    assert pp.chronological_split(1000, 100) == {"train": slice(0, 500), "eval": slice(501, 1000), "pc": slice(0, 100)}
    assert pp.chronological_split(1000, None)["pc"] == slice(0, 500)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)); y = X @ rng.normal(size=4) + 0.1 * rng.normal(size=600)
    a = pp.run_protocol(X[:300], y[:300], X[300:], y[300:], X_real_train=X[:300], seed=3, cfg={"pc_max_rows": 100, "shift_sd": 1.0})
    b = pp.run_protocol(X[:300], y[:300], X[300:], y[300:], X_real_train=X[:300], seed=3, cfg={"pc_max_rows": None, "shift_sd": 1.0})
    assert a["poisoning_pc_rows"] == 100 and b["poisoning_pc_rows"] == 300
    assert a["poisoning_rmse_degradation_pct"] == b["poisoning_rmse_degradation_pct"]          # the MAIN head is untouched
    assert a["poisoning_pc_degradation_pct"] != b["poisoning_pc_degradation_pct"]


def test_a_control_head_that_has_already_failed_flips_sign_and_the_whole_half_does_not():
    """The M8 failure in miniature: the oldest rows sit at a different price level from the scored rows."""
    rng = np.random.default_rng(1)
    n = 6000
    level = np.r_[np.full(1000, 1.085), np.linspace(1.085, 1.13, 2000), 1.13 + 0.02 * np.sin(np.arange(3000) / 300)]
    X = np.c_[level + 0.0005 * rng.normal(size=n), rng.normal(size=(n, 5))]
    X[:1000, 0] = rng.normal(size=1000) * 0.0005 + 1.085                 # no level variation in the oldest rows: nothing to learn from
    y = level + 0.0002 * rng.normal(size=n)
    sp = pp.chronological_split(n, 1000)
    Xe, ye = X[sp["eval"]], y[sp["eval"]]
    old = [pp.degradation(X[:1000], y[:1000], Xe, ye, fraction=0.2, shift_sd=1.0, seed=s)["degradation_pct"] for s in range(12)]
    new = [pp.degradation(X[sp["train"]], y[sp["train"]], Xe, ye, fraction=0.2, shift_sd=1.0, seed=s)["degradation_pct"] for s in range(12)]
    assert min(old) < 5.0                                                 # under the M8 floor for at least one seed
    assert min(new) > 5.0                                                 # every seed degrades once the clean head is sound


def test_set_positive_control_edits_only_the_poisoning_block():
    from scripts import set_positive_control as spc
    text = ("other:\r\n  pc_max_rows: 7\r\n  calibrated: true\r\npoisoning:\r\n  shift_sd: 1.0   # keep\r\n  calibrated: true        # note\r\n"
            "  pc_max_rows: 100000\r\n  pilot_seeds: [100, 101]\r\nnext:\r\n  calibrated: true\r\n")
    new, changes = spc.rewrite(text, "all")
    assert changes == ["poisoning.pc_max_rows: 100000 -> null", "poisoning.calibrated: true -> false"]
    assert "poisoning:\r\n  shift_sd: 1.0   # keep\r\n  calibrated: false        # note\r\n  pc_max_rows: null\r\n" in new
    assert new.startswith("other:\r\n  pc_max_rows: 7\r\n  calibrated: true\r\n") and new.endswith("next:\r\n  calibrated: true\r\n")
    assert spc.rewrite(new, "all") == (new, [])                           # idempotent
    import yaml
    assert yaml.safe_load(new)["poisoning"] == {"shift_sd": 1.0, "calibrated": False, "pc_max_rows": None, "pilot_seeds": [100, 101]}
