import json, os, re, sys, types
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_study as rs

CFG = ("data:\r\n  x: 1\r\n\r\npoisoning:\r\n  protocol: \"v2_triggered_shift\"\r\n  fraction: 0.2\r\n  shift_sd: 3.0\r\n"
       "  calibrated: false\r\n  ridge: 1.0\r\n\r\nstatistics:\r\n  alpha: 0.05\r\n  calibrated: false\r\n")


def make(tmp_path, argv=(), fail=(), cfg=CFG, cal=None):
    cfg_path = tmp_path / "cfg.yaml"; cfg_path.write_bytes(cfg.encode())
    res = tmp_path / "results"; res.mkdir(exist_ok=True)
    if cal is not None:
        (res / "poisoning_calibration_20261110T000000Z.json").write_text(json.dumps(cal))
    calls = []

    def runner(cmd, cwd=None):
        calls.append(cmd)
        joined = " ".join(cmd)
        return types.SimpleNamespace(returncode=1 if any(f in joined for f in fail) else 0)

    a = rs.parse(list(argv))
    d = rs.Driver(a, runner=runner, state_path=str(tmp_path / "state.json"), config_path=str(cfg_path), results_dir=str(res))
    d.experiment_id = a.experiment_id or "EXP-20261110-TEST-abcd"
    return d, calls, cfg_path


def mods(calls):
    return [next((c[i + 1] for i, x in enumerate(c) if x == "-m"), "") for c in calls]


def test_config_helpers_touch_only_the_poisoning_block_and_keep_crlf():
    s = rs.poisoning_section(CFG)
    assert s["protocol"] == "v2_triggered_shift" and s["calibrated"] is False and s["shift_sd"] == 3.0
    assert rs.needs_calibration(CFG) and not rs.needs_calibration(CFG.replace("calibrated: false\r\n  ridge", "calibrated: true\r\n  ridge"))
    new = rs.apply_calibration(CFG, 4.0)
    assert "shift_sd: 4.0" in new and "\r\n" in new and new.count("calibrated: true") == 1
    assert new.endswith("calibrated: false\r\n")                       # the statistics block is untouched


def test_full_order_and_experiment_id_flows_into_the_analysis(tmp_path):
    d, calls, _ = make(tmp_path, ["--allow-uncalibrated", "--new-experiment", "--n-runs", "10", "--base-seed", "53"])
    assert d.execute() == 0
    assert mods(calls) == ["scripts.freeze_preregistration", "scripts.run_full_study", "scripts.run_feature_map_experiment",
                           "scripts.run_feature_map_experiment", "scripts.run_three_arm_matrix", "scripts.run_hypothesis_tests",
                           "scripts.run_temporal_validation", "scripts.run_mediation_screen"]
    hyp = next(c for c in calls if "scripts.run_hypothesis_tests" in c)
    assert "EXP-20261110-TEST-abcd" in hyp and "--feature-map-json" in hyp
    main = next(c for c in calls if "scripts.run_full_study" in c)
    assert "--no-hypothesis-tests" in main and "--new-experiment" in main and main[main.index("--base-seed") + 1] == "53"


def test_uncalibrated_poisoning_stops_before_any_run(tmp_path):
    d, calls, cfg = make(tmp_path, [], cal={"chosen_shift_sd": 4.0})
    assert d.execute() == 3                                            # chose a value but will not edit the config unasked
    assert "scripts.run_full_study" not in mods(calls) and d.state["calibrate"]["status"] == "failed"
    assert b"calibrated: false" in cfg.read_bytes()


def test_apply_calibration_writes_config_and_continues(tmp_path):
    d, calls, cfg = make(tmp_path, ["--apply-calibration"], cal={"chosen_shift_sd": 4.0})
    assert d.execute() == 0
    assert b"shift_sd: 4.0" in cfg.read_bytes() and "scripts.calibrate_poisoning" in mods(calls)


def test_no_shift_reaches_the_target_stops_with_h3_not_decidable(tmp_path):
    d, calls, _ = make(tmp_path, ["--apply-calibration"], cal={"chosen_shift_sd": None})
    assert d.execute() == 1 and "NOT decidable" in d.state["calibrate"]["note"]


def test_drift_without_deviation_stops_and_with_deviation_records_it(tmp_path):
    d, calls, _ = make(tmp_path, ["--allow-uncalibrated"], fail=["--check"])
    assert d.execute() == 2 and mods(calls) == ["scripts.freeze_preregistration"]
    d2, calls2, _ = make(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path,
                         ["--allow-uncalibrated", "--record-deviation", "Amendment 2"], fail=["--check"])
    # the fake fails every "--check" but not "--record-deviation": the run proceeds past the freeze stage
    assert d2.execute() == 0 and any("--record-deviation" in c for c in calls2)


def test_resume_skips_finished_stages_but_always_reenters_main(tmp_path):
    d, calls, _ = make(tmp_path, ["--allow-uncalibrated"], fail=["run_hypothesis_tests"])
    assert d.execute() != 0 and d.state["hypothesis"]["status"] == "failed"
    d2, calls2, _ = make(tmp_path, ["--allow-uncalibrated"])               # same state file, hypothesis now succeeds
    assert d2.execute() == 0
    m2 = mods(calls2)
    assert m2.count("scripts.run_full_study") == 1 and "scripts.run_feature_map_experiment" not in m2   # earlier stages skipped
    assert "scripts.run_hypothesis_tests" in m2


def test_optional_stage_failure_does_not_stop_the_study(tmp_path):
    d, calls, _ = make(tmp_path, ["--allow-uncalibrated"], fail=["run_temporal_validation"])
    assert d.execute() == 0 and d.state["temporal"]["status"] == "failed_optional" and d.state["screen"]["status"] == "ok"


def test_from_only_skip_and_no_id_temporal(tmp_path):
    d, calls, _ = make(tmp_path, ["--allow-uncalibrated", "--from", "hypothesis", "--skip", "screen"])
    assert d.selected() == ["hypothesis", "temporal"]
    d.experiment_id = None
    d.after_main = lambda: None
    d.execute()
    assert "scripts.run_temporal_validation" not in mods(calls)


def test_calibration_precedes_the_freeze_check_so_the_config_edit_is_covered():
    assert rs.STAGES.index("calibrate") < rs.STAGES.index("freeze") < rs.STAGES.index("main") < rs.STAGES.index("hypothesis")


def test_launcher_refuses_uncalibrated_h3_but_status_still_works(tmp_path):
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "config", "default_config.yaml"), encoding="utf8", newline="").read()
    i = re.search(r"(?m)^poisoning:", text).start()                                       # un-calibrate inside the poisoning block only
    forced = text[:i] + re.sub(r"(?m)^([ \t]+calibrated:\s*)true", r"\1false", text[i:], count=1)
    cfg = tmp_path / "cfg.yaml"; cfg.write_bytes(forced.encode())
    assert rs.needs_calibration(forced)
    r = subprocess.run([sys.executable, "-m", "scripts.run_full_study", "--config", str(cfg), "--n-runs", "1", "--legacy", "--no-prepare"],
                       cwd=root, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "poisoning.calibrated is false" in (r.stdout + r.stderr)
    ok = subprocess.run([sys.executable, "-m", "scripts.run_full_study", "--config", str(cfg), "--status"],
                        cwd=root, capture_output=True, text=True, timeout=120)
    assert "refusing to start" not in (ok.stdout + ok.stderr)


def test_run_full_study_calibrates_first_and_forwards_the_flags(monkeypatch):
    """.\\scripts\\run_full_study.ps1 must run `run_study --only calibrate --apply-calibration` before anything else."""
    import subprocess as sp
    from scripts import run_full_study as fs
    calls = []
    monkeypatch.setattr(rs, "main", lambda argv=None: calls.append(("study", list(argv))) or 0)
    monkeypatch.setattr(sp, "run", lambda cmd, **k: calls.append(("sh", list(cmd))) or sp.CompletedProcess(cmd, 0))
    ns = type("A", (), dict(config="config/default_config.yaml", allow_uncalibrated=False, record_deviation=None,
                            strict_freeze=False))()
    fs.prepare_study(ns)
    assert calls[0][0] == "study" and calls[0][1][:2] == ["--only", "calibrate"] and "--apply-calibration" in calls[0][1]
    assert any("scripts.freeze_preregistration" in c[1] for c in calls[1:])


def test_run_full_study_stops_when_calibration_fails(monkeypatch):
    from scripts import run_full_study as fs
    monkeypatch.setattr(rs, "main", lambda argv=None: 1)
    ns = type("A", (), dict(config="x", allow_uncalibrated=False, record_deviation=None, strict_freeze=False))()
    with pytest.raises(SystemExit):
        fs.prepare_study(ns)


def test_run_study_main_stage_does_not_prepare_twice(tmp_path):
    d, calls, _ = make(tmp_path, ["--allow-uncalibrated"])
    assert "--no-prepare" in d.commands("main")[0]


def test_run_full_study_defaults_to_the_reduced_25_run_plan():
    from scripts import run_full_study as fs
    assert len(fs.REDUCED_CONDITIONS) * fs.REDUCED_N_RUNS == 25 and fs.FULL_N_RUNS == 10
    assert len(fs.build_conditions(fs.DEFAULT_LEVELS)) * fs.FULL_N_RUNS == 80


def test_default_plan_resumes_matching_experiment_or_starts_new(monkeypatch):
    from scripts import run_full_study as fs
    models, seeds = list(fs.REDUCED_CONDITIONS), [43, 44, 45, 46, 47]
    man = lambda eid, c, sd: {"experiment_id": eid, "conditions": c, "seeds": sd}
    full = man("EXP-A", ["x"] * 8, list(range(43, 53)))
    red = man("EXP-B", models, seeds)
    monkeypatch.setattr(fs.exp, "list_experiments", lambda: [full, red])
    monkeypatch.setattr(fs.exp, "get_active", lambda: "EXP-A")
    assert fs.default_plan_experiment(models, seeds) == ("EXP-B", False)          # resume the matching one
    monkeypatch.setattr(fs.exp, "list_experiments", lambda: [full])
    assert fs.default_plan_experiment(models, seeds) == (None, True)              # active differs -> new experiment
    monkeypatch.setattr(fs.exp, "get_active", lambda: None)
    assert fs.default_plan_experiment(models, seeds) == (None, False)


def test_run_study_main_stage_asks_for_the_full_plan(tmp_path):
    d, calls, _ = make(tmp_path, ["--allow-uncalibrated"])
    assert "--full" in d.commands("main")[0]
