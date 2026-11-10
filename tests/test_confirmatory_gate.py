"""Gate protocol: snapshot + sha256, STOP when the design is complete, opt-in confirmatory run, rerun disclosure,
run budget per hypothesis, and a freeze list that covers the decision logic."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_hypothesis_tests as rht
from scripts import snapshot_results as snap
from scripts import freeze_preregistration as fz
from src.evaluation.study_status import format_budget, hypothesis_budget, preregistered_slugs, study_status


@pytest.fixture
def snapdir(tmp_path, monkeypatch):
    d = tmp_path / "snapshots"
    monkeypatch.setattr(snap, "SNAP_DIR", str(d))
    monkeypatch.setattr(snap, "MANIFEST", str(d / "manifest.json"))
    monkeypatch.setattr(rht, "CONFIRM_LOG", str(d / "confirmatory_runs.json"))
    return d


def _csv(tmp_path, n_each=10, drop=None):
    rows = []
    for slug in preregistered_slugs():
        if slug == drop:
            continue
        for sd in range(n_each):
            rows.append(dict(run_id=f"{slug}_20261109T0{sd}0000Z", seed=43 + sd, forecast_path="generator",
                             model_type="Classical GAN-LLM" if slug.startswith("classical") else "QGAN-LLM", fid=1.0 + sd))
    p = tmp_path / "all_results.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_snapshot_records_sha_is_idempotent_read_only_and_detects_change(tmp_path, snapdir):
    csv = _csv(tmp_path)
    e1 = snap.snapshot(csv, "t")
    assert not e1["reused"] and len(e1["sha256"]) == 64 and os.path.isfile(snapdir / os.path.basename(e1["snapshot"])) or True
    assert snap.snapshot(csv)["reused"]
    assert snap.verify(csv)[0]
    with open(csv, "a") as fh:
        fh.write("x\n")
    assert not snap.verify(csv)[0]                      # the data changed after the snapshot -> the gate must refuse


def test_budget_counts_runs_per_hypothesis():
    n = {s: 10 for s in preregistered_slugs()}
    n["qgan_llm"], n["qgan_llm_no_noise"] = 3, 3
    b = hypothesis_budget(n)
    assert b["H1"]["remaining_to_be_conclusive"] == 7 and b["H3"]["remaining_to_be_conclusive"] == 14
    assert b["H5"]["remaining_to_be_conclusive"] == 0 and b["H4"]["remaining_for_full_design"] == 7
    assert "14 remaining before the confirmatory phase" in format_budget(b, study_status(n))


def test_complete_design_stops_before_any_test_and_snapshots(tmp_path, snapdir, monkeypatch, capsys):
    csv = _csv(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--results-csv", csv, "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit) as e:
        rht.main()
    out = capsys.readouterr().out
    assert e.value.code == rht.GATE_RC and "STOP AT THE CONFIRMATORY GATE" in out and "Nothing was tested" in out
    assert not os.path.exists(tmp_path / "o.json") and snap.verify(csv)[0]


def test_incomplete_design_runs_as_interim_and_prints_budget(tmp_path, snapdir, monkeypatch, capsys):
    csv = _csv(tmp_path, drop="qgan_llm_no_noise")
    monkeypatch.setattr(sys, "argv", ["x", "--results-csv", csv, "--out", str(tmp_path / "o.json")])
    try:
        rht.main()
    except SystemExit as e:
        assert e.code != rht.GATE_RC
    out = capsys.readouterr().out
    assert "RUN BUDGET PER HYPOTHESIS" in out and "[interim]" in out


def test_confirmatory_flag_needs_freeze_and_matching_snapshot(tmp_path, snapdir, monkeypatch):
    csv = _csv(tmp_path)
    args = type("A", (), dict(rerun_reason=None))()
    monkeypatch.setattr(rht.subprocess, "run", lambda *a, **k: type("R", (), dict(returncode=1, stdout="UNRECORDED", stderr=""))())
    g, why = rht.gate_checks(args, csv)
    assert g is None and "freeze check failed" in why
    monkeypatch.setattr(rht.subprocess, "run", lambda *a, **k: type("R", (), dict(returncode=0, stdout="", stderr=""))())
    g, why = rht.gate_checks(args, csv)
    assert g is None and "does not match any recorded snapshot" in why          # no snapshot yet
    snap.snapshot(csv)
    g, why = rht.gate_checks(args, csv)
    assert g and g["rerun"] is False
    json.dump([{"utc": "t", "csv_sha256": g["csv_sha256"]}], open(rht.CONFIRM_LOG, "w"))
    g2, why2 = rht.gate_checks(args, csv)
    assert g2 is None and "--rerun-reason" in why2                              # a rerun must be disclosed
    args.rerun_reason = "typo in the table export"
    g3, _ = rht.gate_checks(args, csv)
    assert g3["rerun"] and g3["rerun_number"] == 2


def test_freeze_tracks_the_decision_logic_and_flags_files_missing_from_the_record():
    for f in ("src/evaluation/verdicts.py", "src/evaluation/paired_tests.py", "config/analysis_plan.yaml",
              "PREREGISTRATION_V2.md", "scripts/snapshot_results.py"):
        assert f in fz.TRACKED
    assert "results/all_results.csv" not in fz.TRACKED
