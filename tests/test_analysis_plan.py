import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation import analysis_plan as ap  # noqa: E402
from src.evaluation.verdicts import MIN_N_PER_GROUP, NAMES  # noqa: E402
from scripts import run_hypothesis_tests as rht  # noqa: E402

PLAN = ap.load_plan(os.path.join(ROOT, ap.PLAN_PATH))


def test_plan_covers_every_hypothesis_the_code_tests_and_is_complete():
    assert ap.validate(PLAN, list(NAMES)) == []


def test_each_missing_element_is_reported():
    bad = copy.deepcopy(PLAN)
    bad["hypotheses"]["H1_synthetic_data_fidelity"]["fallback"] = ""
    del bad["hypotheses"]["H2_forecast_rmse"]
    bad["hypotheses"]["extra_hypothesis"] = bad["hypotheses"]["H5_noise_attack_success"]
    bad["hypotheses"]["H4_poisoning_resistance_noise"]["replication"]["minimum_for_any_test"] = "three"
    problems = " | ".join(ap.validate(bad, list(NAMES)))
    for needle in ("H1_synthetic_data_fidelity: 'fallback' is empty", "H2_forecast_rmse: not in the analysis plan",
                   "extra_hypothesis: in the plan but not tested", "H4_poisoning_resistance_noise: replication.minimum_for_any_test"):
        assert needle in problems


def test_replication_thresholds_agree_with_the_verdict_constants():
    h = PLAN["hypotheses"]
    assert h["H1_synthetic_data_fidelity"]["replication"]["minimum_for_any_test"] == MIN_N_PER_GROUP
    assert len(h) == 5 and sorted(k.split("_")[0] for k in h) == ["H1", "H2", "H3", "H4", "H5"]
    assert all(v["replication"]["minimum_for_any_test"] == MIN_N_PER_GROUP for v in h.values())


def test_plan_margins_match_the_frozen_margins():
    text = " ".join(str(h["contrast"]) for h in PLAN["hypotheses"].values())
    assert "0.1 x |classical mean|" in text and "2% of persistence RMSE" in text and "5 ASR points" in text
    assert rht.EQUIV_MARGINS["fid"][1] == 0.10 and rht.EQUIV_MARGINS["rmse"][1] == 0.02 and rht.EQUIV_MARGINS["asr"][1] == 5.0


def test_planned_vs_ran_reports_fallbacks():
    res = {"H1_synthetic_data_fidelity": {"status": "ok", "note_covariate": "no usable 'vix' covariate"},
           "H2_forecast_rmse": {"status": "exploratory_single_case"},
           "H3_attack_success_qgan_vs_classical": {"status": "ok"},
           "H4_poisoning_resistance_noise": {"design": "matched_config_noise_only", "decidability": {"decidable": False, "gate": "poisoning_dynamic_range"}},
           "H5_noise_attack_success": {"design": "matched_config_noise_only"}}
    out = ap.planned_vs_ran(res)
    assert out["H1_synthetic_data_fidelity"]["fallback_triggered"] and out["H1_synthetic_data_fidelity"]["ran"] == "Welch two-sample t"
    assert out["H2_forecast_rmse"]["ran"] == "Crawford-Howell single case" and out["H2_forecast_rmse"]["fallback_triggered"]
    assert not out["H3_attack_success_qgan_vs_classical"]["fallback_triggered"]
    assert "poisoning_dynamic_range" in out["H4_poisoning_resistance_noise"]["why"]
    assert not out["H5_noise_attack_success"]["fallback_triggered"]


def test_markdown_table_has_one_row_per_hypothesis():
    md = ap.render_markdown(PLAN).splitlines()
    assert len(md) == 2 + len(PLAN["hypotheses"]) and md[2].startswith("| H1 |")


def test_output_json_records_plan_and_planned_vs_ran(tmp_path):
    import json, subprocess
    import numpy as np, pandas as pd
    rows = []
    for s in range(43, 48):
        for slug, mt in (("qgan_llm", "QGAN-LLM"), ("classical_gan_llm", "Classical GAN-LLM")):
            rows.append({"run_id": f"{slug}_20261107T10000{s % 10}Z", "model_type": mt, "seed": s, "fid": 100 + s % 7,
                         "rmse": 1e-4, "asr": 30, "persistence_rmse": 1e-4, "forecast_path": "generator"})
    pd.DataFrame(rows).to_csv(tmp_path / "a.csv", index=False)
    r = subprocess.run([sys.executable, "-m", "scripts.run_hypothesis_tests", "--results-csv", str(tmp_path / "a.csv"),
                        "--out", str(tmp_path / "o.json")], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stderr[-500:]
    out = json.load(open(tmp_path / "o.json"))
    assert out["analysis_plan_sha"] and "H1_synthetic_data_fidelity" in out["planned_vs_ran"]
