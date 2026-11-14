"""The pilot table is descriptive, one seed, and must stay out of the verdicts and off the three-arm path."""
import ast
import os

import pandas as pd

from scripts import pilot_table as pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _df():
    return pd.DataFrame([
        {"run_id": "classical_lstm_20261024T225906Z", "seed": 42, "rmse": 0.0056, "persistence_rmse": 0.000136, "rmse_over_persistence": 41.2},
        {"run_id": "classical_lstm_20261031T173907Z", "seed": 42, "rmse": 0.00517, "persistence_rmse": 0.000136, "rmse_over_persistence": 38.07},
        {"run_id": "qlstm_forecaster_20261031T173907Z", "seed": 42, "rmse": 0.00012, "persistence_rmse": 0.000136, "rmse_over_persistence": 0.887},
        {"run_id": "qaoa_llm_20261031T184251Z", "seed": 42, "rmse": 0.000152, "persistence_rmse": 0.000136, "rmse_over_persistence": 1.117,
         "llm_rmse_over_persistence": 1.179, "llm_n_forecasts": 102.0, "selected_features": "Stoch_K_14_3,high_low_ratio"},
        {"run_id": "qgan_llm_20261113T163450Z", "seed": 51, "rmse": 0.00014, "persistence_rmse": 0.000136, "rmse_over_persistence": 1.02},
    ])


def test_latest_run_per_pilot_model_and_study_conditions_are_left_out():
    tbl = pt.build(_df())
    by = {r["slug"]: r for r in tbl["rows"]}
    assert set(by) == {"classical_lstm", "qlstm_forecaster", "qaoa_llm"}            # qgan_llm is a study condition: not here
    assert by["classical_lstm"]["latest"]["run_utc"] == "20261031T173907Z" and len(by["classical_lstm"]["earlier"]) == 1
    assert by["qlstm_forecaster"]["latest"]["beats_persistence"] is True and by["classical_lstm"]["latest"]["beats_persistence"] is False
    assert by["qaoa_llm"]["latest"]["n_selected_features"] == 2 and by["qaoa_llm"]["latest"]["llm_n_forecasts"] == 102


def test_frame_comes_from_the_row_when_it_records_one():
    new = {"run_id": "classical_lstm_20261115T010101Z", "seed": 42, "rmse": 0.00015, "persistence_rmse": 0.000136,
           "rmse_over_persistence": 1.1, "target_frame": "delta"}
    df = pd.concat([_df(), pd.DataFrame([new])], ignore_index=True)
    row = {r["slug"]: r for r in pt.build(df)["rows"]}["classical_lstm"]
    assert row["latest"]["head_regresses"] == "next-bar change" and row["earlier"][-1]["head_regresses"] == "price level"


def test_table_says_descriptive_and_carries_no_inference():
    tbl = pt.build(_df())
    md = pt.to_markdown(tbl, all_runs=True)
    assert "DESCRIPTIVE ONLY" in md and "one seed" in md and "(earlier, superseded)" in md
    flat = str(tbl).lower()
    for word in ("p_value", "p-value", "verdict\":", "significant", "ci95", "supported"):
        assert word not in flat.replace("hypothesis verdict", "")


def test_pilot_script_imports_nothing_from_the_study_or_the_three_arm_path():
    tree = ast.parse(open(os.path.join(ROOT, "scripts", "pilot_table.py"), encoding="utf8").read())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and (m.startswith("src") or m.startswith("scripts")) for m in mods), mods
    for consumer in ("scripts/run_hypothesis_tests.py", "scripts/run_walk_forward.py", "scripts/run_naive_baseline.py",
                     "scripts/run_feature_map_experiment.py", "src/evaluation/verdicts.py", "src/evaluation/three_arm.py"):
        assert "pilot_table" not in open(os.path.join(ROOT, consumer), encoding="utf8").read()
