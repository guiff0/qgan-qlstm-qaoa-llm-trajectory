import json, os
from pathlib import Path
from src.utils import run_identity as ri
from src.utils.logging_utils import RunLogger, make_run_id


def test_run_ids_never_collide(tmp_path):
    ids = set()
    for _ in range(5):
        rid = ri.unique_run_id("QGAN-LLM (no noise)", str(tmp_path))
        assert rid not in ids and ri.config_of_run_id(rid) == "qgan_llm_no_noise"
        (tmp_path / f"{rid}.json").write_text("{}")
        ids.add(rid)
    assert len(ids) == 5


def test_pipeline_id_is_shared_through_env(monkeypatch):
    monkeypatch.delenv(ri.ENV_PIPELINE_ID, raising=False)
    a = ri.pipeline_run_id(); assert a.startswith("P") and ri.pipeline_run_id() == a and os.environ[ri.ENV_PIPELINE_ID] == a


def test_archive_keeps_every_distinct_version(tmp_path):
    f = tmp_path / "chapter4_numbers.json"
    f.write_text('{"v": 1}'); a1 = ri.archive_file(f, tag="T1")
    assert ri.archive_file(f, tag="T2") == a1            # identical content: no duplicate copy
    f.write_text('{"v": 2}'); a2 = ri.archive_file(f, tag="T3")
    assert a1 != a2 and json.load(open(a1))["v"] == 1 and json.load(open(a2))["v"] == 2
    a3 = ri.archive_file(f, tag="T3")
    assert a3 == a2
    f.write_text('{"v": 3}'); a4 = ri.archive_file(f, tag="T3")
    assert a4 not in (a1, a2) and Path(a2).exists()      # same tag, new content: numeric suffix, nothing overwritten


def test_save_json_versioned_and_ledger(tmp_path):
    f = tmp_path / "numbers.json"
    ri.save_json_versioned(f, {"v": 1}, tag="A"); old = ri.save_json_versioned(f, {"v": 2}, tag="B")
    assert json.load(open(old))["v"] == 1 and json.load(open(f))["v"] == 2
    ri.append_ledger(str(tmp_path), {"run_id": "a", "rmse": 1})
    ri.append_ledger(str(tmp_path), {"run_id": "b", "rmse": 2, "asr": 0.3})
    import pandas as pd
    led = pd.read_csv(tmp_path / "history" / "all_results_ledger.csv")
    assert list(led.run_id) == ["a", "b"] and "asr" in led.columns


def test_results_row_survives_deleting_all_results_csv(tmp_path):
    lg = RunLogger(make_run_id("Classical LSTM", str(tmp_path / "logs")), log_dir=str(tmp_path / "logs"), results_dir=str(tmp_path / "results"))
    lg.log_config({"x": 1}); lg.append_to_results_csv("Classical LSTM", {"rmse": 0.1})
    os.remove(tmp_path / "results" / "all_results.csv")                       # simulate a reset / overwrite
    lg2 = RunLogger(make_run_id("Classical LSTM", str(tmp_path / "logs")), log_dir=str(tmp_path / "logs"), results_dir=str(tmp_path / "results"))
    lg2.log_config({"x": 1}); lg2.append_to_results_csv("Classical LSTM", {"rmse": 0.2})
    import pandas as pd
    led = pd.read_csv(tmp_path / "results" / "history" / "all_results_ledger.csv")
    assert len(led) == 2 and led.pipeline_run_id.notna().all() and lg.run_id != lg2.run_id
    assert "config_sha256" in lg.record


def test_rebuild_ledger_from_logs_is_idempotent(tmp_path):
    import json
    from scripts.rebuild_ledger import main
    (tmp_path / "logs").mkdir(); (tmp_path / "results").mkdir()
    for rid in ("classical_lstm_20261001T000000Z", "qgan_llm_20261002T000000Z"):
        json.dump({"run_id": rid, "environment": {"timestamp_utc": "2026-10-01T00:00:00+00:00", "git_commit": "abc"},
                   "final_metrics": {"rmse": 0.1, "latency_raw_samples_ms": [1, 2]}}, open(tmp_path / "logs" / f"{rid}.json", "w"))
    assert main(str(tmp_path / "logs"), str(tmp_path / "results")) == 2
    assert main(str(tmp_path / "logs"), str(tmp_path / "results")) == 0
