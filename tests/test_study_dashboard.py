import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from scripts.study_dashboard import StudyDashboard, bar, parse_log  # noqa: E402
from scripts.study_diagnose import diagnose_failure, one_line  # noqa: E402

MARK = "===== attempt started 2026-11-06 01:00:00 =====\n"


def _log(tmp_path, text):
    p = tmp_path / "run.log"
    p.write_text(MARK + text, encoding="utf-8")
    return str(p)


def test_bar():
    assert bar(20, 80).startswith("[#######") and "20/80" in bar(20, 80) and "25%" in bar(20, 80)


def test_parse_log_subtask(tmp_path):
    p = _log(tmp_path, "[2026-11-06 03:00:00] START  train\n[2026-11-06 03:40:00] DONE   train  (40.0 min)\n"
                       "[2026-11-06 03:41:07] START  attacks (FGSM/PGD/CW)\nEpoch 1: 28%|##8 | 14/50\n")
    i = parse_log(p)
    assert i["open"][0] == "attacks (FGSM/PGD/CW)" and i["n_done"] == 1 and "28%" in i["progress"]


@pytest.mark.parametrize("text,rc,rule", [
    ("NvidiaApiKeyMissingError: NVIDIA_API_KEY is not set", 1, "key_missing"),
    ("LLMAuthError: HTTP 401: the key was rejected", 1, "key_rejected"),
    ("LLMModelUnavailableError: HTTP 404 for model 'x'", 1, "model_gone"),
    ("RuntimeError: giving up after 4 attempts (HTTP 429)", 1, "api_busy"),
    ("MemoryError: Unable to allocate 4.2 GiB", 1, "oom"),
    ("FileNotFoundError: Processed data not found at data\\X_train.npy", 1, "data_missing"),
    ("PermissionError: [WinError 32] being used by another process: all_results.csv", 1, "locked"),
    ("ModuleNotFoundError: No module named 'pennylane'", 1, "no_module"),
])
def test_rules(tmp_path, text, rc, rule):
    d = diagnose_failure(_log(tmp_path, text + "\n"), rc, False, 30)
    assert d["rule"] == rule
    assert "why:" in one_line(d) and "fix:" in one_line(d)


def test_module_name_in_sentence(tmp_path):
    d = diagnose_failure(_log(tmp_path, "ModuleNotFoundError: No module named 'pennylane'\n"), 1, False, 5)
    assert "pennylane" in d["cause"]


def test_exit_zero_without_row(tmp_path):
    assert diagnose_failure(_log(tmp_path, "done\n"), 0, False, 100)["rule"] == "no_row"


def test_windows_exit_codes(tmp_path):
    assert diagnose_failure(_log(tmp_path, "x\n"), 3221225786, False, 10)["rule"] == "exit_0xc000013a"
    assert diagnose_failure(_log(tmp_path, "x\n"), -1073741819, False, 10)["rule"] == "exit_0xc0000005"


def test_fast_unclassified_is_startup_and_names_error(tmp_path):
    d = diagnose_failure(_log(tmp_path, "Traceback (most recent call last):\nZeroDivisionError: division by zero\n"), 1, False, 0.6)
    assert d["rule"] == "startup" and "ZeroDivisionError" in d["cause"]


def test_only_latest_attempt_is_read(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(MARK + "MemoryError: Unable to allocate\n" + MARK + "ZeroDivisionError: boom\n", encoding="utf-8")
    assert diagnose_failure(str(p), 1, False, 0.5)["rule"] == "startup"


def test_missing_log(tmp_path):
    assert diagnose_failure(str(tmp_path / "nope.log"), 1, False, 1)["rule"] == "empty_log"


def test_dashboard_failure_banner(tmp_path, capsys):
    done = set()
    models, seeds = ["A", "B"], [1, 2]
    d = StudyDashboard(models, seeds, lambda: [(m, s) for s in seeds for m in models if (m, s) not in done],
                       lambda m, s: str(tmp_path / "x.log"), interval=999)
    d.run_started("A", 1, "x.log")
    d.run_finished("A", 1, False, 1, str(tmp_path / "x.log"), "why: test -> fix: test")
    out = capsys.readouterr().out
    assert "FAIL" in out and "why: test" in out and "4 runs; 4 remaining" in out and "retry queue" in out


def test_status_block_shows_experiment_run_id_progress_and_three_arm(tmp_path, capsys):
    import json
    from scripts.study_dashboard import read_run_id, pipeline_lines
    log = tmp_path / "a.log"
    log.write_text(MARK + "Starting training run for A (run_id=qgan_llm_0093fcc984_20261109T094652Z)\n"
                   "[2026-11-09 04:50:03] START  train\n", encoding="utf-8")
    assert read_run_id(str(log)) == "qgan_llm_0093fcc984_20261109T094652Z"
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "feature_map_advantage_q8.json").write_text(
        json.dumps({"per_run": [{"seed": 0, "arm": "linear"}, {"seed": 0, "arm": "quantum"}, {"seed": 1, "arm": "linear"}]}))
    assert any("q8: 2 seed(s)" in l for l in pipeline_lines(str(tmp_path)))
    models, seeds = ["A", "B"], [1, 2]
    d = StudyDashboard(models, seeds, lambda: [("A", 2), ("B", 2)], lambda m, s: str(log), interval=999,
                       experiment_id="EXP-20261109-TEST-abcd", extra_lines_fn=lambda: pipeline_lines(str(tmp_path)))
    d.run_started("A", 1, str(log))
    capsys.readouterr()
    d.snapshot()
    out = capsys.readouterr().out
    assert "EXP-20261109-TEST-abcd" in out and "run_id: qgan_llm_0093fcc984_20261109T094652Z" in out
    assert "required 4 | done 2 | remaining 2" in out and "three-arm feature-map q8" in out and "three-arm matrix" in out
