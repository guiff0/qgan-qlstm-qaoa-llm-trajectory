"""verify_chapter4 and run_hypothesis_tests must refuse to run before run_all has produced results."""
import subprocess
import sys

import pytest

from src.utils.require_results import count_result_rows, require_results


def test_missing_file_is_zero_rows_and_exits_with_run_all_hint(tmp_path, capsys):
    csv = tmp_path / "all_results.csv"
    assert count_result_rows(str(csv)) == 0
    with pytest.raises(SystemExit) as e:
        require_results(str(csv), "run_hypothesis_tests")
    assert e.value.code == 2
    msg = capsys.readouterr().err
    assert "src.experiments.run_all" in msg and "does not exist" in msg and "run_hypothesis_tests" in msg


def test_header_only_file_counts_as_empty(tmp_path, capsys):
    csv = tmp_path / "all_results.csv"
    csv.write_text("run_id,model_type,seed\n\n")
    assert count_result_rows(str(csv)) == 0
    with pytest.raises(SystemExit) as e:
        require_results(str(csv), "verify_chapter4")
    assert e.value.code == 2 and "has no data rows" in capsys.readouterr().err


def test_file_with_rows_passes_and_returns_the_count(tmp_path):
    csv = tmp_path / "all_results.csv"
    csv.write_text("run_id,model_type,seed\nqgan_llm_20261031T000000Z,QGAN-LLM,42\nclassical_lstm_20261031T000001Z,Classical LSTM,42\n")
    assert require_results(str(csv), "verify_chapter4") == 2


@pytest.mark.parametrize("module", ["scripts.run_hypothesis_tests", "scripts.verify_chapter4"])
def test_scripts_stop_with_exit_code_2_on_an_empty_results_dir(tmp_path, module):
    """End to end: in a directory with no results the script exits non-zero and tells you to run run_all."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": root}
    args = [sys.executable, "-m", module]
    if module.endswith("run_hypothesis_tests"):
        args += ["--results-csv", str(tmp_path / "results" / "all_results.csv")]
    proc = subprocess.run(args, cwd=root if module.endswith("verify_chapter4") else tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "src.experiments.run_all" in (proc.stderr + proc.stdout)
