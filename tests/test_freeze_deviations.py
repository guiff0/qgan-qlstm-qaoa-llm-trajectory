import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import freeze_preregistration as fz  # noqa: E402


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    for p in fz.TRACKED:
        dst = tmp_path / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = os.path.join(ROOT, p)
        dst.write_bytes(open(src, "rb").read() if os.path.exists(src) else b"x")
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(fz, "ROOT", str(tmp_path))
    monkeypatch.setattr(fz, "_margins", lambda: {"alpha": 0.05, "margins": {"fid": ["rel_ref", 0.1]}})
    monkeypatch.setattr(fz, "_commit", lambda: "abc")
    monkeypatch.setattr(sys, "argv", ["x"])
    assert fz.main() == 0
    return tmp_path


def test_unrecorded_change_fails_then_recorded_deviation_passes(frozen, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--check"])
    assert fz.main() == 0
    f = frozen / "scripts" / "run_hypothesis_tests.py"
    f.write_bytes(f.read_bytes() + b"\n# analysis tooling change\n")
    assert fz.main() == 1 and "UNRECORDED" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x", "--record-deviation", "added paired-by-seed analysis"])
    assert fz.main() == 0
    rec = json.load(open(sorted((frozen / "results").glob("deviations_*.json"))[0]))
    assert "scripts/run_hypothesis_tests.py" in rec["changed_files"] and not rec["decision_rules_changed"]
    monkeypatch.setattr(sys, "argv", ["x", "--check"])
    assert fz.main() == 0
    out = capsys.readouterr().out
    assert "recorded deviation: added paired-by-seed analysis" in out and "no unrecorded changes" in out
    # a further edit after the record is unrecorded again
    f.write_bytes(f.read_bytes() + b"# more\n")
    assert fz.main() == 1


def test_changing_the_margins_can_never_be_recorded_away(frozen, monkeypatch):
    monkeypatch.setattr(fz, "_margins", lambda: {"alpha": 0.05, "margins": {"fid": ["rel_ref", 0.2]}})
    monkeypatch.setattr(sys, "argv", ["x", "--record-deviation", "loosened the FID margin"])
    assert fz.main() == 1
    monkeypatch.setattr(sys, "argv", ["x", "--check"])
    assert fz.main() == 1
