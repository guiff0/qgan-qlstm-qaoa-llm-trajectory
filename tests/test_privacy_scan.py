from src.data.privacy_scan import scan_text
from scripts.scan_run_artifacts_for_pii import scan_log_dir


def test_phone_pattern_does_not_match_ordinary_decimal_numbers():
    """Regression for the bug found while wiring this module in: the old
    phone regex matched any run of 8+ digits/spaces/dashes, flagging
    ordinary training-log numbers (losses, timestamps) as "phone
    numbers." """
    assert scan_text('"d_loss": 0.21967713476157172')["phone"] == 0
    assert scan_text("epoch 58061, batch 12345678")["phone"] == 0


def test_phone_pattern_still_catches_real_phone_numbers():
    result = scan_text("call 555-123-4567 or (555) 123-4567")
    assert result["phone"] == 2


def test_email_and_ssn_patterns_still_work():
    result = scan_text("contact jane.doe@example.com, SSN 123-45-6789")
    assert result["email"] == 1
    assert result["ssn"] == 1


def test_scan_log_dir_on_real_logs_has_no_hits(tmp_path):
    (tmp_path / "clean.log").write_text('{"rmse": 0.009011150337755, "epoch": 18}')
    assert scan_log_dir(str(tmp_path)) == {}


def test_scan_log_dir_flags_a_real_hit(tmp_path):
    (tmp_path / "leaky.json").write_text('{"contact": "jane.doe@example.com"}')
    result = scan_log_dir(str(tmp_path))
    assert len(result) == 1
    path, hits = next(iter(result.items()))
    assert hits["email"] == 1
