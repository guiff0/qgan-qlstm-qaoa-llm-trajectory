"""save_json_timestamped: every generated JSON gets an immutable time-stamped twin."""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.run_identity import list_versions, save_json_timestamped  # noqa: E402

STAMP = re.compile(r"__\d{8}T\d{6}Z(_\d+)?\.json$")


def test_writes_versioned_and_latest_identical(tmp_path):
    p = tmp_path / "res.json"
    out = save_json_timestamped(p, {"a": 1}, stamp="20261031T180000Z")
    assert out["versioned"].endswith("res__20261031T180000Z.json")
    assert p.read_text() == open(out["versioned"]).read()


def test_never_overwrites_same_second(tmp_path):
    p = tmp_path / "res.json"
    save_json_timestamped(p, {"v": 1}, stamp="20261031T180000Z")
    save_json_timestamped(p, {"v": 2}, stamp="20261031T180000Z")
    names = [x.name for x in list_versions(p)]
    assert names == ["res__20261031T180000Z.json", "res__20261031T180000Z_1.json"]
    assert json.load(open(p))["v"] == 2
    assert json.load(open(list_versions(p)[0]))["v"] == 1


def test_versions_sort_chronologically_and_ignore_other_files(tmp_path):
    p = tmp_path / "res.json"
    save_json_timestamped(p, {}, stamp="20261031T180002Z")
    save_json_timestamped(p, {}, stamp="20261031T180001Z")
    (tmp_path / "res__notes.json").write_text("{}")
    assert [x.name for x in list_versions(p)] == ["res__20261031T180001Z.json", "res__20261031T180002Z.json"]


def test_nan_becomes_null_valid_json(tmp_path):
    p = tmp_path / "res.json"
    save_json_timestamped(p, {"x": float("nan"), "y": [math.inf, 1.0]})
    assert json.loads(p.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))) == {"x": None, "y": [None, 1.0]}


def test_legacy_latest_is_archived_once(tmp_path):
    p = tmp_path / "res.json"
    p.write_text('{"old": true}')
    out = save_json_timestamped(p, {"new": True}, stamp="20261031T180000Z")
    assert out["archived_legacy"] and json.load(open(out["archived_legacy"])) == {"old": True}
    assert save_json_timestamped(p, {"n": 2}, stamp="20261031T180100Z")["archived_legacy"] is None
