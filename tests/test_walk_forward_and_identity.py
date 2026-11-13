"""Walk-forward windows (chronology, no leakage, same seeds for every arm, ledger display) and checkpoint identification."""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import list_checkpoints as lc
from scripts import run_walk_forward as rwf
from src.evaluation import arm_ledger as al
from src.evaluation import walk_forward as wf
from src.utils import checkpoint_identity as cid


# ---------------------------------------------------------------- windows
def test_windows_are_chronological_disjoint_and_embargoed():
    ws = wf.make_windows(100_000, 4, 5_000, min_train=2_000, gap=1)
    assert [w["window"] for w in ws] == [1, 2, 3, 4] and ws[-1]["test_end"] == 100_000     # window K = most recent data
    for w in ws:
        assert w["train_end"] <= w["val_start"] < w["val_end"] < w["test_start"] and w["test_start"] - w["val_end"] >= 1
    assert all(a["test_end"] <= b["test_start"] for a, b in zip(ws, ws[1:]))
    assert ws[0]["n_train"] < ws[-1]["n_train"]                                              # expanding
    r = wf.make_windows(100_000, 3, 5_000, mode="rolling", rolling_len=20_000)
    assert all(w["val_end"] - w["train_start"] <= 20_000 for w in r)                         # fixed length


def test_leakage_guard_and_bad_plans_are_refused():
    ws = wf.make_windows(50_000, 3, 2_000)
    bad = [dict(ws[0], test_start=ws[0]["val_end"])]                                         # no embargo
    with pytest.raises(AssertionError):
        wf.assert_no_leak(bad)
    with pytest.raises(AssertionError):
        wf.assert_no_leak([ws[1], ws[0]])                                                    # windows out of order
    with pytest.raises(ValueError):
        wf.make_windows(5_000, 5, 2_000, min_train=2_000)                                    # not enough history
    with pytest.raises(ValueError):
        wf.make_windows(50_000, 3, 2_000, gap=0)


def test_run_windows_feeds_the_same_seeds_to_every_arm_and_never_trains_on_test_rows():
    X, y = np.random.default_rng(0).normal(size=(6000, 4)), np.cumsum(np.random.default_rng(1).normal(size=6000))
    ws = wf.make_windows(len(y), 2, 500, min_train=500)
    seen = []

    class Fake:
        def __init__(self, arm, nq, nl, ent, sc, seed): self.arm, self.seed = arm, seed
        def fit(self, Xt, yt, Xv, yv):
            seen.append((self.arm, self.seed, len(Xt), float(yt[-1]), float(yv[-1])))
            return self
        def evaluate(self, Xte, yte, **kw):
            e = np.random.default_rng(self.seed).normal(size=len(yte) - 1) * (0.5 if self.arm == "quantum" else 1.0)
            return {"skill": 0.1 if self.arm == "quantum" else 0.0, "rmse": 1.0, "persistence_rmse": 1.0, "n_test": len(e),
                    "arm": self.arm, "errors": e}
    dm = lambda a, b: {"statistic": 3.0, "p_value": 0.001, "n": len(a)}
    per_run, dms = wf.run_windows(X, y, ws, [0, 1], forecaster=Fake, dm_fn=dm, log=lambda *_: None)
    assert len(per_run) == 2 * 2 * 3 and len(dms) == 2 * 2 * 2
    assert {(r["window"], r["seed"], r["arm"]) for r in per_run} == {(w, s, a) for w in (1, 2) for s in (0, 1) for a in wf.ARMS}
    for arm, seed, n, last_train_y, last_val_y in seen:                                     # train/val end before the test window
        assert last_train_y in y[:ws[-1]["test_start"]] and n > 0
    summ = wf.summarise(per_run, dms, ws)
    assert summ["n_windows_run"] == 2 and summ["windows_quantum_beats_both"] == 2 and summ["windows_quantum_skill_positive"] == 2
    assert summ["windows"][0]["skill_median"]["quantum"] == 0.1 and summ["windows"][0]["quantum_beats"]["linear"] == 2


def test_cli_dry_run_prints_the_plan(tmp_path, capsys):
    d = tmp_path / "p"; d.mkdir()
    for s, n in (("train", 30000), ("val", 5000), ("test", 15000)):
        np.save(d / f"X_{s}.npy", np.zeros((n, 3), np.float32)); np.save(d / f"y_{s}.npy", np.zeros(n))
    assert rwf.main(["--processed-dir", str(d), "--windows", "3", "--test-len", "4000", "--min-train", "5000", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "50,000 chronological rows" in out and "3 expanding windows" in out
    assert rwf.main(["--processed-dir", str(tmp_path / "none"), "--dry-run"]) == 2


def test_ledger_shows_windows_or_says_not_run(tmp_path):
    res = tmp_path / "results"; res.mkdir()
    empty = pd.DataFrame({"run_id": []})
    txt = al.format_ledger(al.build_ledger(empty, str(tmp_path / "m"), str(res)))
    assert "WALK-FORWARD WINDOWS" in txt and "NOT RUN" in txt and "run_walk_forward --dry-run" in txt
    ws = wf.make_windows(30000, 2, 2000, min_train=2000)
    per_run = [{"window": w, "seed": s, "arm": a, "skill": {"linear": 0.0, "classical_rf": 0.001, "quantum": -0.002}[a]}
               for w in (1, 2) for s in (0, 1) for a in wf.ARMS]
    dm = [{"window": w, "seed": s, "vs": c, "statistic": -1.0, "p_value": 0.5} for w in (1, 2) for s in (0, 1) for c in ("linear", "classical_rf")]
    (res / "walk_forward_q8.json").write_text(json.dumps({"args": {"mode": "expanding", "gap": 1, "n_qubits": 8, "seeds": [0, 1]},
                                                         "windows": ws, "summary": wf.summarise(per_run, dm, ws)}))
    (res / "walk_forward_q8__20261110T000000Z.json").write_text("{}")                         # versioned copy is ignored
    txt = al.format_ledger(al.build_ledger(empty, str(tmp_path / "m"), str(res)))
    assert "walk_forward_q8.json: 2 window(s)" in txt and "NOT RUN" not in txt and "quantum beats BOTH controls in every seed in 0 of 2" in txt
    assert "-0.0020" in txt and "0/2 | 0/2" in txt


# ---------------------------------------------------------------- identification
def test_sidecar_written_once_and_shared_use_is_recorded(tmp_path):
    ck = str(tmp_path / "qgan_llm_abc1234567.ckpt")
    assert cid.write_identity_once(ck, {"PIPELINE_CONDITION": "QGAN-LLM", "PIPELINE_SEED": "44"})
    assert not cid.write_identity_once(ck, {"PIPELINE_CONDITION": "QGAN-LLM", "PIPELINE_SEED": "44"})
    assert cid.write_identity_once(ck, {"PIPELINE_CONDITION": "QGAN-LLM (no noise)", "PIPELINE_SEED": "44"})
    sc = cid.read_sidecar(ck)
    assert sc["condition"] == "QGAN-LLM" and sc["seed"] == 44 and sc["also_used_by"] == [{"condition": "QGAN-LLM (no noise)", "seed": 44}]
    assert cid.write_identity_once(str(tmp_path / "x.ckpt"), {}) is False                      # no env -> nothing written, no error


def test_archive_and_fingerprint_identification(tmp_path):
    models = tmp_path / "models"; (models / "history").mkdir(parents=True)
    (models / "history" / "qgan_llm_046dbbdca0__qgan_llm_no_noise_20261107T100913Z.ckpt").write_bytes(b"x")
    (models / "history" / "qgan_llm_046dbbdca0__garbage.ckpt").write_bytes(b"x")
    hist = cid.history_index(str(models))
    assert hist["qgan_llm_046dbbdca0"][0]["condition_slug"] == "qgan_llm_no_noise"
    csvp = tmp_path / "r.csv"
    pd.DataFrame({"run_id": ["qgan_llm_no_noise_20261107T100913Z"], "seed": [46], "experiment_id": ["EXP-1"]}).to_csv(csvp, index=False)
    runs = cid.load_run_table(str(csvp))
    rec = {"file": "qgan_llm_046dbbdca0.ckpt", "fingerprint": "046dbbdca0"}
    i = cid.identify_record(rec, str(models / "qgan_llm_046dbbdca0.ckpt"), hist, runs)
    assert i["by"] == "archive (run_id)" and i["condition"] == "qgan_llm_no_noise" and i["seed"] == 46
    other = {"file": "qgan_llm_zzzzzzzzzz.ckpt", "fingerprint": "zzzzzzzzzz"}
    assert cid.identify_record(other, "x", hist, runs, fp_map={"aaa": ("QGAN-LLM", 43)})["note"].startswith("no condition x seed")
    assert cid.identify_record(other, "x", hist, runs, fp_map={"zzzzzzzzzz": ("QGAN-LLM", 43)})["by"] == "fingerprint (recomputed)"


def test_save_checkpoint_writes_the_sidecar(tmp_path, monkeypatch):
    fake = types.ModuleType("torch")
    fake.save = lambda payload, path: open(path, "wb").write(b"ckpt")
    monkeypatch.setitem(sys.modules, "torch", fake)
    sys.modules.pop("src.utils.gan_training", None)
    from src.utils import gan_training as gt
    monkeypatch.setenv("PIPELINE_CONDITION", "QGAN-LLM (ent-L4 ring-d4)"); monkeypatch.setenv("PIPELINE_SEED", "47")
    p = str(tmp_path / "qgan_llm_abc1234567.ckpt")
    gt.save_checkpoint(p, {"a": 1})
    assert os.path.isfile(p) and cid.read_sidecar(p)["seed"] == 47
    sys.modules.pop("src.utils.gan_training", None)


def test_cli_write_sidecars_backfills_from_archive(tmp_path, capsys):
    from tests.test_arm_ledger import _fake_ckpt
    models = tmp_path / "models"; (models / "history").mkdir(parents=True)
    _fake_ckpt(models / "qgan_llm_abc1234567.ckpt")
    (models / "history" / "qgan_llm_abc1234567__qgan_llm_20261107T100913Z.ckpt").write_bytes(b"x")
    csvp = tmp_path / "r.csv"
    pd.DataFrame({"run_id": ["qgan_llm_20261107T100913Z"], "seed": [43]}).to_csv(csvp, index=False)
    assert lc.main(["--models-dir", str(models), "--csv", str(csvp), "--write-sidecars"]) == 0
    out = capsys.readouterr().out
    assert "IDENTIFIED 1 of 1" in out and "sidecars written: 1" in out
    sc = cid.read_sidecar(str(models / "qgan_llm_abc1234567.ckpt"))
    assert sc["condition"] == "qgan_llm" and sc["seed"] == 43 and "backfilled" in sc["source"]
