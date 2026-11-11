"""Checkpoint inspection without torch, the three-arm ledger, and the chronology."""
import io
import json
import os
import pickle
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import list_checkpoints as lc
from src.evaluation import arm_ledger as al
from src.utils import ckpt_inspect as ci


import contextlib
import types


@contextlib.contextmanager
def _torch_stub():
    """While pickling, make `torch._utils._rebuild_tensor_v2` importable by name, as in a real checkpoint (restored after)."""
    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils", "torch_stub")}
    def _rebuild_tensor_v2(*a, **k): raise RuntimeError("never called: the reader replaces it")
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    stub = types.ModuleType("torch_stub"); stub._rebuild_tensor_v2 = _rebuild_tensor_v2
    tu = types.ModuleType("torch._utils"); tu._rebuild_tensor_v2 = _rebuild_tensor_v2
    sys.modules.update({"torch": types.ModuleType("torch"), "torch._utils": tu, "torch_stub": stub})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _fake_ckpt(path, fp="abc1234567", completed=True, epoch=30, entropies=(1.9, 0.8), head_mse=1e-8):
    """A real torch-format zip (data.pkl + data/<key>) built by hand: persistent ids for the tensors."""
    tensors = {}
    class P:                                                   # pickles as a persistent id, like torch storages
        def __init__(self, key): self.key = key
    arrays = {"theta": np.arange(240, dtype=np.float32) * 0.01, "hw": np.ones((1, 64), np.float32)}
    names = {k: str(i) for i, k in enumerate(arrays)}
    buf = io.BytesIO()
    class Pk(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, P):
                return ("storage", "FloatStorage", obj.key, "cpu", arrays_by_key[obj.key].size)
    arrays_by_key = {names[k]: v for k, v in arrays.items()}
    import collections
    def tensor(k):
        a = arrays[k]
        return _T(P(names[k]), a.shape)
    class _T:
        def __init__(self, st, shape): self.st, self.shape = st, shape
        def __reduce__(self):
            import torch_stub
            return (torch_stub._rebuild_tensor_v2, (self.st, 0, tuple(int(x) for x in self.shape), tuple(int(x) for x in np.cumprod((1,) + self.shape[:0:-1])[::-1]), False, None))
    ck = {"fingerprint": fp, "next_epoch": epoch, "completed": completed, "target_frame": "delta",
          "generator": {"theta": tensor("theta")}, "forecast_head": {"linear.weight": tensor("hw")},
          "head_mse": head_mse, "g_optimizer": {"s": 1}, "d_optimizer": {"s": 1},
          "entanglement_history": [{"epoch": 4 + 5 * i, "entanglement_entropy": e, "max_possible_entropy": 10.0}
                                   for i, e in enumerate(entropies)]}
    with _torch_stub():
        Pk(buf, protocol=2).dump(ck)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("m/data.pkl", buf.getvalue())
        for k, key in names.items():
            z.writestr(f"m/data/{key}", arrays[k].tobytes())


def test_reader_describes_a_complete_and_an_incomplete_checkpoint(tmp_path):
    _fake_ckpt(tmp_path / "qgan_llm_abc1234567.ckpt")
    _fake_ckpt(tmp_path / "qgan_llm_def1234567.ckpt", fp="def1234567", completed=False, epoch=8, entropies=(1.94,), head_mse=float("nan"))
    a = ci.describe(str(tmp_path / "qgan_llm_abc1234567.ckpt"))
    b = ci.describe(str(tmp_path / "qgan_llm_def1234567.ckpt"))
    assert a["state"] == "COMPLETE" and a["fingerprint_matches_name"] and a["theta_params"] == 240 and a["head_input_dim"] == 64
    assert a["entropy_first"] == 1.9 and a["entropy_last"] == 0.8
    assert b["state"].startswith("INCOMPLETE (next epoch 8") and not b["head_fitted"]


def test_reader_refuses_unknown_globals(tmp_path):
    p = tmp_path / "qgan_llm_bad0000000.ckpt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("m/data.pkl", pickle.dumps(os.system))              # a hostile pickle must not execute anything
    r = ci.describe(str(p))
    assert r["state"] == "UNREADABLE" and "refusing" in r["error"]


def test_lock_files_in_both_spellings(tmp_path):
    _fake_ckpt(tmp_path / "qgan_llm_abc1234567.ckpt")
    (tmp_path / "qgan_llm_abc1234567_ckpt_train.lock").write_text("999999\n2026-11-10T21:56:00Z\n")
    lk = ci.lock_info(str(tmp_path / "qgan_llm_abc1234567.ckpt"))
    assert lk["pid"] == 999999 and lk["utc"] == "2026-11-10T21:56:00Z" and lk["alive"] is False
    os.remove(tmp_path / "qgan_llm_abc1234567_ckpt_train.lock")
    (tmp_path / "qgan_llm_abc1234567.ckpt.train.lock").write_text(f"{os.getpid()}\nx\n")
    assert ci.lock_info(str(tmp_path / "qgan_llm_abc1234567.ckpt"))["alive"] is True


def test_cli_lists_and_notes_the_baseline_has_no_checkpoint(tmp_path, capsys):
    _fake_ckpt(tmp_path / "qgan_llm_abc1234567.ckpt")
    np.save(tmp_path / "qgan_llm_zzz0000000_synth.npy", np.zeros((5, 3)))
    assert lc.main(["--models-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "COMPLETE" in out and "NOT HERE BY DESIGN" in out and "WITHOUT a checkpoint" in out


def test_identify_maps_fingerprints_with_injected_factories():
    class M:
        def __init__(self, c): self.config = c
    fac = {"QGAN-LLM": lambda s: M({"noise": 0.01}), "QGAN-LLM (no noise)": lambda s: M({"noise": 0.0})}
    fp = lambda cfg, seed, tag: f"{tag}-{cfg['noise']}-{seed}"
    m = lc.expected_fingerprints(fac, ["QGAN-LLM", "QGAN-LLM (no noise)"], [43, 44], fp)
    recs = [{"fingerprint": "qgan_llm-0.0-44"}, {"fingerprint": "nope"}]
    lc.identify(recs, m)
    assert recs[0]["condition"] == "QGAN-LLM (no noise)" and recs[0]["seed"] == 44 and recs[1]["condition"] is None


def test_ledger_arms_chronology_and_missing_three_arm(tmp_path):
    models, res = tmp_path / "models", tmp_path / "results"
    models.mkdir(); res.mkdir()
    _fake_ckpt(models / "qgan_llm_abc1234567.ckpt")
    (res / "preregistration_20261109T093144Z.json").write_text("{}")
    (res / "poisoning_calibration_20261109T094652Z.json").write_text("{}")
    df = pd.DataFrame({"run_id": ["qgan_llm_20261107T100913Z", "classical_gan_llm_20261109T094817Z"], "seed": [43, 44],
                       "persistence_rmse": [1e-4, 1e-4]})
    led = al.build_ledger(df, str(models), str(res))
    arms = {a["arm"]: a for a in led["arms"]}
    assert "none by design" in arms["baseline"]["checkpoint"] and arms["quantum"]["checkpoint"].startswith("1 complete")
    assert not led["three_arm_ran"] and "NO results/feature_map_advantage" in al.format_ledger(led)
    s = led["chronology_summary"]
    assert s["n_runs"] == 2 and s["runs_before_calibration"] == 1 and s["runs_before_first_freeze"] == 1
    kinds = [e["kind"] for e in led["chronology"]]
    assert kinds == ["run", "freeze", "calibration", "run"]


def test_ledger_counts_feature_map_seeds_and_skips_versioned_copies(tmp_path):
    res = tmp_path / "results"; res.mkdir()
    rows = [{"seed": s, "arm": a} for s in (0, 1) for a in ("linear", "classical_rf", "quantum")]
    for name in ("feature_map_advantage_q8.json", "feature_map_advantage_q8__20261109T000000Z.json"):
        (res / name).write_text(json.dumps({"per_run": rows, "args": {"n_qubits": 8}}))
    led = al.build_ledger(pd.DataFrame({"run_id": []}), str(tmp_path / "models"), str(res))
    assert list(led["feature_map"]) == ["feature_map_advantage_q8.json"] and led["three_arm_ran"]
    assert {a["arm"]: a["feature_map_seeds"] for a in led["arms"]} == {"baseline": 2, "classical": 2, "quantum": 2}
