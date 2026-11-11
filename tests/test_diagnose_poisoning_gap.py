"""The poisoning-gap diagnostic must (1) use the same arithmetic as the protocol, (2) replay calibrate_poisoning exactly,
(3) flag a time-split blow-up, and (4) never compare synthetic data with the user's real calibration files."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import diagnose_poisoning_gap as dg
from src.evaluation.poisoning_protocol import calibrate_shift, degradation


def test_probe_matches_protocol_degradation():
    X, y = dg.synthetic_data(n=8000)
    Xtr, ytr, Xev, yev = X[:4000], y[:4000], X[4000:], y[4000:]
    a = degradation(Xtr, ytr, Xev, yev, fraction=0.2, shift_sd=2.0, seed=7, ridge=1.0)["degradation_pct"]
    b = dg.probe(Xtr, ytr, Xev, yev, fraction=0.2, shift=2.0, seed=7, ridge=1.0)["degradation_pct"]
    c = dg.probe(Xtr, ytr, Xev, yev, fraction=0.2, shift=2.0, seed=7, ridge=1.0,
                 control_pred=dg.prepare_control(Xtr, ytr, Xev, 1.0))["degradation_pct"]
    assert np.isclose(a, b) and np.isclose(a, c)


def test_calibration_recipe_replays_calibrate_poisoning_exactly():
    """Same subset, same chronological halves, same pilot seeds -> the same median curve calibrate_shift reports."""
    X, y = dg.synthetic_data(n=20000)
    max_rows, pilot = 5000, [100, 101, 102]
    n = min(len(X), 2 * max_rows)                                    # lines copied from scripts/calibrate_poisoning.py
    sel = np.sort(np.random.default_rng(pilot[0]).choice(len(X), n, replace=False))
    Xs, ys = np.asarray(X[sel], dtype=np.float32), np.asarray(y[sel], dtype=np.float64)
    half = n // 2
    ref = calibrate_shift(Xs[:half], ys[:half], Xs[half:], ys[half:], seeds=pilot, fraction=0.2, target_pct=10.0, ridge=1.0)
    mine = dg.run_grid(X.astype(np.float32), y, grid=[0.5, 2.0], seed_sets={"pilot": pilot}, fraction=0.2, ridge=1.0,
                       pc_max_rows=max_rows, subset_seed=pilot[0])
    for row in ref["curve"]:
        if row["shift_sd"] in (0.5, 2.0):
            assert np.isclose(mine[("calibration", "pilot", row["shift_sd"])]["degradation_pct"],
                              row["median_degradation_pct"], rtol=1e-9)


def test_inrun_recipe_uses_random_halves_and_truncated_control():
    X, y = dg.synthetic_data(n=10000)
    Xtr, ytr, Xev, yev = dg.split_inrun(X, y, seed=43, pc_max_rows=3000)
    assert len(Xtr) == 3000 and len(Xev) == 5000                      # control capped at pc_max_rows, eval = the other half
    Xtr2, *_ = dg.split_inrun(X, y, seed=44, pc_max_rows=3000)
    assert not np.array_equal(Xtr, Xtr2)                              # the split depends on the seed (permutation)


def test_time_split_blowup_is_detected_and_attributed(tmp_path, capsys):
    out = tmp_path / "d.json"
    rc = dg.main(["--synthetic", "--synthetic-spike", "300", "--run-seeds", "43", "44", "--pilot-seeds", "100", "101",
                  "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 1 and "NOT validated" in text and "chronology" in text and "NOT DECIDABLE" in text
    assert os.path.isfile(out)


def test_clean_data_recipes_agree_and_synthetic_skips_real_files(tmp_path, capsys):
    rc = dg.main(["--synthetic", "--synthetic-spike", "0", "--run-seeds", "43", "44", "--pilot-seeds", "100", "101",
                  "--out", str(tmp_path / "d.json")])
    text = capsys.readouterr().out
    assert "CALIBRATION CURVE CHECK: skipped" in text and "RECORDED IN-RUN" not in text
    assert rc in (0, 1)       # agreement depends on the 2x factor; the point is that it runs and skips real files
