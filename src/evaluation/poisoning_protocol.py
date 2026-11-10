"""H3 poisoning protocol v2: an attack that can move the outcome, a deterministic head, and a positive control.

WHAT WENT WRONG IN PROTOCOL v1 (src/evaluation/poisoning_resistance.py, `legacy`)
  * The attack reflected 5 % of the targets about their mean. A one-step price change is symmetric about its mean and a
    linear head on these features has almost no signal to lose, so the poisoned training set was, in distribution,
    the clean one. Measured degradation: |x| < 0.17 % in every one of the 70 logged runs (median ~1e-5 %).
  * The head was 20 full-batch Adam steps at lr 0.01 on targets of scale ~1e-4, i.e. barely trained, so the
    clean and poisoned heads differ by optimiser noise, not by the poison.
  * There was no positive control, so "the attack does nothing" and "the generator's noise does nothing" looked the same.

WHAT v2 DOES
  1. Attack: `inject_triggered_shift` (src/attacks/poisoning.py), a feature-triggered target shift the head can learn.
  2. Head: closed-form ridge on standardised features and a standardised target. Deterministic: the same data and seed
     give the same head, so a clean-vs-poisoned difference is the poison and nothing else.
  3. Positive control: the SAME attack against a head trained on REAL rows only (no generator involved). If that head does
     not degrade by at least `pc_floor_pct`, the attack is too weak and H3 is not decidable whatever the generator does.
     The control does not use the generator, the noise level or the condition, so it cannot favour either side of H3.
  4. Strength is calibrated ONCE, on pilot seeds outside the confirmatory seed list, with the positive control only
     (`calibrate_shift`, scripts/calibrate_poisoning.py), and then frozen in config `poisoning.shift_sd`. It is never
     tuned per condition or after seeing a confirmatory result.

Pure numpy on purpose: this module has no torch/pennylane dependency, so it is unit-tested without them.
"""
from __future__ import annotations

import numpy as np

from ..attacks.poisoning import inject_triggered_shift

PROTOCOL_NAME = "v2_triggered_shift"
DEFAULTS = {"fraction": 0.2, "shift_sd": 3.0, "ridge": 1.0, "pc_floor_pct": 5.0, "pc_max_rows": 100_000,
            "pc_target_pct": 10.0}


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> dict:
    """Closed-form ridge on standardised X; y is centred. Returns a small dict that `predict` understands."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    ym = float(y.mean())
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "b": ym}


def predict(head: dict, X: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float64) - head["mu"]) / head["sd"]) @ head["w"] + head["b"]


def rmse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def degradation(X_train, y_train, X_eval, y_eval, *, fraction: float, shift_sd: float, seed: int,
                ridge: float = 1.0, direction=None) -> dict:
    """Same recipe, same rows, same seed; only the targets differ. Degradation = (poisoned - control) / control, in %."""
    atk = inject_triggered_shift(X_train, y_train, fraction=fraction, shift_sd=shift_sd, seed=seed, direction=direction)
    control = fit_ridge(X_train, y_train, ridge)
    poisoned = fit_ridge(X_train, atk["y_poisoned"], ridge)
    c, p = rmse(predict(control, X_eval), y_eval), rmse(predict(poisoned, X_eval), y_eval)
    return {"control_rmse": c, "poisoned_rmse": p,
            "degradation_pct": (p - c) / c * 100.0 if c > 0 and np.isfinite(c) else float("nan"),
            "n_poisoned": atk["n_poisoned"], "direction": atk["direction"]}


def run_protocol(X_mixed, y_train, X_eval, y_eval, *, X_real_train=None, seed: int = 0, cfg: dict | None = None) -> dict:
    """Score one run. X_mixed = the training mix the model contributes (real + this generator's synthetic rows);
    X_real_train = the same rows with REAL features only (positive control). Returns flat, CSV-ready keys."""
    c = {**DEFAULTS, **(cfg or {})}
    fraction, shift = float(c["fraction"]), float(c["shift_sd"])
    ridge = float(c["ridge"])
    main = degradation(X_mixed, y_train, X_eval, y_eval, fraction=fraction, shift_sd=shift, seed=seed, ridge=ridge)
    out = {"poisoned_rmse": main["poisoned_rmse"], "poisoning_control_rmse": main["control_rmse"],
           "poisoning_rmse_degradation_pct": main["degradation_pct"], "poisoning_fraction": fraction,
           "poisoning_n_poisoned": main["n_poisoned"], "poisoning_protocol": PROTOCOL_NAME,
           "poisoning_shift_sd": shift, "poisoning_pc_floor_pct": float(c["pc_floor_pct"]),
           "poisoning_calibrated": bool(c.get("calibrated", False))}
    if X_real_train is not None:
        n = min(len(X_real_train), int(c["pc_max_rows"]))
        pc = degradation(np.asarray(X_real_train)[:n], np.asarray(y_train)[:n], X_eval, y_eval, fraction=fraction,
                         shift_sd=shift, seed=seed, ridge=ridge)
        out.update({"poisoning_pc_degradation_pct": pc["degradation_pct"], "poisoning_pc_control_rmse": pc["control_rmse"]})
    return out


def calibrate_shift(X_train, y_train, X_eval, y_eval, *, seeds, fraction: float = 0.2, grid=(0.5, 1, 2, 3, 4, 6, 8),
                    target_pct: float = 10.0, ridge: float = 1.0) -> dict:
    """Smallest shift_sd whose REAL-ONLY positive control degrades by >= target_pct in the median pilot seed.
    Pilot seeds must lie outside the confirmatory seed list. Uses no generator and no condition, so it cannot favour
    the noisy or the un-noised arm. Returns the chosen value and the whole curve (so the choice can be audited)."""
    curve = []
    for s in grid:
        d = [degradation(X_train, y_train, X_eval, y_eval, fraction=fraction, shift_sd=float(s), seed=int(sd),
                         ridge=ridge)["degradation_pct"] for sd in seeds]
        curve.append({"shift_sd": float(s), "median_degradation_pct": float(np.nanmedian(d)),
                      "min_degradation_pct": float(np.nanmin(d)), "per_seed": [float(x) for x in d]})
    chosen = next((r["shift_sd"] for r in curve if r["median_degradation_pct"] >= target_pct), None)
    return {"chosen_shift_sd": chosen, "target_pct": target_pct, "fraction": fraction, "pilot_seeds": [int(s) for s in seeds],
            "curve": curve,
            "note": ("no shift in the grid reaches the target: the head has no usable structure to damage on these "
                     "features; H3 stays not decidable" if chosen is None else "frozen value; record it in the amendment")}
