"""Save the per-row forecasts of one run, so temporal-stability and row-level tests are possible after the fact.

Reviewer claim 4 needs more than one RMSE per run: errors by test block, by VIX regime, and row-level paired tests
(Diebold-Mariano) between conditions. Every GAN run scores the SAME fixed-seed subsample of the test set for a given
seed (src/utils/generator_path.forecast_eval_indices), so the rows of QGAN seed k and classical seed k line up.

dump_eval_predictions() is called by src/experiments/run_all.py right after model.evaluate(). It re-derives the same
rows and forecasts (a second generator pass over the evaluation rows: minutes for the 20-qubit model, seconds for the
classical ones), writes results/predictions/<experiment id>/<condition slug>_seed<k>.npz and checks that the recomputed
RMSE equals the one just logged. It never raises into the run; set PREDICTION_DUMP=0 to switch it off.

npz contents: row_index (position in the one-step-shifted test set), y_true (target in the model's frame), y_pred,
y_persist (the no-change forecast in the same frame), vix (when data/processed/vix_test.npy exists) and meta (JSON).
"""
from __future__ import annotations

import json
import os

import numpy as np


def prediction_path(experiment_id: str, slug: str, seed: int, root: str = os.path.join("results", "predictions")) -> str:
    return os.path.join(root, experiment_id or "unassigned", f"{slug}_seed{int(seed)}.npz")


def dump_eval_predictions(model, model_name: str, seed: int, run_id: str, splits: dict, cfg: dict, metrics: dict,
                          logger=None, root: str = os.path.join("results", "predictions")):
    if os.environ.get("PREDICTION_DUMP", "1") == "0" or not getattr(model, "_gen_path", False):
        return None
    from src.evaluation.one_step_ahead import make_target, resolve_frame
    from src.utils.experiment import current_experiment_id, _slug
    from src.utils.generator_path import forecast_eval_indices

    frame = resolve_frame(model.config)
    te = make_target(splits["X_test"], splits["y_test"], frame)
    fsel = forecast_eval_indices(len(te.X), model.config.get("forecast_eval_samples"), model.seed)
    idx = np.arange(len(te.X)) if fsel is None else np.asarray(fsel)
    X_f = te.X[idx]
    y_true = np.asarray(te.target, dtype=float).reshape(-1)[idx]
    y_pred = np.asarray(model.predict(X_f), dtype=float).reshape(-1)
    y_persist = np.zeros_like(y_true) if frame == "delta" else np.asarray(te.last, dtype=float).reshape(-1)[idx]
    arrays = {"row_index": idx.astype(np.int64), "y_true": y_true, "y_pred": y_pred, "y_persist": y_persist}
    vix_path = os.path.join(cfg.get("data", {}).get("processed_dir", ""), "vix_test.npy")
    if os.path.isfile(vix_path):
        vix = np.load(vix_path).reshape(-1)
        if len(vix) >= len(te.X):
            arrays["vix"] = vix[:len(te.X)][idx].astype(float)
    rmse_re = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmse_logged = float(metrics.get("rmse", float("nan")))
    ok = bool(np.isfinite(rmse_logged) and abs(rmse_re - rmse_logged) <= 1e-6 * max(abs(rmse_logged), 1e-12))
    eid = current_experiment_id()
    meta = {"run_id": run_id, "model_name": model_name, "seed": int(seed), "experiment_id": eid, "frame": frame,
            "n_rows_scored": int(len(idx)), "n_rows_test_shifted": int(len(te.X)),
            "rmse_recomputed": rmse_re, "rmse_logged": rmse_logged, "matches_logged_rmse": ok}
    path = prediction_path(eid, _slug(model_name), seed, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    if logger is not None:
        logger.info(f"Saved {len(idx):,} per-row forecasts to {path}"
                    + ("" if ok else f"  WARNING: recomputed RMSE {rmse_re:.6g} != logged {rmse_logged:.6g}"))
    return path
