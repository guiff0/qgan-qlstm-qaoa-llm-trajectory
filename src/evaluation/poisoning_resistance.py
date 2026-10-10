"""
H3 ("Detection Accuracy" section title, but Ha3's actual claim is about
data-poisoning and model-inversion resistance) operationalization.

Ha3: "QGANs trained with controlled quantum noise injection generate
data that significantly increases the resistance of the LLM to data
poisoning and model inversion attacks, as measured by lower performance
degradation on poisoned datasets." Two sub-claims, both wired here --
neither had any implementation anywhere in this codebase before this:

  1. DATA POISONING resistance, measured as RMSE degradation when a
     downstream forecaster is trained on a mix of real + THIS MODEL'S
     OWN generated synthetic data (using `synthetic_ratio`, a config
     key declared in both GAN baselines' defaults but never actually
     used anywhere -- this is its first real use), with a `fraction` of
     the REAL portion's targets poisoned via attacks/poisoning.py's
     invert_continuous_targets (the regression-appropriate mechanism --
     label-flipping doesn't apply to a continuous forecast target).

     Deliberately does NOT retrain the full model end-to-end -- the
     existing per-epoch training loop is already a documented, severe
     bottleneck (see TODO.md's quantum-training-scale item); doubling it
     for this diagnostic would make that worse. Instead trains a small,
     freshly-initialized nn.Linear head (same shape as the model's own
     forecast_head) on half of X_test/y_test, evaluates on the other
     half -- cheap regardless of qubit count. Comparable ACROSS the
     noise-injection ablations specifically because the synthetic
     portion of the training mix comes from THIS model's own trained
     generator, whose noise_strength/entanglement setting differs per
     ablation -- a no-noise ablation and a noisy one get genuinely
     different synthetic features fed into this test, not just a
     relabeled identical number.

  2. MODEL INVERSION resistance: reuses membership_inference.py's
     real-vs-synthetic distinguishability test as-is. A generator whose
     synthetic output is easily distinguished from real data (high
     membership-inference accuracy) is, by the same logic, leaking more
     about the real training distribution's specific structure -- the
     standard motivation for treating that accuracy as a model-
     inversion/privacy-leakage proxy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..attacks.poisoning import invert_continuous_targets
from .membership_inference import membership_inference_success


def _fit_linear_head(X: np.ndarray, y: np.ndarray, epochs: int = 20,
                      lr: float = 0.01, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    head = nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = head(X_t).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_t)
        loss.backward()
        optimizer.step()
    return head


def evaluate_poisoning_resistance(model, X_test: np.ndarray, y_test: np.ndarray,
                                   clean_rmse: float, seed: int = 0,
                                   fraction: float = 0.05, epochs: int = 20,
                                   max_synthetic: int = 5000) -> dict:
    """
    model: needs .config (dict, read for "synthetic_ratio") and, optionally,
    .generate_synthetic_data(n) -- if the model has neither (e.g. a plain
    classical baseline with no generator), this degrades gracefully to a
    pure-real-data poisoning test (synthetic_ratio treated as 0).
    clean_rmse: this model's own already-computed clean-data RMSE
    (self.results["rmse"]), used as the baseline for the degradation %.
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    n = len(X_test)
    half = n // 2
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_idx, eval_idx = perm[:half], perm[half:]
    X_train_diag, y_train_diag = X_test[train_idx], y_test[train_idx]
    X_eval_diag, y_eval_diag = X_test[eval_idx], y_test[eval_idx]

    synthetic_ratio = float(getattr(model, "config", {}).get("synthetic_ratio", 0.0) or 0.0)
    n_synth = int(len(X_train_diag) * synthetic_ratio)
    X_mixed = X_train_diag.copy()
    if n_synth > 0 and hasattr(model, "generate_synthetic_data"):
        synthetic = np.asarray(model.generate_synthetic_data(n_synth), dtype=np.float64)
        # Synthetic rows REPLACE the first n_synth rows' FEATURES only; the
        # real target at that position stays attached (standard GAN data-
        # augmentation scheme -- not a claim the generator predicts targets).
        X_mixed[:n_synth] = synthetic
    y_mixed = y_train_diag.copy()

    poison = invert_continuous_targets(y_mixed, fraction=fraction, seed=seed)
    y_poisoned = poison["y_poisoned"]

    head = _fit_linear_head(X_mixed, y_poisoned, epochs=epochs, seed=seed)
    with torch.no_grad():
        pred = head(torch.as_tensor(X_eval_diag, dtype=torch.float32)).squeeze(-1).numpy()
    poisoned_rmse = float(np.sqrt(np.mean((pred - y_eval_diag) ** 2)))

    degradation_pct = (
        (poisoned_rmse - clean_rmse) / clean_rmse * 100.0
        if clean_rmse not in (None, 0) and not np.isnan(clean_rmse) else float("nan")
    )

    result = {
        "poisoned_rmse": poisoned_rmse,
        "poisoning_rmse_degradation_pct": degradation_pct,
        "poisoning_fraction": fraction,
        "poisoning_synthetic_ratio_used": synthetic_ratio,
        "poisoning_n_poisoned": poison["n_poisoned"],
    }

    if hasattr(model, "generate_synthetic_data"):
        try:
            synth_for_mi = np.asarray(model.generate_synthetic_data(min(len(X_eval_diag), 5000)))  # MI uses <=5000 rows anyway
            mi = membership_inference_success(X_eval_diag, synth_for_mi, seed=seed)
            result["model_inversion_accuracy"] = mi["membership_inference_accuracy"]
            result["model_inversion_chance_level"] = mi["chance_level"]
        except Exception as exc:  # keep the rest of evaluate() working if this sub-check fails
            result["model_inversion_accuracy"] = float("nan")
            result["model_inversion_error"] = str(exc)
    else:
        result["model_inversion_accuracy"] = float("nan")

    return result
