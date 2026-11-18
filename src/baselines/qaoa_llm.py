"""
QAOA-Enhanced LLM baseline.

======================================================================
WHAT "QAOA-ENHANCED" MEANS HERE -- flagged explicitly, same as
qgan_llm.py flags its own open methodological question above
======================================================================

QAOA is a combinatorial-optimization algorithm (bitstrings in, bitstring
out); it has no native role in continuous price regression. This
baseline's design decision -- made explicit here rather than silently
assumed -- is: QAOA runs ONCE before training starts, as a feature
selector (see src/quantum/qaoa.py's QAOAFeatureSelector), choosing which
of the engineered features the forecast head is allowed to see. That
selector solves a QUBO that rewards importance (correlation with the
target) and penalizes redundant/over-budget feature sets, over a
classically-pre-filtered candidate subset (default: top 16 by
correlation) -- see qaoa.py's module docstring for why the full 32/54
-feature set is not simulable directly.

This is architecturally SIMPLER than Classical GAN-LLM / QGAN-LLM:
there is no generator, no discriminator, no adversarial loop, no
synthetic-data augmentation. The forecast head is the same nn.Linear
shape as the other two baselines' forecast heads (input dim = number
of QAOA-selected features, not the full feature count), so any
RMSE/ASR difference reflects the feature-selection step itself, not a
different-capacity forecaster.

Practically, this also means the quantum cost here is bounded and
front-loaded: QAOA's circuit runs `qaoa_steps` times total (config
default 150), independent of dataset size or number of training
epochs -- unlike QGAN-LLM's generator, which runs once per training
BATCH (tens of thousands of times per epoch). See the QAOAFeatureSelector
benchmark this was validated against before being wired in here.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseForecastingModel
from ..attacks.adversarial import compute_attack_success_rate
from ..evaluation.metrics import rmse as rmse_fn, mae as mae_fn
from ..evaluation.latency import measure_inference_latency
from ..quantum.qaoa import QAOAFeatureSelector, QAOAFeatureSelectorConfig
from ..utils.reproducibility import set_all_seeds, seeded_generator
from ..utils.progress import progress_bar, log_progress_milestone


class QAOALLM(BaseForecastingModel):
    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            # --- QAOA feature-selection stage (src/quantum/qaoa.py) ---
            "n_candidates": 16,
            "target_k": 8,
            "qaoa_layers": 3,
            "budget_penalty": 2.0,
            "redundancy_penalty": 1.0,
            "redundancy_edge_threshold": 0.3,
            "qaoa_steps": 150,
            "qaoa_learning_rate": 0.1,
            "quantum_device": "lightning.qubit",
            # --- Forecast-head training (classical, post-selection) ---
            "n_features": 32,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 30,
        }
        cfg = {**default_config, "ridge": "auto", **(config or {})}
        # feature_frame default: "diff" for delta forecasting, "level" for level forecasting
        if "feature_frame" not in cfg:
            cfg["feature_frame"] = "diff" if cfg.get("forecast_target", "delta") == "delta" else "level"
        # Validate feature_frame + forecast_target combination
        if cfg.get("feature_frame", "diff") == "diff" and cfg.get("forecast_target", "delta") == "level":
            raise ValueError(
                "feature_frame='diff' needs forecast_target: delta -- "
                "diff inputs predict the CHANGE, not the level."
            )
        super().__init__("QAOA-Enhanced LLM", cfg)
        self.seed = seed
        self.selector_result = None
        self.selected_mask = None  # torch.BoolTensor over the FULL feature width, set in train()
        self.target_frame: str = cfg.get("forecast_target", "delta")
        self.feature_frame: str = cfg.get("feature_frame", "diff")
        self.head_fit: str = cfg.get("head_fit", "ridge")
        self.target_scale: float = 1.0
        self.head_info: dict = {}
        self._head_is_no_change: bool = False

    def build(self, n_selected: int):
        set_all_seeds(self.seed)
        self.forecast_head = nn.Linear(n_selected, 1)
        self.forecast_optimizer = torch.optim.Adam(self.forecast_head.parameters(), lr=self.config["learning_rate"])
        self.mse_loss = nn.MSELoss()

    def _select_features(self, X_train, y_train, run_logger=None):
        qaoa_cfg = QAOAFeatureSelectorConfig(
            n_candidates=self.config["n_candidates"],
            target_k=self.config["target_k"],
            n_layers=self.config["qaoa_layers"],
            budget_penalty=self.config["budget_penalty"],
            redundancy_penalty=self.config["redundancy_penalty"],
            redundancy_edge_threshold=self.config["redundancy_edge_threshold"],
            quantum_device=self.config["quantum_device"],
            n_steps=self.config["qaoa_steps"],
            learning_rate=self.config["qaoa_learning_rate"],
            seed=self.seed,
        )
        if run_logger:
            run_logger.info(
                f"QAOA feature selection: {qaoa_cfg.n_candidates} candidate qubits, "
                f"{qaoa_cfg.n_layers} layers, {qaoa_cfg.n_steps} optimizer steps, "
                f"device={qaoa_cfg.quantum_device}"
            )
        result = QAOAFeatureSelector(qaoa_cfg).fit(X_train, y_train)
        if run_logger:
            run_logger.info(
                f"QAOA selected {int(result.selected_mask.sum())} of {len(result.selected_mask)} "
                f"features (indices {np.where(result.selected_mask)[0].tolist()}); "
                f"final QUBO objective={result.qubo_objective:.4f}"
            )
        return result

    def _apply_feature_frame(self, X, trim_for_delta: bool = False):
        """Apply feature_frame transform.
        'diff': (X[1:]-X[:-1])[:-1] giving n-2 rows.
        'level': return X unchanged, or X[:-1] when trim_for_delta=True (aligns with np.diff(y))."""
        X = np.asarray(X, dtype=np.float64)
        if self.feature_frame == "diff":
            return (X[1:] - X[:-1])[:-1]   # n-2 rows
        if trim_for_delta:
            return X[:-1]   # n-1 rows, aligned with np.diff(y)
        return X

    def _solve_ridge(self, X_aug, y, lam: float) -> np.ndarray:
        """Closed-form ridge on augmented feature matrix [X | 1] (bias included).
        Returns coefficient vector w of length n_cols+1."""
        A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
        return np.linalg.solve(A, X_aug.T @ y)

    def _fit_ridge_head(self, X_sel: np.ndarray, y: np.ndarray, lam: float = 1.0):
        """Fit forecast_head (nn.Linear) via closed-form ridge. X_sel already selected."""
        n = len(y)
        X_aug = np.c_[X_sel.astype(np.float64), np.ones(n)]
        w = self._solve_ridge(X_aug, y.astype(np.float64), lam)
        with torch.no_grad():
            self.forecast_head.weight.copy_(torch.tensor(w[:-1], dtype=torch.float32).unsqueeze(0))
            self.forecast_head.bias.copy_(torch.tensor([w[-1]], dtype=torch.float32))

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        # Delta-frame: targets = next-bar changes, standardized
        if self.target_frame == "delta":
            y_arr = np.asarray(y_train, dtype=np.float64)
            diffs = np.diff(y_arr)
            self.target_scale = float(np.std(diffs)) if len(diffs) > 0 else 1.0
            if self.target_scale == 0.0:
                self.target_scale = 1.0
            # For feature_frame='diff': X_train input to selector = (X[1:]-X[:-1])[:-1]  (n-2 rows)
            # targets = diff(y)[1:] / scale = y[2:]-y[1:-1] / scale  aligned with n-2 rows
            # For feature_frame='level': X_train input = X_train[:-1], targets = diff(y) / scale
            X_sel_input = self._apply_feature_frame(X_train, trim_for_delta=(self.feature_frame == "level"))
            if self.feature_frame == "diff":
                y_sel_input = diffs[1:] / self.target_scale          # y[2:]-y[1:-1]
            else:
                y_sel_input = diffs / self.target_scale               # y[1:]-y[:-1], n-1 rows
            if X_val is not None:
                y_val_arr = np.asarray(y_val, dtype=np.float64)
                X_val_sel = self._apply_feature_frame(X_val, trim_for_delta=(self.feature_frame == "level"))
                if self.feature_frame == "diff":
                    y_val_sel = np.diff(y_val_arr)[1:] / self.target_scale
                else:
                    y_val_sel = np.diff(y_val_arr) / self.target_scale
            else:
                X_val_sel, y_val_sel = None, None
        else:
            self.target_scale = 1.0
            X_sel_input = np.asarray(X_train, dtype=np.float64)
            y_sel_input = np.asarray(y_train, dtype=np.float64)
            X_val_sel, y_val_sel = (np.asarray(X_val, dtype=np.float64) if X_val is not None else None,
                                    np.asarray(y_val, dtype=np.float64) if y_val is not None else None)

        self.selector_result = self._select_features(X_sel_input, y_sel_input, run_logger=run_logger)
        self.selected_mask = torch.tensor(self.selector_result.selected_mask, dtype=torch.bool)

        n_selected = int(self.selector_result.selected_mask.sum())
        if n_selected == 0:
            # A degenerate but mathematically valid QUBO optimum if
            # budget_penalty/redundancy_penalty dominate the importance
            # term for this dataset -- fail loudly rather than silently
            # training a zero-input linear layer that predicts a constant.
            raise RuntimeError(
                "QAOA feature selection returned zero selected features. "
                "Check qaoa_llm.budget_penalty / redundancy_penalty / target_k "
                "in config -- see src/quantum/qaoa.py's _build_qubo docstring "
                "for what each controls."
            )
        self.build(n_selected)

        # Get selected columns for fitting
        mask_np = self.selector_result.selected_mask
        X_sel_np = X_sel_input[:, mask_np]

        head_fit = self.head_fit
        ridge_cfg = self.config.get("ridge", "auto")

        if head_fit == "ridge" and ridge_cfg != "auto":
            # Closed-form ridge with a fixed penalty
            lam = 1.0 if isinstance(ridge_cfg, str) else float(ridge_cfg)
            self._fit_ridge_head(X_sel_np, y_sel_input, lam=lam)
            self.head_info = {"mode": "fixed", "penalty": lam, "no_change": False}
            self._head_is_no_change = False
        elif head_fit == "sgd":
            # Standard SGD training loop (the old path)
            train_ds = TensorDataset(
                torch.tensor(X_sel_np, dtype=torch.float32),
                torch.tensor(y_sel_input, dtype=torch.float32),
            )
            train_loader = DataLoader(
                train_ds, batch_size=self.config["batch_size"], shuffle=True,
                generator=seeded_generator(self.seed),
            )
            n_epochs = self.config["epochs"]
            total_batches = len(train_loader)
            with progress_bar(total=n_epochs, desc="QAOA-Enhanced LLM epochs", unit="epoch") as epoch_bar:
                for epoch in range(n_epochs):
                    f_loss_total, n_batches = 0.0, 0
                    batch_bar = progress_bar(total=total_batches, desc=f"  epoch {epoch} batches", unit="batch")
                    for X_batch, y_batch in train_loader:
                        f_loss_total += self._train_forecast_head_step_sel(X_batch, y_batch)
                        n_batches += 1
                        batch_bar.update(1)
                        batch_bar.set_postfix(fcast=f"{f_loss_total / max(n_batches, 1):.3e}")
                        if run_logger:
                            log_progress_milestone(run_logger, f"TRAIN epoch {epoch}", n_batches, total_batches)
                    batch_bar.close()
                    if run_logger:
                        run_logger.log_epoch(epoch, forecast_loss=f_loss_total / max(n_batches, 1))
                    epoch_bar.update(1)
                    epoch_bar.set_postfix(fcast=f"{f_loss_total / max(n_batches, 1):.3e}")
            self.head_info = {"mode": "sgd", "no_change": False}
            self._head_is_no_change = False
        else:
            # head_fit == "ridge" with ridge == "auto": search lambdas on validation set
            # If no validation data, use fixed penalty 1.0
            LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, "no_change"]
            if X_val_sel is None or len(X_val_sel) == 0:
                self._fit_ridge_head(X_sel_np, y_sel_input, lam=1.0)
                self.head_info = {"mode": "fixed", "penalty": 1.0, "no_change": False}
                self._head_is_no_change = False
            else:
                X_val_sel_np = X_val_sel[:, mask_np]
                # Persistence = no-change: predict 0 (since targets are changes)
                y_val_arr_np = np.asarray(y_val_sel, dtype=np.float64)
                persistence_val_rmse = float(np.sqrt(np.mean(y_val_arr_np ** 2)))
                best_lam, best_val_rmse = None, float("inf")
                candidates_rmse = {}
                for lam in LAMBDAS:
                    if lam == "no_change":
                        # predict zero = persistence
                        preds_val = np.zeros(len(y_val_arr_np))
                        val_rmse = float(np.sqrt(np.mean(y_val_arr_np ** 2)))
                    else:
                        self._fit_ridge_head(X_sel_np, y_sel_input, lam=float(lam))
                        with torch.no_grad():
                            preds_val = self.forecast_head(
                                torch.tensor(X_val_sel_np, dtype=torch.float32)
                            ).numpy().flatten()
                        val_rmse = float(np.sqrt(np.mean((preds_val - y_val_arr_np) ** 2)))
                    ratio = val_rmse / persistence_val_rmse if persistence_val_rmse > 0 else float("inf")
                    candidates_rmse[str(lam)] = ratio
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_lam = lam
                overall_ratio = best_val_rmse / persistence_val_rmse if persistence_val_rmse > 0 else float("inf")
                if best_lam == "no_change" or overall_ratio > 1.0:
                    # No head beat persistence: use no-change
                    with torch.no_grad():
                        self.forecast_head.weight.zero_()
                        self.forecast_head.bias.zero_()
                    self._head_is_no_change = True
                    self.head_info = {
                        "mode": "auto", "val_rmse_over_no_change": min(overall_ratio, 1.0),
                        "no_change": True, "candidates_val_rmse_over_no_change": candidates_rmse,
                    }
                else:
                    self._fit_ridge_head(X_sel_np, y_sel_input, lam=float(best_lam))
                    self._head_is_no_change = False
                    self.head_info = {
                        "mode": "auto", "val_rmse_over_no_change": overall_ratio,
                        "no_change": False, "best_lambda": best_lam,
                        "candidates_val_rmse_over_no_change": candidates_rmse,
                    }

        self.is_trained = True

    def _train_forecast_head_step(self, X_batch, y_batch):
        X_sel = X_batch[:, self.selected_mask]
        pred = self.forecast_head(X_sel)
        loss = self.mse_loss(pred.squeeze(-1), y_batch)
        self.forecast_optimizer.zero_grad()
        loss.backward()
        self.forecast_optimizer.step()
        return loss.item()

    def _train_forecast_head_step_sel(self, X_batch_sel, y_batch):
        """Like _train_forecast_head_step but X is already column-selected."""
        pred = self.forecast_head(X_batch_sel)
        loss = self.mse_loss(pred.squeeze(-1), y_batch)
        self.forecast_optimizer.zero_grad()
        loss.backward()
        self.forecast_optimizer.step()
        return loss.item()

    def _forecast_model(self, X: torch.Tensor) -> torch.Tensor:
        """Takes FULL-width features -- same convention as Classical GAN-LLM's
        and QGAN-LLM's _forecast_model, and required so
        compute_attack_success_rate can perturb the full feature vector --
        and applies the QAOA-selected mask internally before the linear
        layer. Gradient w.r.t. non-selected columns is exactly zero, which
        is correct: the model genuinely does not use them, not a bug."""
        X_sel = X[:, self.selected_mask]
        return self.forecast_head(X_sel)

    def measure_latency(self, X_test, n_repeats: int = 100) -> dict:
        self.forecast_head.eval()
        return measure_inference_latency(self._forecast_model, X_test, n_repeats=n_repeats)

    def predict(self, X):
        self.forecast_head.eval()
        X_t = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        with torch.no_grad():
            pred = self._forecast_model(X_t)
        return pred.numpy() if not torch.is_tensor(X) else pred

    def evaluate(self, X_test, y_test, attack_cfg: Dict = None, last_input_prices=None, **kwargs):
        y_test_arr = np.asarray(y_test, dtype=np.float64).flatten()
        n_sel = int(self.selector_result.selected_mask.sum())

        if self.target_frame == "delta":
            # Apply feature_frame transform to test features
            X_eval = self._apply_feature_frame(np.asarray(X_test, dtype=np.float32), trim_for_delta=(self.feature_frame == "level"))
            # Align targets: actual bar changes
            diffs_test = np.diff(y_test_arr)
            if self.feature_frame == "diff":
                actual_changes = diffs_test[1:]          # y[j+2]-y[j+1] aligned with (X[j+1]-X[j])
            else:
                actual_changes = diffs_test               # y[j+1]-y[j] aligned with X[j]
            # Persistence: predict zero change
            persistence_rmse = float(np.sqrt(np.mean(actual_changes ** 2)))
            # Predict: head outputs scaled change
            self.forecast_head.eval()
            X_t = torch.tensor(X_eval[:, self.selector_result.selected_mask], dtype=torch.float32)
            with torch.no_grad():
                preds_scaled = self.forecast_head(X_t).numpy().flatten()
            preds_unscaled = preds_scaled * self.target_scale
            n = min(len(preds_unscaled), len(actual_changes))
            self.results = {
                "rmse": float(np.sqrt(np.mean((preds_unscaled[:n] - actual_changes[:n]) ** 2))),
                "mae": float(np.mean(np.abs(preds_unscaled[:n] - actual_changes[:n]))),
                "model_type": self.name,
                "target_frame": self.target_frame,
                "feature_frame": self.feature_frame,
                "persistence_rmse": persistence_rmse,
                "qaoa_n_selected": n_sel,
                "qaoa_selected_features": np.where(self.selector_result.selected_mask)[0].tolist(),
                "qaoa_qubo_objective": self.selector_result.qubo_objective,
                "head_is_no_change": self._head_is_no_change,
            }
            if attack_cfg is not None and last_input_prices is not None:
                self.forecast_head.eval()
                # For ASR attack, use aligned test features (full width, then mask inside _forecast_model)
                # We need X_test with full columns; _forecast_model applies selected_mask internally.
                # Align: for diff frame, X_eval has n-2 rows; align attack to that.
                X_atk = np.asarray(X_test, dtype=np.float32)
                if self.feature_frame == "diff":
                    X_atk_aligned = (X_atk[1:] - X_atk[:-1])[:-1]
                else:
                    X_atk_aligned = X_atk[:-1]
                n_att = min(len(X_atk_aligned), len(actual_changes))
                # Reconstruct full-width X for _forecast_model (which re-applies mask)
                # We pass X_atk_aligned directly; _forecast_model uses selected_mask on full-width input
                # For diff frame the full-width X_atk_aligned is already the diff-ed features
                X_att_t = torch.tensor(X_atk_aligned[:n_att], dtype=torch.float32)
                y_att_t = torch.tensor(actual_changes[:n_att] / self.target_scale, dtype=torch.float32)
                lp_t = torch.tensor(np.asarray(last_input_prices, dtype=np.float32).flatten()[:n_att], dtype=torch.float32)
                asr_report = compute_attack_success_rate(
                    self._forecast_model, X_att_t, y_att_t, lp_t,
                    attack_cfg, attacks=attack_cfg.get("attacks", ["fgsm", "pgd", "cw"]),
                )
                self.results["asr"] = asr_report["overall_asr"]
                self.results["asr_breakdown"] = asr_report
        else:
            predictions = self.predict(X_test)
            predictions_arr = np.array(predictions).flatten()
            self.results = {
                "rmse": rmse_fn(y_test_arr, predictions_arr),
                "mae": mae_fn(y_test_arr, predictions_arr),
                "model_type": self.name,
                "target_frame": self.target_frame,
                "qaoa_n_selected": n_sel,
                "qaoa_selected_features": np.where(self.selector_result.selected_mask)[0].tolist(),
                "qaoa_qubo_objective": self.selector_result.qubo_objective,
                "head_is_no_change": self._head_is_no_change,
            }
            if attack_cfg is not None and last_input_prices is not None:
                self.forecast_head.eval()
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                y_test_t = torch.tensor(y_test, dtype=torch.float32)
                last_prices_t = torch.tensor(last_input_prices, dtype=torch.float32)
                asr_report = compute_attack_success_rate(
                    self._forecast_model, X_test_t, y_test_t, last_prices_t,
                    attack_cfg, attacks=attack_cfg.get("attacks", ["fgsm", "pgd", "cw"]),
                )
                self.results["asr"] = asr_report["overall_asr"]
                self.results["asr_breakdown"] = asr_report

        return self.results
