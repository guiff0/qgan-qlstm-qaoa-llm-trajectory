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
from ..evaluation.one_step_ahead import make_target, resolve_frame
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
            # "delta" (default since 14 Nov 2026): features at row k -> the next-close CHANGE y[k+1] - y[k]; QAOA selects
            # against that same target and the price forecast is last close + change, so a head that learns nothing
            # scores exactly persistence. "level" = the October behaviour, unchanged (features at row k -> close at row k).
            "forecast_target": "delta",
            # Inputs (14 Nov 2026). "diff": each row is the CHANGE of the features since the previous bar, X[k] - X[k-1].
            # Reason, measured on train/validation only: a linear next-bar model on the raw (level) components is worse
            # than no-change on validation (1.0004); on first differences it is not (0.9999). None = "diff" in the delta
            # frame, "level" in the level frame (the October behaviour).
            "feature_frame": None,
            # Head. "ridge": closed-form ridge regression on the selected features -- the same kind of head the GAN arms use,
            # exact, and seconds instead of 30 SGD epochs at lr 3e-4. "sgd": the October loop. None = "ridge" in the delta
            # frame, "sgd" in the level frame.
            "head_fit": None,
            # Ridge penalty. A number = that penalty (on features standardised by their training SD). "auto" = chosen on the
            # VALIDATION rows from RIDGE_GRID (penalty = n_train x value), with "no head at all" (the no-change forecast) as
            # one of the candidates: when no penalty beats no-change on validation the head is zero and the run says so.
            # The test rows are never used for this.
            "ridge": "auto",
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("QAOA-Enhanced LLM", cfg)
        self.seed = seed
        self.target_frame = resolve_frame(cfg)
        self.target_scale = 1.0           # SD of the training next-close changes (delta frame); 1 in the level frame
        self.selector_result = None
        self.selected_mask = None  # torch.BoolTensor over the FULL feature width, set in train()
        self.feature_frame = cfg.get("feature_frame") or ("diff" if self.target_frame == "delta" else "level")
        self.head_fit = cfg.get("head_fit") or ("ridge" if self.target_frame == "delta" else "sgd")
        if self.feature_frame not in ("diff", "level") or self.head_fit not in ("ridge", "sgd"):
            raise ValueError("qaoa_llm.feature_frame must be 'diff' or 'level' and head_fit 'ridge' or 'sgd'")
        if self.feature_frame == "diff" and self.target_frame != "delta":
            raise ValueError("qaoa_llm.feature_frame: 'diff' needs forecast_target: delta")

    def _frame(self, X, y):
        """(inputs, next-close change, next close, last close) in this model's frame, one row per forecast.
        diff : inputs[j] = X[j+1] - X[j]  ->  change y[j+2] - y[j+1]   (both known at bar j+1; nothing from bar j+2)
        level: inputs[j] = X[j]           ->  change y[j+1] - y[j]"""
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)[:, 0]
        if self.feature_frame == "diff":
            X, y = np.diff(X, axis=0), y[1:]
        t = make_target(X, y, "delta")
        return t.X, t.target, t.y_next, t.last

    RIDGE_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)      # x n_train; searched only when ridge == "auto"

    def _fit_ridge_head(self, X_sel: np.ndarray, target: np.ndarray, X_val_sel=None, target_val=None):
        """Closed-form ridge with an unpenalised intercept on features standardised by their training SD, accumulated in
        float64 chunks (no second copy of X). ridge == "auto": the penalty is the candidate with the lowest VALIDATION
        error, and the no-change forecast (zero head) is a candidate; ties go to the stronger penalty."""
        k = X_sel.shape[1]
        G, b, s, n, ts = np.zeros((k, k)), np.zeros(k), np.zeros(k), 0, 0.0
        for i in range(0, len(X_sel), 500_000):
            A = np.asarray(X_sel[i:i + 500_000], dtype=np.float64)
            t = np.asarray(target[i:i + 500_000], dtype=np.float64)
            G += A.T @ A; b += A.T @ t; s += A.sum(0); n += len(A); ts += float(t.sum())
        mu, tm = s / n, ts / n
        Gc, bc = G - n * np.outer(mu, mu), b - n * mu * tm
        sd = np.sqrt(np.clip(np.diag(Gc) / n, 1e-300, None))
        Gs, bs = Gc / np.outer(sd, sd), bc / sd                       # standardised features

        def solve(pen):
            w = np.linalg.solve(Gs + pen * np.eye(k), bs) / sd        # back on the raw feature scale
            return w, tm - float(mu @ w)
        ridge = self.config.get("ridge", "auto")
        info = {"mode": "fixed", "penalty": None, "no_change": False}
        if ridge == "auto" and X_val_sel is not None and len(X_val_sel) > 1:
            V, tv = np.asarray(X_val_sel, dtype=np.float64), np.asarray(target_val, dtype=np.float64)
            base = float(np.sqrt(np.mean(tv ** 2)))                    # the no-change forecast on validation
            best = (base, np.zeros(k), 0.0, None)                      # (val rmse, w, intercept, grid value)
            tried = {"no_change": 1.0}
            for a in sorted(self.RIDGE_GRID, reverse=True):            # strongest first: a tie keeps the stronger penalty
                w, c = solve(a * n if a > 0 else 1.0)
                r = float(np.sqrt(np.mean((V @ w + c - tv) ** 2)))
                tried[f"{a:g}"] = r / base
                if r < best[0]:
                    best = (r, w, c, a)
            _, w, c, a = best
            info = {"mode": "auto (validation)", "penalty": a, "no_change": a is None, "val_rmse_over_no_change": best[0] / base,
                    "candidates_val_rmse_over_no_change": tried}
        else:
            pen = 1.0 if ridge == "auto" else float(ridge)             # "auto" without validation rows: the old fixed 1.0
            w, c = solve(pen)
            info["penalty"] = pen
        self.head_info = info
        with torch.no_grad():
            self.forecast_head.weight.copy_(torch.tensor(w, dtype=torch.float32).reshape(1, -1))
            self.forecast_head.bias.copy_(torch.tensor([c], dtype=torch.float32))

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

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        if self.target_frame == "delta":
            X_train, _target, _yn, _last = self._frame(X_train, y_train)      # differenced in float64
            self.target_scale = float(np.std(_target)) if float(np.std(_target)) > 0 else 1.0
            # QAOA scores candidates by |correlation with the target|: it must be the target the head is trained on.
            y_train = _target / self.target_scale
            if run_logger:
                run_logger.info(f"QAOA-Enhanced LLM: target = next-close change (scale {self.target_scale:.3e}); inputs = "
                                f"{self.feature_frame}; head = {self.head_fit}; selection and head use the same rows")
        self.selector_result = self._select_features(X_train, y_train, run_logger=run_logger)
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
        if self.head_fit == "ridge":
            Xv = tv = None
            if X_val is not None and y_val is not None and len(X_val) > 3 and self.target_frame == "delta":
                Xv, tv, _a, _b = self._frame(X_val, y_val)
                Xv, tv = Xv[:, self.selector_result.selected_mask], tv / self.target_scale
            self._fit_ridge_head(np.asarray(X_train)[:, self.selector_result.selected_mask], y_train, Xv, tv)
            if run_logger:
                run_logger.info(f"closed-form ridge head fitted on {len(X_train):,} rows x {n_selected} selected features: "
                                f"{self.head_info}")
            self.is_trained = True
            return

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
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
                    f_loss_total += self._train_forecast_head_step(X_batch, y_batch)
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

        self.is_trained = True

    def _train_forecast_head_step(self, X_batch, y_batch):
        X_sel = X_batch[:, self.selected_mask]
        pred = self.forecast_head(X_sel)
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
        return self.forecast_head(X_sel) * self.target_scale      # price units: next-close change (delta) or level

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
        extra = {}
        if self.target_frame == "delta":
            X_test, y_test, _yn, _last = self._frame(X_test, y_test)   # next-close CHANGE: this RMSE equals the price RMSE
            last_input_prices = np.zeros(len(y_test))        # a directional flip is a sign change of the forecast change
            extra["persistence_rmse"] = float(np.sqrt(np.mean((_yn - _last) ** 2)))              # same rows as `rmse`
            extra["feature_frame"], extra["head_fit"] = self.feature_frame, self.head_fit
            extra["qaoa_n_selected"] = int(self.selector_result.selected_mask.sum())
            hi = getattr(self, "head_info", None) or {}
            extra["head_ridge_mode"], extra["head_ridge_penalty"] = hi.get("mode"), hi.get("penalty")
            extra["head_is_no_change"] = bool(hi.get("no_change", False))     # True = validation kept the no-change forecast
            extra["head_val_rmse_over_no_change"] = hi.get("val_rmse_over_no_change")
        predictions = self.predict(X_test)
        y_test_arr = np.array(y_test, dtype=np.float64).flatten()
        predictions_arr = np.array(predictions, dtype=np.float64).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": self.name,
            "target_frame": self.target_frame,
            **extra,
            "qaoa_selected_features": np.where(self.selector_result.selected_mask)[0].tolist(),
            "qaoa_qubo_objective": self.selector_result.qubo_objective,
        }

        if attack_cfg is not None and last_input_prices is not None:
            self.forecast_head.eval()  # compute_attack_success_rate no longer does this itself
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
