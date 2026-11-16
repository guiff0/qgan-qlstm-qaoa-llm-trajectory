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
        cfg = {**default_config, **(config or {})}
        super().__init__("QAOA-Enhanced LLM", cfg)
        self.seed = seed
        self.selector_result = None
        self.selected_mask = None  # torch.BoolTensor over the FULL feature width, set in train()

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
        predictions = self.predict(X_test)
        y_test_arr = np.array(y_test).flatten()
        predictions_arr = np.array(predictions).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": self.name,
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
