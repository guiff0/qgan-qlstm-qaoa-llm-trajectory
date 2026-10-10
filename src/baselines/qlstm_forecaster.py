"""
QLSTMForecaster: a standalone quantum forecasting baseline.

This is the class that was missing entirely from the original code and
from this rebuild until now: the direct quantum-side counterpart to
ClassicalLSTM. Every other quantum-related baseline (QGANLLM) only ever
uses the quantum circuit to generate synthetic training data -- the
actual forecast at inference time is a plain classical linear layer
that never touches a qubit. That leaves an open question (documented in
QGANLLM._forecast_model) about whether "quantum overhead" should show
up in inference latency at all under that design.

This class settles that ambiguity for at least one clean baseline: here,
the quantum circuit genuinely is the forecaster. Real market features go
IN to the quantum circuit; a prediction comes OUT. No discriminator, no
adversarial training, no GAN loss -- just supervised regression, exactly
like ClassicalLSTM, so the two are comparable on equal footing except
for the one thing that differs: classical LSTM cell vs. quantum circuit.

======================================================================
IMPORTANT COMPARABILITY CAVEAT -- read before citing this baseline
against ClassicalLSTM
======================================================================
ClassicalLSTM (src/baselines/classical_lstm.py) was fixed to use a real
60-step sliding window (src/data/windowing.py) -- it sees 60 timesteps
of history per prediction. The quantum circuit here, like QLSTMGenerator,
encodes each qubit from ONE feature via RY/RZ rotation; there is no
window-of-60 equivalent in this encoding scheme (that would need a
fundamentally different amplitude/basis encoding, out of scope for this
rebuild). So QLSTMForecaster sees only the single most recent row of
features per prediction -- a single-timestep model, not a 60-step one.

That means a fair "does quantum help" comparison for a like-for-like
input horizon is QLSTMForecaster vs. Classical GAN-LLM's forecast_head
(also single-timestep) or vs. a plain single-timestep classical
feedforward net -- NOT directly vs. ClassicalLSTM's 60-step-window RMSE,
which has access to far more temporal context and would be expected to
win partly for that reason alone, independent of classical-vs-quantum.
Report which comparison you're making explicitly.
"""
from __future__ import annotations

from typing import Dict

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseForecastingModel
from ..attacks.adversarial import compute_attack_success_rate, compute_clean_asr
from ..evaluation.latency import measure_inference_latency
from ..evaluation.metrics import mae as mae_fn
from ..evaluation.metrics import rmse as rmse_fn
from ..evaluation.one_step_ahead import shift_for_one_step_ahead
from ..evaluation.poisoning_resistance import evaluate_poisoning_resistance
from ..evaluation.threat_detection import evaluate_threat_detection
from ..evaluation.resilience_suite import run_resilience_suite
from ..quantum.circuits import QLSTMGenerator
from ..quantum.tomography import entanglement_metrics
from ..utils.reproducibility import seeded_generator, set_all_seeds
from ..utils.progress import progress_bar, log_progress_milestone
from ..utils.gan_training import (
    config_fingerprint, checkpoint_file, save_checkpoint, load_checkpoint, epoch_row_indices,
)


class QLSTMForecaster(BaseForecastingModel):
    """
    THE QUANTUM CIRCUIT IS THE FORECASTER (no GAN, no discriminator).

    build() constructs:
      self.quantum_circuit  -- a QLSTMGenerator, reused unchanged so this
                                model uses the identical circuit
                                architecture (qubit count, entanglement
                                topology, noise strength) as QGANLLM's
                                generator, keeping any RQ4 quantum-vs-
                                quantum-usage comparison apples-to-apples.
      self.forecast_head    -- nn.Linear(n_features, 1), mapping the
                                circuit's output to a single scalar
                                prediction.

    forward(X): X (real features, NOT noise) -> quantum_circuit -> forecast_head -> prediction
    """

    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            "n_qubits": 20,
            "n_layers": 4,
            "n_features": 32,
            "entanglement": "ring",
            "noise_strength": 0.01,
            "quantum_device": "default.qubit",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "early_stopping_patience": 10,
            "max_batches_per_epoch": None,  # None = full pass over the training set
            "val_samples": None,            # None = whole validation split; else fixed-seed subsample
            "eval_samples": None,           # rows for RMSE/MAE/poisoning (None = whole test set)
            "attack_samples": None,         # rows for adversarial attacks + threat detection
            "resume": True,
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("QLSTM Forecaster", cfg)
        self.seed = seed

    def build(self):
        set_all_seeds(self.seed)
        self.quantum_circuit = QLSTMGenerator(
            n_qubits=self.config["n_qubits"],
            n_layers=self.config["n_layers"],
            n_features=self.config["n_features"],
            entanglement=self.config["entanglement"],
            noise_strength=self.config["noise_strength"],
            quantum_device=self.config["quantum_device"],
        )
        self.forecast_head = nn.Linear(self.config["n_features"], 1)
        self.optimizer = torch.optim.Adam(
            list(self.quantum_circuit.parameters()) + list(self.forecast_head.parameters()),
            lr=self.config["learning_rate"],
        )
        self.criterion = nn.MSELoss()

    def _forecast_model(self, X: torch.Tensor) -> torch.Tensor:
        """Real features -> quantum circuit -> forecast_head. This IS the
        quantum circuit being invoked at inference (unlike QGANLLM's
        equivalent) -- see the module docstring."""
        quantum_features = self.quantum_circuit(X)
        return self.forecast_head(quantum_features)

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        self.build()
        # Same-row target leakage fix -- see
        # src/evaluation/one_step_ahead.py's module docstring. This model
        # (unlike QGANLLM) actually uses X_val/y_val below for early
        # stopping, so both need to be shifted, not just train.
        X_train, y_train = shift_for_one_step_ahead(np.asarray(X_train), np.asarray(y_train))
        X_val, y_val = shift_for_one_step_ahead(np.asarray(X_val), np.asarray(y_val))

        log = run_logger.info if run_logger else (lambda *_a, **_k: None)
        batch_size = self.config["batch_size"]
        max_batches = self.config.get("max_batches_per_epoch")
        n_epochs = self.config["epochs"]
        patience_limit = self.config["early_stopping_patience"]
        fp = config_fingerprint(self.config, self.seed, "qlstm_forecaster")
        ckpt_path = checkpoint_file("qlstm_forecaster", fp)
        best_path = f"models/qlstm_forecaster_best_{fp}.pt"

        X_t = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
        y_t = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
        n_val = self.config.get("val_samples")
        if n_val and n_val < len(X_val):
            sel = np.sort(np.random.default_rng(self.seed).choice(len(X_val), int(n_val), replace=False))
            X_val, y_val = X_val[sel], y_val[sel]
        Xv = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))
        yv = torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.float32))

        best_val_loss, patience_counter, start_epoch, finished = float("inf"), 0, 0, False
        ckpt = load_checkpoint(ckpt_path, fp) if self.config.get("resume", True) else None
        if ckpt is not None:
            self.quantum_circuit.load_state_dict(ckpt["quantum_circuit"])
            self.forecast_head.load_state_dict(ckpt["forecast_head"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch, best_val_loss = int(ckpt["next_epoch"]), ckpt["best_val_loss"]
            patience_counter, finished = int(ckpt["patience"]), bool(ckpt["finished"])
            log(f"Resuming from {ckpt_path}: next_epoch={start_epoch}/{n_epochs}"
                + (" (training complete)" if finished else ""))

        with progress_bar(total=n_epochs, desc=f"QLSTM Forecaster ({self.config['n_qubits']}q) epochs",
                          unit="epoch") as epoch_bar:
            if start_epoch:
                epoch_bar.update(start_epoch)
            for epoch in range(start_epoch, 0 if finished else n_epochs):
                torch.manual_seed(self.seed * 1_000_003 + epoch)
                self.quantum_circuit.train()
                idx = epoch_row_indices(len(X_t), batch_size, max_batches, self.seed, epoch)
                total_batches = (len(idx) + batch_size - 1) // batch_size
                epoch_loss, n_batches = 0.0, 0
                batch_bar = progress_bar(total=total_batches, desc=f"  epoch {epoch} batches", unit="batch")
                for s_ in range(0, len(idx), batch_size):
                    X_batch, y_batch = X_t[idx[s_:s_ + batch_size]], y_t[idx[s_:s_ + batch_size]]
                    self.optimizer.zero_grad()
                    pred = self._forecast_model(X_batch)
                    loss = self.criterion(pred.squeeze(-1), y_batch)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                    batch_bar.update(1)
                    batch_bar.set_postfix(loss=f"{epoch_loss / n_batches:.3e}")
                    if run_logger:
                        log_progress_milestone(run_logger, f"TRAIN epoch {epoch}", n_batches, total_batches)
                batch_bar.close()
                train_loss = epoch_loss / max(n_batches, 1)

                self.quantum_circuit.eval()
                val_loss_total, n_val_batches = 0.0, 0
                with torch.no_grad():
                    for s_ in progress_bar(range(0, len(Xv), batch_size), total=-(-len(Xv) // batch_size),
                                           desc=f"  epoch {epoch} validation", unit="batch"):
                        val_pred = self._forecast_model(Xv[s_:s_ + batch_size])
                        val_loss_total += self.criterion(val_pred.squeeze(-1), yv[s_:s_ + batch_size]).item()
                        n_val_batches += 1
                val_loss = val_loss_total / max(n_val_batches, 1)

                if run_logger:
                    run_logger.log_epoch(epoch, train_loss=train_loss, val_loss=val_loss)
                epoch_bar.update(1)
                epoch_bar.set_postfix(train_loss=f"{train_loss:.3e}", val_loss=f"{val_loss:.3e}",
                                      best=f"{best_val_loss:.3e}", patience=patience_counter)

                stop = False
                if val_loss < best_val_loss:
                    best_val_loss, patience_counter = val_loss, 0
                    self.save_model(best_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        log(f"Early stopping at epoch {epoch}")
                        stop = True
                save_checkpoint(ckpt_path, {
                    "fingerprint": fp, "next_epoch": epoch + 1, "best_val_loss": best_val_loss,
                    "patience": patience_counter, "finished": stop or (epoch + 1 == n_epochs),
                    "quantum_circuit": self.quantum_circuit.state_dict(),
                    "forecast_head": self.forecast_head.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                })
                if stop:
                    break

        self.is_trained = True
        if os.path.isfile(best_path):
            self.load_model(best_path)

    def save_model(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {"quantum_circuit": self.quantum_circuit.state_dict(),
             "forecast_head": self.forecast_head.state_dict()},
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.quantum_circuit.load_state_dict(checkpoint["quantum_circuit"])
        self.forecast_head.load_state_dict(checkpoint["forecast_head"])
        self.is_trained = True

    def predict(self, X):
        """Chunked inference through the PQC, mirroring ClassicalLSTM.predict().

        BUG FIXED: this previously ran the ENTIRE test set through
        _forecast_model() in a single call. Unlike the classical models,
        every row here means a real per-qubit circuit evaluation
        (parameter-shift execution), so this was both the most memory-
        and time-intensive predict() in the project and had no way to
        show progress. Chunking makes memory O(batch) instead of
        O(n_test_rows) and gives predict() something to report progress on.
        """
        self.quantum_circuit.eval()
        bs = int(self.config.get("predict_batch_size", 512))
        n_rows = len(X)
        n_chunks = max(1, -(-n_rows // bs))
        X_t = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        outs = []
        with torch.no_grad():
            for s in progress_bar(range(0, n_rows, bs), total=n_chunks,
                                  desc=f"QLSTM Forecaster ({self.config['n_qubits']}q) predict", unit="chunk"):
                outs.append(self._forecast_model(X_t[s: s + bs]).numpy())
        return np.concatenate(outs, axis=0)

    def measure_latency(self, X_test, n_repeats: int = 100) -> dict:
        self.quantum_circuit.eval()
        return measure_inference_latency(self._forecast_model, X_test, n_repeats=n_repeats)

    def measure_entanglement(self, sample_input: np.ndarray) -> dict:
        """RQ5-relevant: unlike QGANLLM (whose entanglement is measured
        from the generator's noise-conditioned state), this measures
        entanglement of the state actually used to make a real
        prediction -- arguably a more direct test of H5's claim that
        entanglement structure explains forecasting/detection behavior,
        since here entanglement and prediction come from the same
        circuit invocation, not a separate generator network."""
        from ..quantum.circuits import apply_circuit_gates_only

        n_qubits = self.config["n_qubits"]
        weights = self.quantum_circuit.theta.detach().numpy()
        circuit_fn = apply_circuit_gates_only(
            np.asarray(sample_input[:n_qubits], dtype=np.float64), weights,
            n_qubits, self.config["n_layers"], self.config["entanglement"],
        )
        return entanglement_metrics(
            circuit_fn, weights, n_qubits,
            dev_name=self.config["quantum_device"],
        )

    def evaluate(self, X_test, y_test, attack_cfg: dict = None, last_input_prices=None, **kwargs):
        X_test, y_test = shift_for_one_step_ahead(np.asarray(X_test), np.asarray(y_test))
        if last_input_prices is not None:
            last_input_prices = np.asarray(last_input_prices)[:-1]  # keep row-aligned with the shift above
        # Every row costs a real 20-qubit circuit evaluation (~0.3 s forward,
        # ~2.5 s with input gradients), so evaluating all 744k test rows would
        # take days. Use a fixed-seed random subsample (rows are independent
        # after the one-step-ahead shift).
        n_eval = self.config.get("eval_samples")
        if n_eval and n_eval < len(X_test):
            sel = np.sort(np.random.default_rng(self.seed).choice(len(X_test), int(n_eval), replace=False))
            X_test, y_test = X_test[sel], y_test[sel]
            if last_input_prices is not None:
                last_input_prices = last_input_prices[sel]
        self.results_n_eval = len(X_test)
        n_att = int(self.config.get("attack_samples") or len(X_test))
        n_att = min(n_att, len(X_test))
        predictions = self.predict(X_test)
        y_test_arr = np.array(y_test).flatten()
        predictions_arr = np.array(predictions).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": "QLSTM Forecaster",
            "eval_n_samples": len(X_test),
            "attack_n_samples": n_att,
        }

        if attack_cfg is not None and last_input_prices is not None:
            self.quantum_circuit.eval()
            X_test_t = torch.tensor(X_test[:n_att], dtype=torch.float32)
            y_test_t = torch.tensor(y_test[:n_att], dtype=torch.float32)
            last_prices_t = torch.tensor(last_input_prices[:n_att], dtype=torch.float32)
            asr_report = compute_attack_success_rate(
                self._forecast_model, X_test_t, y_test_t, last_prices_t,
                attack_cfg, attacks=attack_cfg.get("attacks", ["fgsm", "pgd", "cw"]),
            )
            self.results["asr"] = asr_report["overall_asr"]
            self.results["asr_breakdown"] = asr_report
            self.results["asr_clean"] = compute_clean_asr(
                self._forecast_model, X_test_t, y_test_t, last_prices_t, attack_cfg,
            )

        if len(X_test) > 0:
            ent_report = self.measure_entanglement(np.asarray(X_test[0], dtype=np.float32))
            self.results["entanglement_entropy"] = ent_report["entanglement_entropy"]
            self.results["purity"] = ent_report["purity"]

        # H3 (poisoning resistance -- no generate_synthetic_data on this
        # pure forecaster, so this degrades gracefully to a pure-real-data
        # poisoning test; model_inversion_accuracy reports NaN, honestly
        # reflecting that it doesn't apply to a non-generative model) and
        # H5 (threat detection/FPR) -- see QGANLLM.evaluate for full
        # rationale, identical wiring here.
        self.results.update(evaluate_poisoning_resistance(
            self, np.asarray(X_test), np.asarray(y_test), self.results["rmse"], seed=self.seed,
        ))
        if attack_cfg is not None:
            self.quantum_circuit.eval()
            self.results.update(evaluate_threat_detection(
                self._forecast_model, X_test[:n_att], y_test[:n_att], attack_cfg, seed=self.seed,
                n_clean=attack_cfg.get("n_benign_samples", 1000),
            ))

        if len(X_test) > 0:
            X_test_arr = np.asarray(X_test)
            self.quantum_circuit.eval()
            self.results.update(run_resilience_suite(
                self._forecast_model, self.quantum_circuit.theta.detach().numpy(),
                X_test_arr[0], float(np.asarray(y_test).flatten()[0]), X_test_arr[:50],
                self.config["n_qubits"], self.config["n_layers"], self.config["entanglement"],
                dev_name=self.config["quantum_device"], seed=self.seed,
            ))

        return self.results
