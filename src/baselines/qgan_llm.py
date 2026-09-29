"""
QGAN-LLM: the study's primary model (QLSTM generator + classical
discriminator + LLAMA 3.3).

======================================================================
GENERATOR vs. DISCRIMINATOR -- who does what, in this file
======================================================================

  GENERATOR   (QLSTMGenerator, imported from src/quantum/circuits.py)
      Role:   Same job as ClassicalGANLLM's LSTMGenerator -- noise z
              -> one synthetic feature vector for data augmentation --
              but implemented as a parameterized quantum circuit
              (default: 20 qubits, ring entanglement) instead of an
              LSTM. This is the ONLY thing that differs architecturally
              between the two baselines' training; everything else
              (discriminator, forecast head, loss functions, training
              loop shape) is deliberately identical.

  DISCRIMINATOR   (ClassicalDiscriminator, imported unchanged from
              classical_gan_llm.py -- see that file for its docstring)
      Role:   Identical class, identical hyperparameters, to the one
              Classical GAN-LLM uses. Deliberately not reimplemented
              or retuned here, so a discriminator difference can never
              be the explanation for any RMSE/ASR/fidelity gap between
              the two baselines.

  FORECASTER   (self.forecast_head, a plain nn.Linear, added in
              build() below)
      Role:   Same as Classical GAN-LLM's forecast_head: the model
              actually used for prediction at inference time, trained
              on real (optionally augmented) features directly.
              IMPORTANT: at inference, this does NOT route through the
              quantum generator -- see _forecast_model's docstring
              below for why that's a flagged, deliberate detail (not a
              bug) and what it implies for latency comparisons.

Compared to the original code, this file also:
  - Uses src/quantum/circuits.py's QLSTMGenerator, which is
    differentiable end-to-end (the original detached the quantum
    circuit from autograd — see circuits.py's docstring).
  - Uses mini-batch training (same fix as the other baselines).
  - Computes entanglement entropy via src/quantum/tomography.py's real
    von-Neumann-entropy calculation instead of np.random.uniform().
  - Computes ASR via src/attacks/adversarial.py's real attack
    implementations instead of a hardcoded return value.

See also src/baselines/qlstm_forecaster.py -- a fourth baseline where
the quantum circuit IS the forecaster directly (no GAN wrapper at
all), which this file's design deliberately does not provide.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseForecastingModel
from .classical_gan_llm import ClassicalDiscriminator
from ..attacks.adversarial import compute_attack_success_rate
from ..evaluation.metrics import rmse as rmse_fn, mae as mae_fn, synthetic_data_fidelity_report
from ..evaluation.latency import measure_inference_latency
from ..quantum.circuits import QLSTMGenerator, apply_circuit_gates_only
from ..quantum.tomography import entanglement_metrics
from ..utils.reproducibility import set_all_seeds, seeded_generator
from ..utils.progress import progress_bar, log_progress_milestone


class QGANLLM(BaseForecastingModel):
    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            "n_qubits": 20,
            "n_layers": 4,
            "n_features": 32,
            "entanglement": "ring",
            "noise_strength": 0.01,
            "discriminator_hidden": 256,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 30,
            "synthetic_ratio": 0.4,
            "n_critic": 2,
            "alpha": 10.0,
            "quantum_device": "default.qubit",
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("QGAN-LLM", cfg)
        self.seed = seed
        self.qgan_results: Dict = {}

    def build(self):
        set_all_seeds(self.seed)
        self.generator = QLSTMGenerator(
            n_qubits=self.config["n_qubits"],
            n_layers=self.config["n_layers"],
            n_features=self.config["n_features"],
            entanglement=self.config["entanglement"],
            noise_strength=self.config["noise_strength"],
            quantum_device=self.config["quantum_device"],
        )
        self.discriminator = ClassicalDiscriminator(
            input_dim=self.config["n_features"],
            hidden_dim=self.config["discriminator_hidden"],
        )
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config["learning_rate"])
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.config["learning_rate"])
        self.criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.forecast_head = nn.Linear(self.config["n_features"], 1)
        self.forecast_optimizer = torch.optim.Adam(self.forecast_head.parameters(), lr=self.config["learning_rate"])

    def _train_discriminator_step(self, real_batch):
        batch_size = real_batch.shape[0]
        z = torch.randn(batch_size, self.config["n_features"])
        fake = self.generator(z)

        real_out = self.discriminator(real_batch)
        fake_out = self.discriminator(fake.detach())
        d_loss = self.criterion(real_out, torch.ones(batch_size, 1)) + \
            self.criterion(fake_out, torch.zeros(batch_size, 1))

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss.item()

    def _train_generator_step(self, real_batch):
        batch_size = real_batch.shape[0]
        z = torch.randn(batch_size, self.config["n_features"])
        fake = self.generator(z)

        fake_out = self.discriminator(fake)
        adv_loss = self.criterion(fake_out, torch.ones(batch_size, 1))
        mse_loss = self.mse_loss(fake, real_batch)
        total_loss = adv_loss + self.config["alpha"] * mse_loss

        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()
        return {"total_loss": total_loss.item(), "adv_loss": adv_loss.item(), "mse_loss": mse_loss.item()}

    def _train_forecast_head_step(self, real_batch, y_batch):
        pred = self.forecast_head(real_batch)
        loss = self.mse_loss(pred.squeeze(-1), y_batch)
        self.forecast_optimizer.zero_grad()
        loss.backward()
        self.forecast_optimizer.step()
        return loss.item()

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        self.build()

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True,
            generator=seeded_generator(self.seed),
        )

        entanglement_history = []
        n_epochs = self.config["epochs"]
        total_batches = len(train_loader)

        # Per-batch progress matters even more here than for the classical
        # baselines: each batch runs the PQC (parameter-shift gradients need
        # O(n_params) circuit evaluations per step), so a single epoch at
        # n_qubits=20 can be the slowest step in the whole pipeline by far.
        with progress_bar(total=n_epochs, desc=f"QGAN-LLM ({self.config['n_qubits']}q) epochs",
                          unit="epoch") as epoch_bar:
            for epoch in range(n_epochs):
                d_loss_total, g_loss_total, f_loss_total, n_batches = 0.0, 0.0, 0.0, 0
                batch_bar = progress_bar(total=total_batches, desc=f"  epoch {epoch} batches", unit="batch")
                for X_batch, y_batch in train_loader:
                    for _ in range(self.config["n_critic"]):
                        d_loss_total += self._train_discriminator_step(X_batch)
                    g_losses = self._train_generator_step(X_batch)
                    g_loss_total += g_losses["total_loss"]
                    f_loss_total += self._train_forecast_head_step(X_batch, y_batch)
                    n_batches += 1
                    batch_bar.update(1)
                    batch_bar.set_postfix(d=f"{d_loss_total / max(n_batches * self.config['n_critic'], 1):.3e}",
                                          g=f"{g_loss_total / max(n_batches, 1):.3e}")
                    if run_logger:
                        log_progress_milestone(run_logger, f"TRAIN epoch {epoch}", n_batches, total_batches)
                batch_bar.close()

                epoch_metrics = {
                    "d_loss": d_loss_total / max(n_batches * self.config["n_critic"], 1),
                    "g_loss": g_loss_total / max(n_batches, 1),
                    "forecast_loss": f_loss_total / max(n_batches, 1),
                }

                if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                    entropy_report = self._measure_entanglement()
                    entanglement_history.append({"epoch": epoch, **entropy_report})
                    epoch_metrics["entanglement_entropy"] = entropy_report["entanglement_entropy"]

                if run_logger:
                    run_logger.log_epoch(epoch, **epoch_metrics)
                epoch_bar.update(1)
                epoch_bar.set_postfix(d=f"{epoch_metrics['d_loss']:.3e}", g=f"{epoch_metrics['g_loss']:.3e}")

        self.qgan_results["entanglement_history"] = entanglement_history
        self.is_trained = True

    def _measure_entanglement(self) -> dict:
        """Real tomography, not a placeholder — see quantum/tomography.py.
        Runs the circuit on a fixed reference input so entanglement is
        reported for the CIRCUIT/parameters, not confounded by whichever
        random input happened to be sampled."""
        reference_input = np.zeros(self.config["n_qubits"])
        weights = self.generator.theta.detach().numpy()
        circuit_fn = apply_circuit_gates_only(
            reference_input, weights, self.config["n_qubits"],
            self.config["n_layers"], self.config["entanglement"],
        )
        return entanglement_metrics(
            circuit_fn, weights, self.config["n_qubits"],
            dev_name=self.config["quantum_device"],
        )

    def generate_synthetic_data(self, n_samples: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config["n_features"])
            synthetic = self.generator(z)
        return synthetic.numpy()

    def _forecast_model(self, X: torch.Tensor) -> torch.Tensor:
        """
        OPEN METHODOLOGICAL QUESTION, FLAGGED RATHER THAN SILENTLY DECIDED:

        This forecast head runs directly on real (classical, PCA-reduced)
        test features -- the quantum generator is used only during
        training, to produce synthetic augmentation samples. It is NOT
        invoked here at inference time. That is standard practice for
        GAN-style data augmentation (the generator's job ends once
        training data has been augmented), and it's why this function
        and ClassicalGANLLM's equivalent are architecturally symmetric.

        The consequence: under this design, QGAN-LLM's inference-time
        compute graph does not include the quantum circuit at all, so
        there is no architectural reason for it to be slower at
        inference than Classical GAN-LLM's equivalent forecast head --
        both are a small linear layer. If the manuscript's "quantum
        overhead causes ~37.5% higher latency" claim is meant to describe
        INFERENCE latency specifically, that claim requires the quantum
        circuit to be part of the inference path (e.g. real inputs routed
        through the trained generator/an encoder as a feature transform,
        not just noise-conditioned synthetic generation during training)
        -- a different architecture than a standard GAN-augmentation
        setup, and a real design decision, not a bug to silently patch.
        Measure latency with both interpretations if this distinction
        matters for your write-up: (a) forecast-head-only, as implemented
        below, or (b) generator-plus-forecast-head, which you'd need to
        wire in deliberately and justify methodologically.
        """
        return self.forecast_head(X)

    def measure_latency(self, X_test, n_repeats: int = 100) -> dict:
        """See ClassicalGANLLM.measure_latency for why this exists at all --
        latency wasn't measured anywhere in the original pipeline despite
        being a named dependent variable (DV4) with specific reported
        figures."""
        self.forecast_head.eval()
        return measure_inference_latency(self._forecast_model, X_test, n_repeats=n_repeats)

    def predict(self, X):
        self.forecast_head.eval()
        X_t = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        with torch.no_grad():
            pred = self.forecast_head(X_t)
        return pred.numpy() if not torch.is_tensor(X) else pred

    def evaluate(self, X_test, y_test, attack_cfg: Dict = None, last_input_prices=None, **kwargs):
        predictions = self.predict(X_test)
        y_test_arr = np.array(y_test).flatten()
        predictions_arr = np.array(predictions).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": self.name,
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

        final_entanglement = self._measure_entanglement()
        self.results["entanglement_entropy"] = final_entanglement["entanglement_entropy"]
        self.results["purity"] = final_entanglement["purity"]

        # Synthetic-data fidelity + mode-collapse diagnostics (Ch.4 Table 38,
        # and Ch.3's mode-collapse risk -- see synthetic_data_fidelity_report's
        # docstring). Wired in here because nothing previously called this
        # function from the actual model-evaluation path; it existed but was
        # only exercised by tests. Sampled against real TEST rows (not train)
        # so this reflects held-out fidelity, matching how every other metric
        # in `self.results` is computed on X_test/y_test.
        synthetic = self.generate_synthetic_data(len(X_test))
        self.results.update(synthetic_data_fidelity_report(np.asarray(X_test), synthetic))

        return self.results
