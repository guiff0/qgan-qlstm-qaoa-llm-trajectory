"""
Classical GAN-LLM baseline.

======================================================================
GENERATOR vs. DISCRIMINATOR -- who does what, in this file
======================================================================

  GENERATOR   (LSTMGenerator, defined below)
      Role:   Takes random noise z ~ N(0, 1) and outputs one synthetic
              feature vector meant to resemble a real row of market
              data (the same 32 PCA-reduced features -- OHLCV +
              technical indicators -- that a real training row has).
              Its job is data augmentation: the synthetic rows it
              produces get mixed into the LLM's fine-tuning set
              (60% real / 40% synthetic per Ch.3), not used for
              forecasting directly.
      Trained to: fool the discriminator (make synthetic rows
              indistinguishable from real ones) while staying close to
              real data in MSE (the composite generator loss below).

  DISCRIMINATOR   (ClassicalDiscriminator, defined below)
      Role:   Takes ONE feature vector (real or generator-produced)
              and outputs a single probability: "is this real?".
              This is the adversary the generator is trained against
              -- it never sees noise, never produces market data
              itself, and is not used for forecasting either.
      Trained to: correctly separate real rows from the generator's
              synthetic rows (standard binary GAN discriminator).

  FORECASTER   (self.forecast_head, a plain nn.Linear -- NOT shown as
              its own class since it's a single layer, added in
              build() below)
      Role:   This is the model that actually predicts price movement
              at inference time. It is trained directly on real
              features (optionally augmented with the generator's
              synthetic rows) -- NEITHER the generator nor the
              discriminator is invoked at inference. See
              qgan_llm.py's _forecast_model docstring for why this
              matters to the "quantum overhead" latency question.

The identical ClassicalDiscriminator class (same architecture, same
hyperparameters) is reused unchanged in qgan_llm.py -- deliberately, so
that any RMSE/ASR/fidelity difference between Classical GAN-LLM and
QGAN-LLM can be attributed to the GENERATOR (classical LSTM vs.
quantum circuit), not confounded by the two baselines using different
discriminators.

Also fixed here (unrelated to the generator/discriminator split
above): the original train() ran the full training set through the
generator/discriminator once per "epoch" with no DataLoader --
infeasible at 5.5M rows and not a meaningful training loop. Fixed with
real mini-batching. Also replaces the hardcoded
`return 31.0  # Expected ASR from Table 47` with a real call into
src/attacks/adversarial.py.
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
from ..evaluation.metrics import rmse as rmse_fn, mae as mae_fn, synthetic_data_fidelity_report
from ..evaluation.one_step_ahead import shift_for_one_step_ahead
from ..evaluation.poisoning_resistance import evaluate_poisoning_resistance
from ..evaluation.threat_detection import evaluate_threat_detection
from ..evaluation.latency import measure_inference_latency
from ..utils.reproducibility import set_all_seeds, seeded_generator
from ..utils.progress import progress_bar, log_progress_milestone
from ..utils.gan_training import (
    config_fingerprint, checkpoint_file, save_checkpoint, load_checkpoint,
    fit_linear_head_closed_form, epoch_row_indices,
)


class LSTMGenerator(nn.Module):
    """THE GENERATOR (classical side). noise z -> one synthetic feature
    vector. See the module docstring above for its role vs. the
    discriminator and forecaster."""
    def __init__(self, latent_dim=32, hidden_dim=128, output_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (batch, latent_dim) -> add a length-1 sequence dim for the LSTM
        z = z.unsqueeze(1)
        lstm_out, _ = self.lstm(z)
        return self.fc(lstm_out[:, -1, :])


class ClassicalDiscriminator(nn.Module):
    """THE DISCRIMINATOR (shared, unchanged, between both baselines).
    one feature vector (real or synthetic) -> P(real). See the module
    docstring above for why this class is intentionally identical in
    both Classical GAN-LLM and QGAN-LLM."""
    def __init__(self, input_dim=32, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
        # BUG FIX: BatchNorm1d fails on batch_size=1 (which the final,
        # possibly-partial batch of an epoch can be). LayerNorm has no
        # such restriction and is a safe drop-in replacement here.
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc4(x))


class ClassicalGANLLM(BaseForecastingModel):
    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            "latent_dim": 32,
            "generator_hidden": 128,
            "discriminator_hidden": 256,
            "output_dim": 32,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 30,
            "synthetic_ratio": 0.4,
            "n_critic": 2,
            "max_batches_per_epoch": None,   # None = full pass over the training set
            "eval_samples": 20000,           # rows used for synthetic-fidelity metrics
            "resume": True,                  # resume/skip from models/*.ckpt when config+seed match
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("Classical GAN-LLM", cfg)
        self.seed = seed

    def build(self):
        set_all_seeds(self.seed)
        self.generator = LSTMGenerator(
            latent_dim=self.config["latent_dim"],
            hidden_dim=self.config["generator_hidden"],
            output_dim=self.config["output_dim"],
        )
        self.discriminator = ClassicalDiscriminator(
            input_dim=self.config["output_dim"],
            hidden_dim=self.config["discriminator_hidden"],
        )
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config["learning_rate"])
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.config["learning_rate"])
        self.criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        # A small forecasting head trained on the generator's representation,
        # so this baseline has an actual point-forecast to evaluate RMSE on
        # (the original code's `predict()` just returned raw generator
        # output and called it a forecast, conflating "synthetic sample"
        # with "next-step prediction" — those are different tasks).
        self.forecast_head = nn.Linear(self.config["output_dim"], 1)
        self.forecast_optimizer = torch.optim.Adam(self.forecast_head.parameters(), lr=self.config["learning_rate"])

    def _train_discriminator_step(self, real_batch, fake=None):
        batch_size = real_batch.shape[0]
        if fake is None:
            with torch.no_grad():
                fake = self.generator(torch.randn(batch_size, self.config["latent_dim"]))

        real_out = self.discriminator(real_batch)
        fake_out = self.discriminator(fake.detach())
        d_loss = self.criterion(real_out, torch.ones(batch_size, 1)) + \
            self.criterion(fake_out, torch.zeros(batch_size, 1))

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss.item()

    def _train_generator_step(self, real_batch, fake=None):
        batch_size = real_batch.shape[0]
        if fake is None:
            fake = self.generator(torch.randn(batch_size, self.config["latent_dim"]))

        fake_out = self.discriminator(fake)
        adv_loss = self.criterion(fake_out, torch.ones(batch_size, 1))
        mse_loss = self.mse_loss(fake, real_batch)
        g_loss = adv_loss + 0.1 * mse_loss

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()

    def _train_forecast_head_step(self, real_batch, y_batch):
        pred = self.forecast_head(real_batch)
        loss = self.mse_loss(pred.squeeze(-1), y_batch)
        self.forecast_optimizer.zero_grad()
        loss.backward()
        self.forecast_optimizer.step()
        return loss.item()

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        self.build()
        # Same-row target leakage fix -- see
        # src/evaluation/one_step_ahead.py's module docstring.
        X_train, y_train = shift_for_one_step_ahead(np.asarray(X_train), np.asarray(y_train))
        X_val, y_val = shift_for_one_step_ahead(np.asarray(X_val), np.asarray(y_val))
        log = run_logger.info if run_logger else (lambda *_a, **_k: None)

        n_epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        max_batches = self.config.get("max_batches_per_epoch")
        fp = config_fingerprint(self.config, self.seed, "classical_gan_llm")
        self.checkpoint_path = checkpoint_file("classical_gan_llm", fp)

        # ---- resume / skip-if-complete -----------------------------------
        start_epoch, head_mse = 0, float("nan")
        ckpt = load_checkpoint(self.checkpoint_path, fp) if self.config.get("resume", True) else None
        if ckpt is not None:
            self.generator.load_state_dict(ckpt["generator"])
            self.discriminator.load_state_dict(ckpt["discriminator"])
            self.forecast_head.load_state_dict(ckpt["forecast_head"])
            self.g_optimizer.load_state_dict(ckpt["g_optimizer"])
            self.d_optimizer.load_state_dict(ckpt["d_optimizer"])
            start_epoch, head_mse = int(ckpt["next_epoch"]), ckpt.get("head_mse", float("nan"))
            log(f"Resuming from checkpoint {self.checkpoint_path}: next_epoch={start_epoch}/{n_epochs}"
                + (" (training already complete -- skipping to evaluation)" if ckpt.get("completed") else ""))
            if ckpt.get("completed"):
                self.is_trained = True
                return
        else:
            # The forecast head only ever sees REAL rows (the generator never
            # touches it), so it is a convex least-squares problem: fit it
            # exactly instead of 58k Adam steps per epoch.
            head_mse = fit_linear_head_closed_form(self.forecast_head, X_train, y_train)
            log(f"Forecast head fitted in closed form on {len(X_train):,} real rows: train MSE={head_mse:.3e}")

        X_t = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
        n_rows = len(X_t)

        with progress_bar(total=n_epochs, desc="Classical GAN-LLM epochs", unit="epoch") as epoch_bar:
            if start_epoch:
                epoch_bar.update(start_epoch)
            for epoch in range(start_epoch, n_epochs):
                torch.manual_seed(self.seed * 1_000_003 + epoch)  # resume-reproducible noise
                idx = epoch_row_indices(n_rows, batch_size, max_batches, self.seed, epoch)
                total_batches = (len(idx) + batch_size - 1) // batch_size
                d_loss_total, g_loss_total, n_batches = 0.0, 0.0, 0
                batch_bar = progress_bar(total=total_batches, desc=f"  epoch {epoch} batches", unit="batch")
                for s_ in range(0, len(idx), batch_size):
                    X_batch = X_t[idx[s_:s_ + batch_size]]
                    fake = self.generator(torch.randn(len(X_batch), self.config["latent_dim"]))
                    for _ in range(self.config["n_critic"]):
                        d_loss_total += self._train_discriminator_step(X_batch, fake.detach())
                    g_loss_total += self._train_generator_step(X_batch, fake)
                    n_batches += 1
                    batch_bar.update(1)
                    batch_bar.set_postfix(d=f"{d_loss_total / max(n_batches * self.config['n_critic'], 1):.3e}",
                                          g=f"{g_loss_total / max(n_batches, 1):.3e}")
                    if run_logger:
                        log_progress_milestone(run_logger, f"TRAIN epoch {epoch}", n_batches, total_batches)
                batch_bar.close()

                d_avg = d_loss_total / max(n_batches * self.config["n_critic"], 1)
                g_avg = g_loss_total / max(n_batches, 1)
                if run_logger:
                    run_logger.log_epoch(epoch, d_loss=d_avg, g_loss=g_avg, forecast_loss=head_mse)

                save_checkpoint(self.checkpoint_path, {
                    "fingerprint": fp, "next_epoch": epoch + 1, "completed": (epoch + 1 == n_epochs),
                    "generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "forecast_head": self.forecast_head.state_dict(),
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "head_mse": head_mse,
                })
                epoch_bar.update(1)
                epoch_bar.set_postfix(d=f"{d_avg:.3e}", g=f"{g_avg:.3e}")

        self.is_trained = True

    def generate_synthetic_data(self, n_samples: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config["latent_dim"])
            synthetic = self.generator(z)
        return synthetic.numpy()

    def _forecast_model(self, X: torch.Tensor) -> torch.Tensor:
        """Callable used by the adversarial-attack module: takes raw
        features straight to the forecast head (the attack perturbs the
        input features, not the GAN's latent noise)."""
        return self.forecast_head(X)

    def measure_latency(self, X_test, n_repeats: int = 100) -> dict:
        """Real single-sample inference timing (Ch.3 DV4 / Ch.4's reported
        mean=39.2ms, SD=4.1 latency figures) -- not present at all until
        this rebuild, despite the dissertation reporting specific
        mean/SD/distribution latency numbers. Uses this model's own
        forecast callable, so the timing reflects exactly the forward
        pass whose accuracy is also being reported."""
        self.forecast_head.eval()
        return measure_inference_latency(self._forecast_model, X_test, n_repeats=n_repeats)

    def predict(self, X):
        self.forecast_head.eval()
        X_t = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        with torch.no_grad():
            pred = self.forecast_head(X_t)
        return pred.numpy() if not torch.is_tensor(X) else pred

    def evaluate(self, X_test, y_test, attack_cfg: Dict = None, last_input_prices=None, **kwargs):
        X_test, y_test = shift_for_one_step_ahead(np.asarray(X_test), np.asarray(y_test))
        if last_input_prices is not None:
            last_input_prices = np.asarray(last_input_prices)[:-1]  # keep row-aligned with the shift above
        predictions = self.predict(X_test)
        y_test_arr = np.array(y_test).flatten()
        predictions_arr = np.array(predictions).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": "Classical GAN-LLM",
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
            self.results["asr_clean"] = compute_clean_asr(
                self._forecast_model, X_test_t, y_test_t, last_prices_t, attack_cfg,
            )

        # See QGANLLM.evaluate for why this call is here at all (it existed
        # only in tests before now) and why it's sampled against X_test.
        # FIX: was len(X_test) (744,785 rows) -> MMD built a 2 TiB kernel matrix.
        # Fidelity is an estimate; use a fixed-seed subsample of `eval_samples`
        # real test rows vs the same number of synthetic rows (identical for both
        # GAN baselines so FID/MMD/Wasserstein stay comparable).
        n_eval = min(len(X_test), int(self.config.get("eval_samples", 20000)))
        sel = np.sort(np.random.default_rng(self.seed).choice(len(X_test), n_eval, replace=False))
        synthetic = self.generate_synthetic_data(n_eval)
        self.results.update(synthetic_data_fidelity_report(np.asarray(X_test)[sel], synthetic))
        self.results["fidelity_n_samples"] = n_eval

        # See QGANLLM.evaluate for full rationale (same H3/H5 wiring;
        # this baseline just skips the quantum-only resilience suite).
        self.results.update(evaluate_poisoning_resistance(
            self, np.asarray(X_test), np.asarray(y_test), self.results["rmse"], seed=self.seed,
        ))
        if attack_cfg is not None:
            self.forecast_head.eval()
            self.results.update(evaluate_threat_detection(
                self._forecast_model, X_test, y_test, attack_cfg, seed=self.seed,
                n_clean=attack_cfg.get("n_benign_samples", 1000),
            ))

        return self.results
