"""
Standalone Classical LSTM baseline.

TWO ISSUES FIXED IN THIS FILE:

1. The original train() did:
       lstm_out, _ = self.model(X_train)     # X_train = ENTIRE training set
   passing all ~5.5 million rows through the LSTM in one forward pass per
   "epoch" -- infeasible at this scale and not a meaningful training loop
   (one gradient update per epoch instead of thousands). Fixed with a
   DataLoader-based mini-batch loop.

2. METHODOLOGY GAP: every row was treated as an independent, length-1
   "sequence" (X.unsqueeze(1)), which never lets the LSTM's recurrence
   see more than one timestep -- it contributes nothing over a plain
   gated feedforward layer. Fixed by using a genuine sliding window
   (src/data/windowing.py), matching the sequence_length: 60 lookback
   already named (but previously unused) in config/default_config.yaml.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseForecastingModel
from ..data.windowing import WindowedSequenceDataset
from ..evaluation.metrics import rmse as rmse_fn, mae as mae_fn
from ..evaluation.latency import measure_inference_latency
from ..attacks.adversarial import compute_attack_success_rate, compute_clean_asr
from ..utils.reproducibility import set_all_seeds, seeded_generator
from ..utils.progress import progress_bar, log_progress_milestone


CHECKPOINT_PATH = "models/classical_lstm_best.pt"
_CHECKPOINT_FORMAT_VERSION = 2


from ..utils.stage_log import apply_timing as _apply_timing
_apply_timing(globals(), {'compute_attack_success_rate': 'attacks (FGSM/PGD/CW)', 'compute_clean_asr': 'clean-baseline ASR', 'synthetic_data_fidelity_report': 'fidelity metrics (FID/MMD/Wasserstein)', 'evaluate_poisoning_resistance': 'poisoning + model-inversion tests', 'evaluate_threat_detection': 'threat detection', 'run_resilience_suite': 'quantum resilience suite (total)', 'entanglement_metrics': 'entanglement tomography', 'measure_inference_latency': 'latency measurement'})


class ClassicalLSTM(BaseForecastingModel):
    checkpoint_path = CHECKPOINT_PATH

    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            "input_size": 32,
            "hidden_size": 128,
            "num_layers": 2,
            "output_size": 1,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "early_stopping_patience": 10,
            "sequence_length": 60,
            # Inference-time chunk size. Memory per chunk is roughly
            # predict_batch_size * sequence_length * n_features * 4 bytes
            # (4096 * 60 * 32 * 4 = ~31 MB), independent of test-set size.
            "predict_batch_size": 4096,
            "attack_samples": 2000,  # random windows attacked (fixed seed) -- needed for ASR / TaLIS
            # Heartbeat during training so a 50-minute epoch isn't silent.
            "log_every_batches": 5000,
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("Classical LSTM", cfg)
        self.seed = seed

    def build(self):
        set_all_seeds(self.seed)
        self.model = nn.LSTM(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"] if self.config["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(self.config["hidden_size"], self.config["output_size"])
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()),
            lr=self.config["learning_rate"],
        )
        self.criterion = nn.MSELoss()

    def _features(self, X_seq: torch.Tensor) -> torch.Tensor:
        """Last-timestep LSTM hidden state, (batch, hidden_size)."""
        lstm_out, _ = self.model(X_seq)
        return lstm_out[:, -1, :]

    def _forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        """X_seq: (batch, sequence_length, n_features) -- a REAL sequence,
        not a length-1 stand-in."""
        return self.fc(self._features(X_seq))

    # ------------------------------------------------------------------
    # Checkpointing
    #
    # BUG FIXED: BaseForecastingModel.save_model/load_model only handle
    # self.model (the nn.LSTM). The output head self.fc (nn.Linear) was
    # never saved, so "restore the best epoch" at the end of train()
    # restored the best-epoch LSTM but left fc at its LAST-epoch weights
    # -- a mismatched pair whose predictions are not those of any epoch
    # that was actually validated. Both modules are saved together now.
    # ------------------------------------------------------------------
    def save_model(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "format_version": _CHECKPOINT_FORMAT_VERSION,
                "lstm": self.model.state_dict(),
                "fc": self.fc.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        if not (isinstance(ckpt, dict) and "lstm" in ckpt and "fc" in ckpt):
            raise RuntimeError(
                f"{path} is a legacy LSTM-only checkpoint with no output head (fc). "
                f"Loading it would silently pair its LSTM weights with an untrained "
                f"head. Refit the head from the frozen LSTM with:\n"
                f"    python -m src.experiments.refit_lstm_head\n"
                f"or retrain."
            )
        self.model.load_state_dict(ckpt["lstm"])
        self.fc.load_state_dict(ckpt["fc"])
        self.is_trained = True

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        self.build()
        seq_len = self.config["sequence_length"]

        train_ds = WindowedSequenceDataset(X_train, y_train, sequence_length=seq_len)
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True,
            generator=seeded_generator(self.seed),
        )
        val_ds = WindowedSequenceDataset(X_val, y_val, sequence_length=seq_len)
        val_loader = DataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        n_epochs = self.config["epochs"]
        total_batches = len(train_loader)

        # Two-level progress: an outer bar over epochs (stays on screen the
        # whole run) and an inner bar over batches within the current epoch
        # (so a single 50-minute epoch -- see the classical_lstm crash log
        # this project started from -- still shows live %, not silence).
        with progress_bar(total=n_epochs, desc="Classical LSTM epochs", unit="epoch") as epoch_bar:
            for epoch in range(n_epochs):
                self.model.train()
                epoch_loss, n_batches = 0.0, 0
                epoch_start = time.time()
                log_every = self.config.get("log_every_batches") or 0
                batch_bar = progress_bar(total=total_batches, desc=f"  epoch {epoch} batches", unit="batch")
                for X_batch, y_batch, _last_price in train_loader:
                    self.optimizer.zero_grad()
                    pred = self._forward(X_batch)
                    loss = self.criterion(pred.squeeze(-1), y_batch)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                    batch_bar.update(1)
                    batch_bar.set_postfix(loss=f"{epoch_loss / n_batches:.3e}")
                    if run_logger and log_every and n_batches % log_every == 0:
                        log_progress_milestone(run_logger, f"TRAIN epoch {epoch}", n_batches, total_batches,
                                               running_loss=f"{epoch_loss / n_batches:.3e}")
                batch_bar.close()
                train_loss = epoch_loss / max(n_batches, 1)

                self.model.eval()
                val_loss_total, n_val_batches = 0.0, 0
                with torch.no_grad():
                    for X_batch, y_batch, _last_price in val_loader:
                        val_pred = self._forward(X_batch)
                        val_loss_total += self.criterion(val_pred.squeeze(-1), y_batch).item()
                        n_val_batches += 1
                val_loss = val_loss_total / max(n_val_batches, 1)

                if run_logger:
                    run_logger.log_epoch(epoch, train_loss=train_loss, val_loss=val_loss)
                epoch_bar.update(1)
                epoch_bar.set_postfix(train_loss=f"{train_loss:.3e}", val_loss=f"{val_loss:.3e}",
                                      best=f"{best_val_loss:.3e}", patience=patience_counter)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(CHECKPOINT_PATH)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["early_stopping_patience"]:
                        if run_logger:
                            run_logger.info(f"Early stopping at epoch {epoch}")
                        break

        self.is_trained = True
        self.load_model(CHECKPOINT_PATH)

    def _window_view(self, X) -> np.ndarray:
        """Zero-copy (n_windows, seq_len, n_features) VIEW of a flat
        (n_rows, n_features) array. Nothing is materialized until a slice
        is copied out, so this costs O(1) memory regardless of test-set size.

        Window i is rows [i, i + seq_len), for i in [0, len(X) - seq_len),
        exactly the windows the old np.stack version produced (so the
        y_test[seq_len:] alignment in evaluate() is unchanged).

        BUG FIXED: the old _window_inputs_only() np.stack'ed EVERY window
        into one array. On a ~2.9M-row test split that is 2.9M * 60 * 32 * 4
        bytes = ~22 GB -- the 'not enough memory: 22355374080 bytes' crash
        -- and measure_latency() then built (and re-copied) it a second time.
        """
        seq_len = self.config["sequence_length"]
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        n_windows = len(X) - seq_len
        if n_windows <= 0:
            raise ValueError(
                f"Need more than {seq_len} rows to build a windowed sequence; got {len(X)}."
            )
        return sliding_window_view(X, (seq_len, X.shape[1]))[:n_windows, 0]

    def _iter_window_batches(self, X, batch_size: int) -> Iterator[torch.Tensor]:
        if torch.is_tensor(X) and X.dim() == 3:  # caller already windowed
            for s in range(0, len(X), batch_size):
                yield X[s: s + batch_size]
            return
        view = self._window_view(X.numpy() if torch.is_tensor(X) else X)
        for s in range(0, len(view), batch_size):
            yield torch.from_numpy(np.ascontiguousarray(view[s: s + batch_size]))

    def predict(self, X):
        """Chunked inference: peak memory is one batch, not the whole split."""
        self.model.eval()
        bs = int(self.config.get("predict_batch_size", 4096))
        n_rows = len(X) if not torch.is_tensor(X) or X.dim() != 3 else len(X)
        n_chunks = max(1, -(-max(n_rows - self.config["sequence_length"], 0) // bs))
        outs = []
        with torch.no_grad():
            for X_seq in progress_bar(self._iter_window_batches(X, bs), total=n_chunks,
                                      desc="Classical LSTM predict", unit="chunk"):
                outs.append(self._forward(X_seq).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def measure_latency(self, X_test, n_repeats: int = 100) -> dict:
        """Timed on one full window (sequence_length rows in, one
        prediction out) -- the unit of a single 'input to output' pass
        for a sequence model, matching Ch.3's operational definition of
        DV4 (Response Latency). See ClassicalGANLLM.measure_latency for
        why this exists at all.

        Only a small fixed pool of randomly chosen windows is built (not
        every window in the test set); measure_inference_latency then draws
        single windows from that pool. Pool selection is seeded, so it is
        reproducible."""
        self.model.eval()
        view = self._window_view(X_test)
        rng = np.random.default_rng(42)
        pool_size = min(len(view), 512)
        idx = rng.choice(len(view), size=pool_size, replace=False)
        pool = np.stack([view[i] for i in idx])  # (pool_size, seq_len, n_features)
        return measure_inference_latency(self._forward, pool, n_repeats=n_repeats)

    def evaluate(self, X_test, y_test, attack_cfg: dict = None, last_input_prices=None, **kwargs):
        """NOTE: predictions correspond to rows [sequence_length, len(X_test))
        of the input -- the first `sequence_length` rows have no full window
        and are dropped, so y_test is trimmed to match before scoring.

        ASR (needed for TaLIS, which is NaN without it): the attacks perturb
        the whole input window (L-inf budget epsilon on every one of the
        sequence_length x n_features entries) of `attack_samples` randomly
        chosen windows (fixed seed). last_input_price is the first feature of
        the window's final row, same convention as run_all.py's other models.
        Threat detection is not run here: its classifier expects flat 2-D rows,
        not windows."""
        seq_len = self.config["sequence_length"]
        predictions = self.predict(X_test)
        y_test_arr = np.array(y_test).flatten()[seq_len:]
        predictions_arr = np.array(predictions).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": "Classical LSTM",
        }

        if attack_cfg is not None:
            self.model.eval()
            view = self._window_view(X_test)
            n_att = min(int(self.config.get("attack_samples", 2000)), len(view))
            sel = np.sort(np.random.default_rng(self.seed).choice(len(view), n_att, replace=False))
            X_w = torch.from_numpy(np.ascontiguousarray(view[sel]))
            y_w = torch.tensor(y_test_arr[sel], dtype=torch.float32)
            X_np = np.asarray(X_test, dtype=np.float32)
            last_w = torch.tensor(X_np[sel + seq_len - 1, 0], dtype=torch.float32)
            asr_report = compute_attack_success_rate(
                self._forward, X_w, y_w, last_w, attack_cfg,
                attacks=attack_cfg.get("attacks", ["fgsm", "pgd", "cw"]),
            )
            self.results["asr"] = asr_report["overall_asr"]
            self.results["asr_breakdown"] = asr_report
            self.results["asr_clean"] = compute_clean_asr(self._forward, X_w, y_w, last_w, attack_cfg)
            self.results["attack_n_samples"] = n_att
        return self.results
