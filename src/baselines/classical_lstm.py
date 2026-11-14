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
from ..evaluation.one_step_ahead import resolve_frame
from ..evaluation.latency import measure_inference_latency
from ..attacks.adversarial import compute_attack_success_rate, compute_clean_asr
from ..utils.reproducibility import set_all_seeds, seeded_generator
from ..utils.progress import progress_bar, log_progress_milestone


CHECKPOINT_PATH = "models/classical_lstm_best.pt"              # level frame (the October checkpoint; never overwritten by a delta run)
CHECKPOINT_PATH_DELTA = "models/classical_lstm_delta_best.pt"  # delta frame (14 Nov 2026)
_CHECKPOINT_FORMAT_VERSION = 3                                 # 3 = records target_frame and target_scale


def window_targets(y, frame: str, scale: float = 1.0) -> np.ndarray:
    """Per-row training target for WindowedSequenceDataset, which pairs window rows [i, i+L) with target[i+L].
      level: target[j] = y[j]                        (the original behaviour)
      delta: target[j] = (y[j] - y[j-1]) / scale     (next-close CHANGE after the window, standardised)
    The difference is taken in float64: prices ~1.1 with changes ~1e-4 lose precision in float32. target[0] has no
    previous close; it is 0 and is never read (the first window's target is row L >= 1)."""
    y64 = np.asarray(y, dtype=np.float64).reshape(len(y), -1)[:, 0]
    if frame == "level":
        return y64.astype(np.float32)
    t = np.zeros_like(y64)
    t[1:] = (y64[1:] - y64[:-1]) / (scale if scale > 0 else 1.0)
    return t.astype(np.float32)


from ..utils.stage_log import apply_timing as _apply_timing
_apply_timing(globals(), {'compute_attack_success_rate': 'attacks (FGSM/PGD/CW)', 'compute_clean_asr': 'clean-baseline ASR', 'synthetic_data_fidelity_report': 'fidelity metrics (FID/MMD/Wasserstein)', 'evaluate_poisoning_resistance': 'poisoning + model-inversion tests', 'evaluate_threat_detection': 'threat detection', 'run_resilience_suite': 'quantum resilience suite (total)', 'entanglement_metrics': 'entanglement tomography', 'measure_inference_latency': 'latency measurement'})


class ClassicalLSTM(BaseForecastingModel):
    checkpoint_path = CHECKPOINT_PATH     # replaced per instance in __init__ (one file per target frame)

    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            # "delta" (default since 14 Nov 2026): the network forecasts the next-close CHANGE; price forecast = last
            # close + change, so a network that learns nothing scores exactly persistence. "level" = the October
            # behaviour (regress the price itself), kept so the old rows stay reproducible.
            "forecast_target": "delta",
            # Input centring (14 Nov 2026). "window": every 60-row window has its own mean subtracted before the LSTM sees
            # it, so the network gets where each feature sits RELATIVE to the last hour, not its absolute level. Reason,
            # measured on train/validation only: four PCA components sit 2.6-4.8 training SDs away in the validation
            # years (they carry the price and macro LEVEL), and a linear next-bar model on the raw components is worse
            # than no-change on validation (1.0004) while the same model on drift-free inputs is not (0.9999).
            # None = "window" in the delta frame, "none" in the level frame (the October behaviour).
            "input_centering": None,
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
        self.target_frame = resolve_frame(cfg)
        self.target_scale = 1.0           # SD of the training next-close changes (delta frame); 1 in the level frame
        # run_all reuses `checkpoint_path` when the file exists: a delta run must never pick up the level checkpoint.
        self.checkpoint_path = CHECKPOINT_PATH if self.target_frame == "level" else CHECKPOINT_PATH_DELTA
        cen = cfg.get("input_centering") or ("window" if self.target_frame == "delta" else "none")
        if cen not in ("window", "none"):
            raise ValueError(f"classical_lstm.input_centering must be 'window' or 'none', got {cen!r}")
        self.input_centering = cen

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
        if self.target_frame == "delta":
            # Start AT the no-change forecast (output 0) instead of at a random offset: training can then only be kept
            # when it improves on persistence on the validation rows (see the epoch-0 baseline in train()).
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()),
            lr=self.config["learning_rate"],
        )
        self.criterion = nn.MSELoss()

    def _features(self, X_seq: torch.Tensor) -> torch.Tensor:
        """Last-timestep LSTM hidden state, (batch, hidden_size)."""
        if self.input_centering == "window":
            X_seq = X_seq - X_seq.mean(dim=1, keepdim=True)      # per window, per feature; uses rows inside the window only
        lstm_out, _ = self.model(X_seq)
        return lstm_out[:, -1, :]

    def _forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        """X_seq: (batch, sequence_length, n_features) -- a REAL sequence,
        not a length-1 stand-in."""
        return self.fc(self._features(X_seq))

    def _forecast(self, X_seq: torch.Tensor) -> torch.Tensor:
        """Forecast in price units for this frame: the next-close CHANGE (delta) or the next close (level)."""
        return self._forward(X_seq) * self.target_scale

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
                "target_frame": self.target_frame,
                "target_scale": float(self.target_scale),
                "input_centering": self.input_centering,
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
        ck_frame = ckpt.get("target_frame", "level")        # checkpoints written before 14 Nov 2026 are level-frame
        if ck_frame != self.target_frame:
            raise RuntimeError(
                f"{path} was trained in the '{ck_frame}' frame but this run asks for '{self.target_frame}' "
                f"(classical_lstm.forecast_target). The network is target-specific: retrain (do not pass "
                f"--reuse-checkpoint), or set forecast_target: {ck_frame} to evaluate the old checkpoint.")
        self.model.load_state_dict(ckpt["lstm"])
        self.fc.load_state_dict(ckpt["fc"])
        self.target_scale = float(ckpt.get("target_scale", 1.0))
        self.input_centering = ckpt.get("input_centering", "none")     # the network was trained with this; never mix
        self.is_trained = True

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        self.build()
        seq_len = self.config["sequence_length"]
        if self.target_frame == "delta":
            # The network is supervised on the target, so a change of ~1e-4 must be standardised (same reason as the
            # QLSTM Forecaster). The scale comes from the TRAINING rows only and is undone in _forecast().
            d = np.diff(np.asarray(y_train, dtype=np.float64).reshape(len(y_train), -1)[:, 0])
            self.target_scale = float(np.std(d)) if len(d) and float(np.std(d)) > 0 else 1.0
        else:
            self.target_scale = 1.0
        y_train = window_targets(y_train, self.target_frame, self.target_scale)
        y_val = window_targets(y_val, self.target_frame, self.target_scale)
        if run_logger:
            run_logger.info(f"Classical LSTM target frame: {self.target_frame} (target scale {self.target_scale:.3e}); "
                            f"checkpoint {self.checkpoint_path}")

        train_ds = WindowedSequenceDataset(X_train, y_train, sequence_length=seq_len)
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True,
            generator=seeded_generator(self.seed),
        )
        val_ds = WindowedSequenceDataset(X_val, y_val, sequence_length=seq_len)
        # Early stopping only needs a stable validation signal, not all ~744k overlapping windows every
        # epoch: a fixed-seed subsample (same rows every epoch, so val losses stay comparable).
        n_val = self.config.get("val_samples")
        if n_val and int(n_val) < len(val_ds):
            sel = np.sort(np.random.default_rng(self.seed).choice(len(val_ds), int(n_val), replace=False))
            val_ds = torch.utils.data.Subset(val_ds, sel.tolist())
        val_loader = DataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        n_epochs = self.config["epochs"]
        if self.target_frame == "delta":
            # Epoch-0 baseline: the untrained network outputs 0 = the no-change forecast. Its validation loss is the bar
            # every epoch has to beat; if none does, the saved model IS persistence and the run says so.
            self.model.eval()
            v0, nb0 = 0.0, 0
            with torch.no_grad():
                for X_batch, y_batch, _lp in val_loader:
                    v0 += self.criterion(self._forward(X_batch).squeeze(-1), y_batch).item()
                    nb0 += 1
            best_val_loss = v0 / max(nb0, 1)
            self.save_model(self.checkpoint_path)
            self.baseline_val_loss, self.best_epoch = best_val_loss, -1
            if run_logger:
                run_logger.info(f"epoch-0 baseline (no-change forecast): val_loss={best_val_loss:.6e}")
        # Compute budget. Consecutive 1-minute windows share 59 of 60 rows, so a full pass over all ~3.7M
        # windows (58k batches of 64, ~2 h/epoch on CPU) mostly re-reads the same data. With shuffle=True the
        # first `max_batches_per_epoch` batches are a fresh random subsample every epoch (None = full pass).
        max_batches = self.config.get("max_batches_per_epoch")
        total_batches = min(len(train_loader), int(max_batches)) if max_batches else len(train_loader)

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
                    if n_batches >= total_batches:
                        break
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
                    self.best_epoch = epoch
                    patience_counter = 0
                    self.save_model(self.checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["early_stopping_patience"]:
                        if run_logger:
                            run_logger.info(f"Early stopping at epoch {epoch}")
                        break

        self.is_trained = True
        self.load_model(self.checkpoint_path)

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
                outs.append(self._forecast(X_seq).cpu().numpy())
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
        chosen windows (fixed seed). last_input_price is the unshifted close at the window's
        final row (run_all.py passes y_test), same as the other models.
        Threat detection is not run here: its classifier expects flat 2-D rows,
        not windows."""
        seq_len = self.config["sequence_length"]
        predictions = self.predict(X_test)
        y64 = np.asarray(y_test, dtype=np.float64).reshape(len(y_test), -1)[:, 0]
        y_next, last = y64[seq_len:], y64[seq_len - 1:-1]      # close after each window / last close inside it
        # delta: scored on the next-close CHANGE; this RMSE equals the price-forecast RMSE (last close + change vs next
        # close), so it is directly comparable with persistence and with level-frame rows.
        y_test_arr = (y_next - last) if self.target_frame == "delta" else y_next
        predictions_arr = np.array(predictions, dtype=np.float64).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": "Classical LSTM",
            "target_frame": self.target_frame,
            "input_centering": self.input_centering,
            "best_epoch": getattr(self, "best_epoch", None),      # -1 = no epoch beat the no-change forecast on validation
            # persistence (next close = last close) on EXACTLY the windows `rmse` was scored on
            "persistence_rmse": float(np.sqrt(np.mean((y_next - last) ** 2))),
        }

        if attack_cfg is not None:
            self.model.eval()
            view = self._window_view(X_test)
            n_att = min(int(self.config.get("attack_samples", 2000)), len(view))
            sel = np.sort(np.random.default_rng(self.seed).choice(len(view), n_att, replace=False))
            X_w = torch.from_numpy(np.ascontiguousarray(view[sel]))
            y_w = torch.tensor(y_test_arr[sel], dtype=torch.float32)
            X_np = np.asarray(X_test, dtype=np.float32)
            if self.target_frame == "delta":
                # the forecast IS a change, so the directional-flip reference is 0 (same convention as make_target)
                last_w = torch.zeros(n_att, dtype=torch.float32)
            elif last_input_prices is not None:
                # The window ending at row k predicts the close at row k+1, so the
                # last observed close is the UNSHIFTED y_test value at row k.
                lp = np.asarray(last_input_prices, dtype=np.float32).flatten()
                last_w = torch.tensor(lp[sel + seq_len - 1], dtype=torch.float32)
            else:
                import warnings
                warnings.warn("last_input_prices not given: falling back to PCA component 0, "
                              "which is NOT a price; the resulting ASR is not valid.")
                last_w = torch.tensor(X_np[sel + seq_len - 1, 0], dtype=torch.float32)
            asr_report = compute_attack_success_rate(
                self._forecast, X_w, y_w, last_w, attack_cfg,
                attacks=attack_cfg.get("attacks", ["fgsm", "pgd", "cw"]),
            )
            self.results["asr"] = asr_report["overall_asr"]
            self.results["asr_breakdown"] = asr_report
            self.results["asr_clean"] = compute_clean_asr(self._forecast, X_w, y_w, last_w, attack_cfg)
            self.results["attack_n_samples"] = n_att
        return self.results
