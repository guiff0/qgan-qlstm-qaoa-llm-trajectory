"""
Shared helpers for the two GAN baselines (Classical GAN-LLM, QGAN-LLM).

Added to fix three problems seen in the first full run:
  1. Nothing was ever saved, so a crash in the evaluation step (MMD memory
     error) discarded 12 hours of GAN training. Now: an atomic checkpoint
     after EVERY epoch, and a completed-run marker, so re-running resumes
     mid-training or skips straight to evaluation.
  2. The forecast head is a single Linear layer trained with MSE on REAL
     rows only -- the generator never touches it -- so it is a convex
     problem. It was being fitted by 58,062 Adam steps per epoch inside the
     GAN loop (the dominant cost for the classical GAN and a wasted one).
     Fit exactly in closed form instead.
  3. torch DataLoader over a TensorDataset indexes row-by-row in Python
     (64 index ops per batch). Batches are now sliced straight from the
     tensors.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np
import torch


_FINGERPRINT_IGNORE = {"forecast_target"}


def config_fingerprint(config: dict, seed: int, tag: str = "") -> str:
    """Stable 10-char hash of everything that determines a training run, so a
    stale checkpoint from a DIFFERENT config/seed/ablation is never resumed
    (QGAN-LLM and its three ablations all share the model name)."""
    # `forecast_target` only changes how the closed-form HEAD is fitted, never the generator, so it is
    # excluded: switching it must not orphan a finished (12 h) generator checkpoint.
    config = {k: v for k, v in config.items() if k not in _FINGERPRINT_IGNORE}
    blob = json.dumps({"c": config, "s": seed, "t": tag}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def checkpoint_file(prefix: str, fingerprint: str, directory: str = "models") -> str:
    return os.path.join(directory, f"{prefix}_{fingerprint}.ckpt")


def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-write can never corrupt the last good checkpoint


def fresh_run() -> bool:
    """True when run_pipeline.py was told to start from scratch (answer 'n', --no-skip, --reset):
    it exports PIPELINE_FRESH=1 to every step. In a fresh run no checkpoint is resumed and no
    cached synthetic rows are reused."""
    return os.environ.get("PIPELINE_FRESH") == "1"


def load_checkpoint(path: str, fingerprint: str) -> Optional[dict]:
    if fresh_run():
        return None   # fresh run: train from epoch 0, even if a matching checkpoint exists
    if not os.path.isfile(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None  # unreadable/partial file -> start fresh rather than crash
    return ckpt if ckpt.get("fingerprint") == fingerprint else None


def fit_linear_head_closed_form(head: torch.nn.Linear, X: np.ndarray, y: np.ndarray,
                                chunk: int = 500_000) -> float:
    """Exact least-squares fit of `head` (weights + bias), accumulating the
    normal equations in chunks so the 3.7M-row set is never copied. Returns
    the training MSE."""
    d = X.shape[1]
    A = np.zeros((d + 1, d + 1))
    b = np.zeros(d + 1)
    for s in range(0, len(X), chunk):
        xb = np.hstack([np.asarray(X[s:s + chunk], dtype=np.float64),
                        np.ones((len(X[s:s + chunk]), 1))])
        A += xb.T @ xb
        b += xb.T @ np.asarray(y[s:s + chunk], dtype=np.float64)
    w = np.linalg.solve(A + 1e-8 * np.eye(d + 1), b)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(w[:d], dtype=torch.float32).unsqueeze(0))
        head.bias.copy_(torch.tensor([w[d]], dtype=torch.float32))
    sse, n = 0.0, 0
    for s in range(0, len(X), chunk):
        pred = np.asarray(X[s:s + chunk], dtype=np.float64) @ w[:d] + w[d]
        sse += float(np.sum((pred - np.asarray(y[s:s + chunk], dtype=np.float64)) ** 2))
        n += len(pred)
    return sse / max(n, 1)


def refit_head_for_frame(head, X, y, ckpt: dict, ckpt_path: str, frame: str, log=None) -> float:
    """A checkpoint written in another target frame (old runs: "level") holds a head fitted for that
    frame. The generator does not depend on the head, so instead of retraining it, archive the old
    checkpoint, refit the head in closed form in the current frame, and rewrite the checkpoint."""
    from .run_identity import archive_file
    before = ckpt.get("target_frame", "level")
    kept = archive_file(ckpt_path, tag=f"{before}_frame")
    mse = fit_linear_head_closed_form(head, X, y)
    ckpt["forecast_head"] = head.state_dict()
    ckpt["head_mse"] = mse
    ckpt["target_frame"] = frame
    save_checkpoint(ckpt_path, ckpt)
    if log:
        log(f"Checkpoint was fitted in the '{before}' frame; head refitted in "
            f"closed form for the '{frame}' frame (train MSE={mse:.3e}); generator untouched. "
            f"Previous checkpoint kept at {kept}")
    return mse


def epoch_row_indices(n_rows: int, batch_size: int, max_batches: Optional[int],
                      seed: int, epoch: int) -> torch.Tensor:
    """Random row order for one epoch. Deterministic in (seed, epoch) so a
    resumed run sees the same data order it would have seen uninterrupted.
    With `max_batches` set, only that many batches' worth of rows is drawn
    (without replacement) -- the compute-budget knob."""
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(epoch))
    perm = torch.randperm(n_rows, generator=g)
    if max_batches:
        perm = perm[: int(max_batches) * int(batch_size)]
    return perm
