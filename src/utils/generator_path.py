"""
Generator-on-the-forecast-path ("forecast_path: generator") for Classical GAN-LLM and QGAN-LLM.

WHY
Until now both GAN baselines predicted with a linear head fitted on REAL rows only. The generator
only produced synthetic rows for the fidelity / poisoning diagnostics, so RMSE, ASR and latency were
mathematically independent of the generator: any "quantum vs classical" difference in them was
noise, and H4 (and the H2/H5 entanglement-vs-ASR tests) could not test anything about the circuit.

WHAT
    forecast(X) = Linear( [ X , G(standardise(X)) ] )

X is the 32 PCA features, G the trained generator (QLSTM circuit or LSTM) applied to the standardised
real input as its latent code, and the head is a ridge-regularised least-squares fit on
[X, G(X)] features. So the circuit is on the inference path: it changes the prediction, the
gradients the adversarial attacks use, and the latency. Everything else (discriminator, GAN training,
head class, features, rows used to fit, evaluation rows) is identical for both models, so a gap
between them is attributable to the generator.

HONEST CAVEATS
  * G was trained on noise z ~ N(0, I), not on real X. Standardising X makes it N(0,1)-like per
    feature, but G(X) is best read as "a learned nonlinear feature map seeded by the GAN", not as a
    synthetic sample.
  * QLSTMGenerator only encodes the FIRST n_qubits entries of its input (20 of 32 features at the
    default width, 12 at the sweep width); the classical LSTM sees all of them.
  * The head is linear in [X, G(X)], so it can never do worse than a pure-X ridge fit on the rows it
    was fitted on; any gain comes from G(X) carrying usable nonlinear structure.
  * Cost is real circuit evaluations: ~0.3 s/row forward at 20 qubits (see qgan_llm.py), more with
    input gradients. head_fit_rows / forecast_eval_samples / attack_samples bound it.
  * The old "head_only" mode is unchanged and remains the default in code, so existing checkpoints
    and results keep their meaning.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

HEAD_ONLY = "head_only"
GENERATOR = "generator"


def use_generator_path(config: dict) -> bool:
    mode = str(config.get("forecast_path", HEAD_ONLY)).lower()
    if mode not in (HEAD_ONLY, GENERATOR):
        raise ValueError(f"forecast_path must be '{HEAD_ONLY}' or '{GENERATOR}', got {mode!r}")
    return mode == GENERATOR


# ---------------------------------------------------------------- pure numpy
def ridge_fit(F: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, float, float]:
    """Ridge least squares with an unpenalised intercept, on per-column standardised features
    (so the penalty treats X and G(X) columns alike), mapped back to raw-feature weights.

    A plain normal-equation solve with a 1e-8 jitter (what the real-rows head uses) is unsafe here:
    [X, G(X)] has more columns than independent directions whenever G's output rank is limited by
    n_qubits, so the matrix is singular. Returns (w, b, train_mse)."""
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if F.ndim != 2 or len(F) != len(y):
        raise ValueError(f"F must be (n, d) and match y: got {F.shape} vs {y.shape}")
    mu, sd = F.mean(axis=0), F.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)            # constant column: leave unscaled, ridge shrinks it to 0
    Z = (F - mu) / sd
    ym = y.mean()
    n, d = Z.shape
    A = Z.T @ Z + ridge * n * np.eye(d)
    w_std = np.linalg.solve(A, Z.T @ (y - ym))
    w = w_std / sd
    b = float(ym - np.sum(w_std * mu / sd))
    mse = float(np.mean((F @ w + b - y) ** 2))
    return w, b, mse


def forecast_eval_indices(n_rows: int, n_wanted: Optional[int], seed: int) -> Optional[np.ndarray]:
    """Fixed-seed sorted subsample used for every forecast-side metric in generator mode, identical
    for both GAN baselines (same seed + same test set -> same rows). None = use everything."""
    if not n_wanted or int(n_wanted) >= n_rows:
        return None
    return np.sort(np.random.default_rng(int(seed)).choice(n_rows, int(n_wanted), replace=False))


def persistence_rmse_on(last: np.ndarray, y_next: np.ndarray, sel: Optional[np.ndarray] = None) -> float:
    """RMSE of the 'no change' forecast (next close = last close) on the SAME rows the model was scored on."""
    last, y_next = np.asarray(last, float).reshape(-1), np.asarray(y_next, float).reshape(-1)
    if sel is not None:
        last, y_next = last[sel], y_next[sel]
    return float(np.sqrt(np.mean((y_next - last) ** 2)))


# --------------------------------------------------------------- torch glue
def _torch():
    import torch  # local import: the numpy helpers above must stay importable without torch
    return torch


def make_generator_path_head(generator, n_features: int, latent_dim: Optional[int] = None,
                             active_dims: Optional[int] = None):
    """Build the head module. Defined lazily so importing this file never requires torch."""
    torch = _torch()
    nn = torch.nn

    class GeneratorPathHead(nn.Module):
        """forecast(X) = linear([X, G(standardise(X))]).

        The generator is held in a plain list on purpose: registering it as a submodule would copy
        its weights into this head's state_dict (and so into every checkpoint a second time) and
        would let the head's optimiser see them. Only the linear layer and the two standardisation
        buffers are saved/loaded here; the generator has its own entry in the checkpoint."""

        def __init__(self):
            super().__init__()
            self._gen_ref = [generator]
            self.n_features = int(n_features)
            self.latent_dim = None if latent_dim is None else int(latent_dim)
            # active_dims: only the first k standardised features reach the generator; the rest are
            # set to 0 (their mean). Gives a classical generator the SAME input information as a
            # circuit that encodes only its first n_qubits features, so a gap is not an input-width gap.
            self.active_dims = None if active_dims is None else int(active_dims)
            self.linear = nn.Linear(2 * self.n_features, 1)
            self.register_buffer("z_mean", torch.zeros(self.n_features))
            self.register_buffer("z_std", torch.ones(self.n_features))

        @property
        def generator(self):
            return self._gen_ref[0]

        def _latent(self, X):
            z = (X - self.z_mean) / self.z_std
            k = self.active_dims
            if k is not None and 0 < k < z.shape[1]:
                z = torch.cat([z[:, :k], torch.zeros_like(z[:, k:])], dim=1)
            if self.latent_dim is not None and z.shape[1] != self.latent_dim:
                if z.shape[1] > self.latent_dim:
                    z = z[:, : self.latent_dim]
                else:
                    z = torch.cat([z, torch.zeros(z.shape[0], self.latent_dim - z.shape[1],
                                                  dtype=z.dtype, device=z.device)], dim=1)
            return z

        def features(self, X):
            return torch.cat([X, self.generator(self._latent(X))], dim=1)

        def forward(self, X):
            return self.linear(self.features(X))

        def train(self, mode: bool = True):
            super().train(mode)
            self.generator.train(mode)     # eval() must also switch off the generator's training-time noise
            return self

    return GeneratorPathHead()


def chunked_forward(head, X, chunk: int = 256):
    """head(X) in chunks, no grad. A circuit evaluation per row cannot be batched into one giant call."""
    torch = _torch()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), int(chunk)):
            outs.append(head(X[s:s + int(chunk)]))
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, 1))


def fit_generator_path_head(head, X_train, y_train, n_rows: int, seed: int, chunk: int = 256, log=None,
                            ridge: float = 1e-6) -> float:
    """Fit the head on [X, G(X)] for a fixed-seed subsample of the TRAIN rows (the same rows for both
    GAN models). Sets the standardiser buffers, then the linear weights. Returns the train MSE."""
    torch = _torch()
    n = len(X_train)
    rows = min(int(n_rows), n)
    sel = np.sort(np.random.default_rng(int(seed)).choice(n, rows, replace=False))
    Xs = np.ascontiguousarray(np.asarray(X_train)[sel], dtype=np.float32)
    ys = np.asarray(y_train, dtype=np.float64).reshape(-1)[sel]
    with torch.no_grad():
        head.z_mean.copy_(torch.from_numpy(Xs.mean(axis=0)))
        head.z_std.copy_(torch.from_numpy(Xs.std(axis=0)).clamp_min(1e-8))
    head.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, rows, int(chunk)):
            feats.append(head.features(torch.from_numpy(Xs[s:s + int(chunk)])).numpy())
            if log and (s // int(chunk)) % 10 == 0:
                log(f"  generator-path head: features for {min(s + int(chunk), rows):,}/{rows:,} rows")
    F = np.concatenate(feats, axis=0)
    w, b, mse = ridge_fit(F, ys, ridge=ridge)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor(w, dtype=torch.float32).unsqueeze(0))
        head.linear.bias.copy_(torch.tensor([b], dtype=torch.float32))
    return mse
