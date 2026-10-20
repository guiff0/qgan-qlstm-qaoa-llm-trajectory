"""
Feature-map forecaster: the minimal FAIR test of quantum advantage in
forecasting that this repository can run.

Three arms, identical in every respect except the feature map:

  linear        ridge on x                          (no extra features)
  classical_rf  ridge on [x, phi_classical(x)]      (matched random features)
  quantum       ridge on [x, phi_quantum(x)]        (PQC <Z> features)

Differences from the GAN baselines that the null result exposed:

  * The circuit IS on the inference path (phi_quantum is applied to every
    test row), so forecasts can differ between arms.
  * The target is the one-step RESIDUAL over persistence,
        r_t = close_{t+1} - close_t,
    so the model is scored on what persistence cannot explain. In the logged
    runs persistence (val RMSE 1.64e-4) beat every model, leaving no
    headroom for any architecture; the residual target removes that floor.
  * Everything is fitted in closed form with a ridge strength chosen on the
    validation split from one shared grid, so no arm gets more tuning.
  * Seeds vary the random circuit/feature parameters, giving real replicates
    (the earlier study used a single seed).

Pure numpy: no torch / PennyLane needed.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..evaluation.advantage_diagnostics import skill_vs_persistence
from ..evaluation.one_step_ahead import shift_for_one_step_ahead
from ..quantum.feature_map import ClassicalRandomFeatureMap, QuantumFeatureMap

ARMS = ("linear", "classical_rf", "quantum")
# penalty = lam * n_rows * I on standardised columns, so lam is scale-free
DEFAULT_RIDGE_GRID = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)


class FeatureMapForecaster:
    def __init__(self, arm: str = "linear", n_qubits: int = 8, n_layers: int = 3,
                 entanglement: str = "ring", input_scale: float = 1.0, seed: int = 0,
                 ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID, chunk: int = 20_000):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.arm, self.n_qubits, self.n_layers = arm, n_qubits, n_layers
        self.entanglement, self.input_scale, self.seed = entanglement, input_scale, seed
        self.ridge_grid, self.chunk = tuple(ridge_grid), chunk
        self.fmap = None
        if arm == "quantum":
            self.fmap = QuantumFeatureMap(n_qubits, n_layers, entanglement, seed=seed, chunk=chunk)
        elif arm == "classical_rf":
            n_params = n_layers * n_qubits * 3
            self.fmap = ClassicalRandomFeatureMap(n_qubits, n_qubits, n_params, seed=seed)
        self._fitted = False

    # -- feature construction ------------------------------------------------
    def _angles(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64)[:, : self.n_qubits] - self.mu_) / self.sd_
        return Z * self.input_scale

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.fmap is None:
            return X
        return np.hstack([X, self.fmap.transform(self._angles(X))])

    def _design_chunked(self, X: np.ndarray):
        for s in range(0, len(X), self.chunk):
            yield s, self._design(X[s:s + self.chunk])

    # -- fitting ---------------------------------------------------------------
    @staticmethod
    def _residual_pairs(X, y):
        """(X[t], close[t], close[t+1]) for t = 0..n-2."""
        Xs, y_next = shift_for_one_step_ahead(np.asarray(X), np.asarray(y))
        y_cur = np.asarray(y, dtype=np.float64)[:-1]
        return Xs, y_cur, np.asarray(y_next, dtype=np.float64)

    def fit(self, X_train, y_train, X_val, y_val) -> "FeatureMapForecaster":
        Xt, cur_t, nxt_t = self._residual_pairs(X_train, y_train)
        Xv, cur_v, nxt_v = self._residual_pairs(X_val, y_val)
        r_t, r_v = nxt_t - cur_t, nxt_v - cur_v

        Xt64 = np.asarray(Xt, dtype=np.float64)
        self.mu_ = Xt64[:, : self.n_qubits].mean(axis=0)
        self.sd_ = Xt64[:, : self.n_qubits].std(axis=0) + 1e-12

        # accumulate normal equations on standardised design columns
        d = None
        n = 0
        s1 = s2 = None
        for _, D in self._design_chunked(Xt):
            if d is None:
                d = D.shape[1]
                s1, s2 = np.zeros(d), np.zeros(d)
            s1 += D.sum(axis=0)
            s2 += (D ** 2).sum(axis=0)
            n += len(D)
        self.col_mu_ = s1 / n
        self.col_sd_ = np.sqrt(np.maximum(s2 / n - self.col_mu_ ** 2, 0.0)) + 1e-12
        self.r_mu_ = float(r_t.mean())

        A = np.zeros((d, d))
        b = np.zeros(d)
        for s, D in self._design_chunked(Xt):
            Dz = (D - self.col_mu_) / self.col_sd_
            A += Dz.T @ Dz
            b += Dz.T @ (r_t[s:s + len(D)] - self.r_mu_)

        best = (np.inf, None, None)
        Vz = np.vstack([(D - self.col_mu_) / self.col_sd_ for _, D in self._design_chunked(Xv)])
        for lam in self.ridge_grid:
            w = np.linalg.solve(A + lam * n * np.eye(d), b)
            mse = float(np.mean((self.r_mu_ + Vz @ w - r_v) ** 2))
            if mse < best[0]:
                best = (mse, lam, w)
        self.val_mse_, self.ridge_, self.w_ = best
        self.n_features_ = d
        self._fitted = True
        return self

    # -- inference -----------------------------------------------------------
    def predict_residual(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        out = []
        for _, D in self._design_chunked(np.asarray(X)):
            out.append(self.r_mu_ + ((D - self.col_mu_) / self.col_sd_) @ self.w_)
        return np.concatenate(out)

    def evaluate(self, X_test, y_test) -> Dict:
        """RMSE, persistence RMSE, MSE skill score, and the raw error series
        (for paired Diebold-Mariano tests between arms on the SAME rows)."""
        Xs, cur, nxt = self._residual_pairs(X_test, y_test)
        pred = cur + self.predict_residual(Xs)
        rep = skill_vs_persistence(nxt, pred, cur)
        rep.update({"arm": self.arm, "ridge": self.ridge_, "n_features": self.n_features_,
                    "n_test": int(len(nxt)), "errors": (nxt - pred)})
        return rep
