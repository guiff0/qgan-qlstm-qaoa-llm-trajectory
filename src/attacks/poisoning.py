"""
Label-flipping data poisoning injection -- ported and adapted from an
external repository audit (this was a genuinely real, working file
there). Injects poisoning by flipping a fraction of training labels,
which is what RQ3/H3's "robustness under data poisoning" results
(e.g. Table 45's "Detection Accuracy (5% Poisoning)") need as their
actual poisoning mechanism -- something not previously implemented
anywhere in this codebase.

Adapted from the original (numpy-only, fixed RNG) to accept a seed
parameter explicitly, matching this codebase's reproducibility
conventions (src/utils/reproducibility.py) rather than hardcoding
np.random.default_rng(0) inside the function.
"""
from __future__ import annotations

import numpy as np


def inject_label_flip_poisoning(X: np.ndarray, y: np.ndarray, fraction: float = 0.05,
                                 seed: int = 0) -> dict:
    """
    Flips the labels of a random `fraction` of samples (5% by default,
    matching Table 45's "5% Poisoning" condition). For binary labels,
    "flip" means 1 - y; for continuous targets (e.g. regression labels),
    a binary flip doesn't apply -- see invert_continuous_targets() below
    for the regression-appropriate equivalent.

    Returns the (unmodified) X, the poisoned y, and the indices that
    were poisoned -- callers need those indices to later check e.g.
    "was this specific poisoned sample correctly flagged," not just an
    aggregate accuracy number.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1); got {fraction}")

    n = len(y)
    k = int(n * fraction)
    rng = np.random.default_rng(seed)
    poisoned_idx = rng.choice(n, size=k, replace=False)

    y_poisoned = y.copy()
    y_poisoned[poisoned_idx] = 1 - y_poisoned[poisoned_idx]

    return {
        "X": X,
        "y_poisoned": y_poisoned,
        "poisoned_indices": poisoned_idx,
        "fraction_requested": fraction,
        "n_poisoned": k,
    }


def invert_continuous_targets(y: np.ndarray, fraction: float = 0.05, seed: int = 0,
                               scale: float | None = None) -> dict:
    """
    Regression-appropriate poisoning: for a random `fraction` of samples,
    reflects the target around the batch mean (y -> 2*mean - y) rather
    than flipping a binary label, which has no meaning for a continuous
    forecast target. `scale` optionally rescales the reflected deviation
    (e.g. scale=1.5 makes the poisoned points more extreme than a pure
    reflection) -- defaults to an exact reflection (scale=1.0 behavior)
    when None.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1); got {fraction}")

    n = len(y)
    k = int(n * fraction)
    rng = np.random.default_rng(seed)
    poisoned_idx = rng.choice(n, size=k, replace=False)

    y_poisoned = y.copy().astype(float)
    mean_y = float(np.mean(y))
    multiplier = 1.0 if scale is None else scale
    y_poisoned[poisoned_idx] = mean_y + multiplier * (mean_y - y[poisoned_idx])

    return {
        "y_poisoned": y_poisoned,
        "poisoned_indices": poisoned_idx,
        "fraction_requested": fraction,
        "n_poisoned": k,
    }

def inject_triggered_shift(X: np.ndarray, y: np.ndarray, fraction: float = 0.2, shift_sd: float = 3.0,
                           seed: int = 0, direction: np.ndarray | None = None) -> dict:
    """
    Feature-triggered target shift (a backdoor-style poisoning attack) for a continuous forecast target.

    WHY THIS EXISTS. `invert_continuous_targets` reflects a random subset of targets about their mean. For a
    target whose distribution is symmetric about its mean (a one-step price CHANGE is) and a head that has little
    signal to lose, that reflection leaves the training distribution unchanged, so a 5 % reflection moves the head's
    RMSE by ~1e-5 %. The attack was a near no-op, so H3 compared two conditions on an outcome that could not move.

    This attack adds a *systematic* dependence the head can learn: the top `fraction` rows (ranked by trigger score
    z = standardised_X @ u) have their target shifted upward by shift_sd * sd(y). The direction u is chosen
    adversarially as the gradient direction X.T @ residuals from a clean ridge fit, which guarantees the shift
    opposes the clean model and maximally perturbs predictions. A linear head fitted on the poisoned rows learns a
    spurious slope along u and carries it to clean data, raising clean-data RMSE consistently across seeds.

    The previous implementation used sign(z[idx] - median(z)) as a multiplier, which produced a mean-zero shift
    (~50% positive, ~50% negative per batch). That mean-zero property meant a ridge head saw no consistent
    adversarial direction, so degradation sign was random across seeds. This version fixes both issues: the
    direction u is data-driven (not random), and all selected rows are shifted in the same (+) direction.

    Returns y_poisoned, the poisoned indices, and the direction u (so a caller can log it).
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1); got {fraction}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Adversarial direction: fit a clean ridge head, then set u = X.T @ residuals.
    # This is the steepest-ascent direction for MSE given the clean fit -- guaranteed
    # to push predictions away from truth regardless of random seed.
    if direction is None:
        A = X.T @ X + 1.0 * np.eye(X.shape[1])
        w_clean = np.linalg.solve(A, X.T @ y)
        residuals = y - X @ w_clean
        direction = X.T @ residuals
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) or 1.0)

    # Score each sample by how strongly it aligns with the adversarial direction.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = ((X - mu) / sd) @ direction

    # Select the top-k rows (highest trigger score) and shift ALL of them upward.
    # Unidirectional shift is essential: a mean-zero shift (sign-mixed) gives a ridge
    # head no consistent adversarial slope to learn, so degradation becomes seed-random.
    k = int(n * fraction)
    idx = np.argpartition(z, -k)[-k:]
    y_p = y.copy()
    y_p[idx] += shift_sd * float(np.std(y))
    return {"y_poisoned": y_p, "poisoned_indices": idx, "fraction_requested": fraction,
            "n_poisoned": k, "shift_sd": float(shift_sd), "direction": direction}

