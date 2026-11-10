"""How many runs does 'no practically relevant correlation' need? Descriptive, never a decision rule.

H2b / H5b ask whether |r| < r0 (0.30). The TOST on r passes only when the 90 % Fisher-z CI of r lies inside +/- r0.
With r_hat near 0 that takes n ~ 32; with r_hat = -0.085 it takes n ~ 57; with r_hat = -0.15, n ~ 111. The required n
grows steeply with |r_hat|, so a reported number is a planning estimate conditional on the observed r_hat, not a promise.
"""
from __future__ import annotations

from math import atanh, sqrt

from scipy.stats import norm


def runs_needed_for_equivalence(r_hat: float, r0: float = 0.30, alpha: float = 0.05, n_max: int = 5000):
    """Smallest n for which a correlation equal to `r_hat` would be declared equivalent to 0 within +/- r0
    (90 % CI for alpha = .05). None if |r_hat| >= r0 or no n <= n_max works."""
    if abs(r_hat) >= r0:
        return None
    z, z0, crit = abs(atanh(r_hat)), atanh(r0), norm.ppf(1 - alpha)
    for n in range(5, n_max):
        if z + crit / sqrt(n - 3) < z0:
            return n
    return None
