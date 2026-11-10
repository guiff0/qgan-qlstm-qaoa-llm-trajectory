import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.simulate_entropy_dose import curve, half_entropy, state


def test_theta_zero_is_a_product_state_and_entropy_is_bounded():
    r = np.random.default_rng(0)
    n, nl = 6, 3
    x, w = r.normal(size=n), r.normal(size=n * nl * 3)
    assert half_entropy(state(n, nl, 0.0, x, w), n) < 1e-9
    s = half_entropy(state(n, nl, math.pi / 4, x, w), n)
    assert 0 < s <= n // 2 + 1e-9


def test_last_layer_rzz_is_skipped_and_curve_shape():
    r = np.random.default_rng(1)
    n = 6
    x, w = r.normal(size=n), r.normal(size=n * 3)
    assert abs(half_entropy(state(n, 1, 0.9, x, w), n)) < 1e-9          # one layer = last layer: no entangler
    R = curve(6, 3, [0, math.pi / 4], 0.5, draws=5)
    assert R.shape == (5, 2) and np.allclose(R[:, 0], 0, atol=1e-9) and (R[:, 1] > 0).all()
