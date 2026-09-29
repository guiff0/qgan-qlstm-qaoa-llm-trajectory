import numpy as np
from src.quantum.qpca import QuantumPCA, QPCAConfig


def _data():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((3000, 2))
    return np.column_stack([f[:, 0], f[:, 0] + 0.3 * rng.standard_normal(3000),
                            f[:, 1], f[:, 1] + 0.4 * rng.standard_normal(3000),
                            rng.standard_normal(3000)])


def test_qpca_matches_classical_pca():
    res = QuantumPCA(QPCAConfig(n_components=2, n_clock=6)).fit(_data())
    assert res.eigenvalue_max_abs_error < 1 / 64          # within QPE resolution
    assert np.all(res.eigenvector_abs_overlap > 0.99)


def test_transform_shape():
    X = _data()
    res = QuantumPCA(QPCAConfig(n_components=2, n_clock=6)).fit(X)
    assert QuantumPCA.transform(X, res).shape == (len(X), 2)
