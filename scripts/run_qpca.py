"""Run QPCA (phase estimation + inverse QFT) on the processed feature matrix and
compare against classical PCA. Writes results/qpca_results.json.

    python -m scripts.run_qpca --features 5 --rows 20000 --clock 6

Qubits simulated = n_clock + 2*ceil(log2(features)); keep --features <= 16.
"""
import argparse, json, os
import numpy as np
from src.quantum.qpca import QuantumPCA, QPCAConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/X_train.npy")
    ap.add_argument("--features", type=int, default=5, help="first N columns (PCA-ordered) to analyze")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--clock", type=int, default=6)
    ap.add_argument("--components", type=int, default=2)
    ap.add_argument("--out", default="results/qpca_results.json")
    a = ap.parse_args()

    X = np.load(a.data, mmap_mode="r")[: a.rows, : a.features]
    res = QuantumPCA(QPCAConfig(n_components=a.components, n_clock=a.clock)).fit(np.asarray(X))
    out = {
        "n_rows": int(len(X)), "n_features": int(a.features), "n_clock": a.clock,
        "qubits_simulated": res.n_qubits_simulated,
        "quantum_eigenvalues": res.eigenvalues.tolist(),
        "classical_eigenvalues": res.classical_eigenvalues.tolist(),
        "eigenvalue_max_abs_error": res.eigenvalue_max_abs_error,
        "eigenvector_abs_overlap": res.eigenvector_abs_overlap.tolist(),
        "explained_variance_ratio": res.explained_variance_ratio.tolist(),
        "eigenvectors": res.eigenvectors.tolist(),
        "note": "Simulated QPCA on classical data features; NOT on quantum-state metrics.",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "eigenvectors"}, indent=2))


if __name__ == "__main__":
    main()
