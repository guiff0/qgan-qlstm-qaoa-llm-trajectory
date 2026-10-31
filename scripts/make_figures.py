"""
Dissertation figures that are computed from saved run output (never from invented numbers).

    Figure 31  QPCA component loadings       <- results/qpca_results.json          (scripts/run_qpca.py)
    Figure 32  Pairwise entanglement map     <- results/entanglement_maps/*.json   (written by QGAN-LLM evaluate())

    python -m scripts.make_figures              # writes PNGs to results/figures/
    python -m scripts.make_figures --out-dir X

A figure whose input file is missing is skipped with a one-line reason; the script still exits 0 so it
can run as an always-on analysis step. Nothing here is a claim of quantum advantage: both figures
describe the simulated circuits themselves.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def figure_31(qpca_json: str, out_dir: str) -> str | None:
    if not os.path.exists(qpca_json):
        print(f"[figure 31] skipped: {qpca_json} not found (run `python -m scripts.run_qpca`, or the pipeline step run_qpca).")
        return None
    r = json.load(open(qpca_json))
    n_feat = int(r["n_features"])
    V = np.asarray(r["eigenvectors"], dtype=float)
    if V.shape[0] != n_feat and V.shape[1] == n_feat:
        V = V.T                                            # stored (k, n_features) -> (n_features, k)
    k = V.shape[1]
    qe = np.asarray(r["quantum_eigenvalues"], float)
    ce = np.asarray(r["classical_eigenvalues"], float)
    ev = np.asarray(r.get("explained_variance_ratio", []), float)
    plt = _plt()
    if plt is None:
        print("[figure 31] skipped: matplotlib is not installed (pip install matplotlib).")
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    im = a1.imshow(V, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    a1.set_xticks(range(k)); a1.set_xticklabels([f"PC{i+1}" + (f"\n{ev[i]*100:.1f}% var" if i < len(ev) else "")
                                               for i in range(k)])
    a1.set_yticks(range(n_feat)); a1.set_yticklabels([f"feature {i}" for i in range(n_feat)])
    for i in range(n_feat):
        for j in range(k):
            a1.text(j, i, f"{V[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    a1.set_title("Loadings (QPCA eigenvectors)")
    fig.colorbar(im, ax=a1, fraction=0.046)
    m = min(len(qe), len(ce))
    x = np.arange(m)
    a2.bar(x - 0.2, ce[:m], 0.4, label="classical PCA"); a2.bar(x + 0.2, qe[:m], 0.4, label="QPCA (simulated)")
    a2.set_xticks(x); a2.set_xticklabels([f"λ{i+1}" for i in range(m)]); a2.set_ylabel("eigenvalue")
    a2.set_title(f"Eigenvalues (max abs error {r.get('eigenvalue_max_abs_error', float('nan')):.2e})")
    a2.legend()
    fig.suptitle(f"Figure 31: QPCA component loadings  ({r['n_rows']} rows, {n_feat} features, "
                 f"{r['qubits_simulated']} simulated qubits)", fontsize=11)
    fig.text(0.5, 0.005, r.get("note", ""), ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = os.path.join(out_dir, "figure_31_qpca_loadings.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def figure_32(map_dir: str, out_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(map_dir, "*.json")))
    if not files:
        print(f"[figure 32] skipped: no maps in {map_dir} (they are written by QGAN-LLM evaluate(); "
              f"re-run a QGAN-LLM train_eval step).")
        return []
    maps = [json.load(open(f)) for f in files]
    plt = _plt()
    if plt is None:
        print("[figure 32] skipped: matplotlib is not installed (pip install matplotlib).")
        return []
    vmax = max(m["max_pairwise_mi"] for m in maps) or 1.0
    cols = min(3, len(maps)); rows = int(np.ceil(len(maps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    last = None
    for ax, m in zip(axes.ravel(), maps):
        mi = np.asarray(m["mutual_information"])
        ax.axis("on")
        last = ax.imshow(mi, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{m.get('model', '?')}\n{m.get('entanglement_topology', '')}, "
                     f"noise={m.get('noise_strength')}, seed {m.get('seed')}\nmean I={m['mean_pairwise_mi']:.3g} bits",
                     fontsize=9)
        ax.set_xlabel("qubit"); ax.set_ylabel("qubit")
    fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, label="mutual information (bits)")
    fig.suptitle("Figure 32: Entanglement map (pairwise quantum mutual information, simulated statevector)", fontsize=11, y=1.06)
    out = os.path.join(out_dir, "figure_32_entanglement_map.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return [out]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qpca-json", default="results/qpca_results.json")
    ap.add_argument("--map-dir", default="results/entanglement_maps")
    ap.add_argument("--out-dir", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    made = [p for p in [figure_31(a.qpca_json, a.out_dir)] if p] + figure_32(a.map_dir, a.out_dir)
    print("figures written:", made if made else "none")


if __name__ == "__main__":
    main()
