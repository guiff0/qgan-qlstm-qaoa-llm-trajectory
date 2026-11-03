"""
Loss landscape of a TRAINED QGAN-LLM generator -> results/loss_landscape.json + results/figures/figure_loss_landscape.png

    python -m scripts.run_loss_landscape --checkpoint models/qgan_llm_<fp>.ckpt \
        [--n-qubits 12 --n-layers 4 --entanglement linear --noise-strength 0.0] [--batch 16 --grid 15 --span 0.3]

The loss is the generator's training loss  BCE(D(G(z)), 1) + alpha * MSE(G(z), x)  on ONE fixed batch of real rows
and ONE fixed noise draw, with the checkpoint's own discriminator frozen.  Overrides must match the config the
checkpoint was trained with (a shape mismatch aborts with a clear error).  Cost: grid*grid forward passes for the
slice plus ~2 gradient evaluations per Hessian iteration -- on the 20-qubit simulator use a small --grid first.

This characterises one fixed batch around one trained point.  It is NOT a statement that entanglement hardens the
model: with a single run per configuration there is nothing to compare across (see Chapter 4, RQ2).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def build_closures(ckpt_path: str, overrides: dict, batch: int, data_path: str, seed: int):
    import torch
    from torch import nn
    from src.baselines.qgan_llm import QGANLLM
    from src.utils.config import load_config

    cfg = {**load_config()["qgan_llm"], **{k: v for k, v in overrides.items() if v is not None}}
    m = QGANLLM(cfg, seed=seed)
    m.build()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.generator.load_state_dict(ck["generator"])
    m.discriminator.load_state_dict(ck["discriminator"])
    m.generator.eval(); m.discriminator.eval()
    for p in m.discriminator.parameters():
        p.requires_grad_(False)

    X = np.load(data_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    xb = torch.from_numpy(np.ascontiguousarray(X[rng.choice(len(X), size=batch, replace=False)], dtype=np.float32))
    nf = cfg["n_features"]
    z = torch.randn(batch, nf, generator=torch.Generator().manual_seed(seed))
    bce, mse, alpha = nn.BCELoss(), nn.MSELoss(), float(cfg["alpha"])
    params = list(m.generator.parameters())
    vec = lambda: torch.nn.utils.parameters_to_vector(params)

    def _set(theta):
        torch.nn.utils.vector_to_parameters(torch.as_tensor(theta, dtype=vec().dtype), params)

    def _loss_t():
        torch.manual_seed(seed)                      # fix any noise injected inside the circuit -> deterministic in theta
        fake = m.generator(z)
        return bce(m.discriminator(fake), torch.ones(batch, 1)) + alpha * mse(fake, xb)

    theta0 = vec().detach().numpy().astype(np.float64).copy()

    def loss_fn(theta):
        _set(theta)
        with torch.no_grad():
            return float(_loss_t())

    def grad_fn(theta):
        _set(theta)
        for p in params:
            p.grad = None
        _loss_t().backward()
        return torch.nn.utils.parameters_to_vector([p.grad for p in params]).detach().numpy().astype(np.float64)

    return theta0, loss_fn, grad_fn, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/processed/X_train.npy")
    ap.add_argument("--n-qubits", type=int); ap.add_argument("--n-layers", type=int)
    ap.add_argument("--entanglement"); ap.add_argument("--noise-strength", type=float)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--span", type=float, default=0.3)
    ap.add_argument("--hess-iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/loss_landscape.json")
    ap.add_argument("--out-dir", default="results/figures")
    a = ap.parse_args()

    from src.evaluation import loss_landscape as ll
    theta, loss_fn, grad_fn, cfg = build_closures(
        a.checkpoint, {"n_qubits": a.n_qubits, "n_layers": a.n_layers, "entanglement": a.entanglement,
                       "noise_strength": a.noise_strength}, a.batch, a.data, a.seed)
    rng = np.random.default_rng(a.seed)
    d1, d2 = ll.random_direction(theta, rng), ll.random_direction(theta, rng)
    out = {"checkpoint": os.path.basename(a.checkpoint), "n_params": int(theta.size), "seed": a.seed, "batch": a.batch,
           "config": {k: cfg[k] for k in ("n_qubits", "n_layers", "entanglement", "noise_strength", "alpha")},
           "slice": ll.slice_2d(loss_fn, theta, d1, d2, span=a.span, n=a.grid),
           "line": ll.roughness(loss_fn, theta, d1, span=a.span),
           "hessian": ll.hessian_top_eig(grad_fn, theta, iters=a.hess_iters, seed=a.seed),
           "hutchinson": ll.hutchinson_trace(grad_fn, theta, n_probes=6, seed=a.seed),
           "note": "One fixed batch, one trained point, random directions scaled to ||theta||; no condition number or "
                   "local-minima density is computed."}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("n_params", "hessian", "hutchinson")}, indent=2)[:1500])

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib missing: JSON written, figure skipped."); return
    s = out["slice"]; ax = np.asarray(s["axis"]); Z = np.asarray(s["loss"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    cs = a1.contourf(ax, ax, Z.T, levels=20, cmap="viridis"); fig.colorbar(cs, ax=a1, label="generator loss")
    a1.plot([0], [0], "r*", ms=12); a1.set_xlabel("direction 1 (fraction of ||θ||)"); a1.set_ylabel("direction 2")
    a1.set_title("2-D slice around the trained generator")
    a2.plot(out["line"]["t"], out["line"]["loss"]); a2.set_xlabel("step along direction 1"); a2.set_ylabel("generator loss")
    a2.set_title(f"1-D line; top Hessian eig = {out['hessian']['top_eigenvalue']:.3g}")
    fig.suptitle("Loss landscape of one trained QGAN generator (fixed batch; simulated circuit)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(a.out_dir, exist_ok=True)
    fig.savefig(os.path.join(a.out_dir, "figure_loss_landscape.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
