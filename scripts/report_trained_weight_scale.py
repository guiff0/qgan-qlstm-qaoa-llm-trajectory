"""How big are the TRAINED circuit weights? The entropy dose only rises monotonically with theta when the rotations are large enough.

    python -m scripts.report_trained_weight_scale [--models-dir models] [--simulate]

Reads every models/qgan_llm_*.ckpt (needs torch), prints the spread of the trained circuit parameters (generator `theta`), and with
--simulate runs scripts.simulate_entropy_dose at the median trained scale for the study geometry. Pre-Stage-1 screen from the v2
pre-registration: if the median trained weight SD is below 0.5 the simulated dose curve is not reliably increasing and Stage 1 is
expected to fail; do not spend pilot runs, redesign first (and amend the pre-registration).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def theta_of(ckpt: dict):
    sd = ckpt.get("generator") or {}
    for k, v in sd.items():
        if k.split(".")[-1] == "theta":
            return np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=float).ravel()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--n-qubits", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    import torch
    rows = []
    for p in sorted(glob.glob(os.path.join(a.models_dir, "qgan_llm_*.ckpt"))):
        try:
            th = theta_of(torch.load(p, map_location="cpu", weights_only=False))
        except Exception as exc:  # noqa: BLE001
            print(f"  {os.path.basename(p)}: unreadable ({exc})"); continue
        if th is None:
            continue
        rows.append((os.path.basename(p), len(th), float(th.std()), float(np.abs(th).mean()), float(np.abs(th).max())))
    if not rows:
        print("no readable qgan_llm checkpoints with a `theta` parameter"); return 2
    print(f"{'checkpoint':<34}{'n_params':>9}{'std':>9}{'mean|t|':>9}{'max|t|':>9}")
    for r in rows:
        print(f"{r[0]:<34}{r[1]:>9}{r[2]:>9.3f}{r[3]:>9.3f}{r[4]:>9.3f}")
    med = float(np.median([r[2] for r in rows]))
    print(f"\nmedian trained weight SD = {med:.3f}   ({'>= 0.5: dose can be monotone' if med >= 0.5 else '< 0.5: simulated dose curve NOT reliably increasing'})")
    if a.simulate:
        import math
        from scripts.simulate_entropy_dose import curve
        th = [0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2]
        R = curve(a.n_qubits, a.layers, th, med, 30)
        print("simulated mean S_E at the median trained scale:", np.round(R.mean(0), 3),
              "| mean curve increasing:", bool((np.diff(R.mean(0)) > 0).all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
