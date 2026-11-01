"""
The controlled entanglement sweep: one place that defines the levels, names the
config entries, and reads the level back out of results/all_results.csv.

WHY THIS EXISTS
H2 / H5 / M2 are claims about entanglement as a *dose*. The earlier ablations
(20-qubit, 12-qubit, linear, no-noise) differ from each other in several ways at
once, so correlating entropy with ASR across them cannot separate entanglement
from everything else, and there are only 4-5 of them. The sweep instead holds
qubit count, noise and everything else fixed and changes ONLY the entangling
structure (topology and circuit depth), with several seeds per level, so each
(level, seed) run is one independent observation.

HOW THE LEVEL IS RECOVERED
Every QGAN-LLM ablation is written to all_results.csv with model_type == "QGAN-LLM"
(see run_identity.py), so the config is only recoverable from the run_id slug.
The sweep names are chosen so the slug carries the design:

    "QGAN-LLM (ent-L3 linear-d4)"  ->  run_id  qgan_llm_ent_l3_linear_d4_<UTC stamp>
                                        level 3, topology linear, depth 4

LEVEL ORDER (low -> high entangling-gate count per forward pass)
    n_qubits = n:  linear = n-1 CNOTs/layer, ring = n, full = n(n-1)/2.
    L0 none x4 (no entangling gates) | L1 linear x1 layer | L2 linear x2 | L3 linear x4 | L4 ring x4 | L5 full x2 | L6 full x4
At n = 12 that is 0, 11, 22, 44, 48, 132, 264 CNOTs. The ORDER is a design assumption;
the manipulation check reported by the hypothesis script (does the level really move the
MEASURED entanglement entropy?) is what tells you whether it held.

KNOWN CONFOUND: depth changes the parameter count as well as the entanglement
(n_layers * n_qubits * 3 weights), and topology changes trainability. Report the sweep as
"entangling structure", not as "entanglement and nothing else".
"""
from __future__ import annotations

import re
from typing import Optional

# (level, topology, n_layers)
LEVELS: list[tuple[int, str, int]] = [
    (0, "none", 4),      # zero-entanglement control: no CNOTs; same depth/params as L3 and L4
    (1, "linear", 1),
    (2, "linear", 2),
    (3, "linear", 4),
    (4, "ring", 4),
    (5, "full", 2),
    (6, "full", 4),
]

DEFAULT_N_QUBITS = 12          # same width as the existing "QGAN-LLM (12-qubit)" ablation; 20 is impractical x36 runs
MIN_RUNS_PER_LEVEL = 3         # fewer than this and a level's spread cannot be estimated
MIN_SWEEP_RUNS = 12            # below this the pooled-config fallback is used instead
MIN_SWEEP_LEVELS = 3

_SLUG = re.compile(r"(?:^|_)qgan_llm_ent_l(\d+)_(none|linear|ring|full)_d(\d+)(?:_\d{8}T\d{6}Z)?$")


def sweep_name(level: int, topology: str, depth: int) -> str:
    """The ablation / --model name for one level. Characters are restricted to what
    run_identity.model_slug maps cleanly (letters, digits, space, '-', parentheses)."""
    return f"QGAN-LLM (ent-L{level} {topology}-d{depth})"


def all_sweep_names() -> list[str]:
    return [sweep_name(*lv) for lv in LEVELS]


def cnots_per_forward(n_qubits: int, topology: str, depth: int) -> int:
    per_layer = {"none": 0, "linear": n_qubits - 1, "ring": n_qubits, "full": n_qubits * (n_qubits - 1) // 2}[topology]
    return per_layer * depth


def parse_sweep_config(run_id_or_slug: str) -> Optional[dict]:
    """{'level','topology','depth'} for a sweep run_id / slug, else None."""
    m = _SLUG.search(str(run_id_or_slug))
    if not m:
        return None
    return {"level": int(m.group(1)), "topology": m.group(2), "depth": int(m.group(3))}


def add_sweep_columns(df):
    """Copy of df with sweep_level / sweep_topology / sweep_depth (NaN for non-sweep rows)."""
    import numpy as np
    out = df.copy()
    if "run_id" not in out.columns:
        for c in ("sweep_level", "sweep_topology", "sweep_depth"):
            out[c] = np.nan
        return out
    parsed = out["run_id"].map(parse_sweep_config)
    out["sweep_level"] = parsed.map(lambda d: d["level"] if d else np.nan)
    out["sweep_topology"] = parsed.map(lambda d: d["topology"] if d else None)
    out["sweep_depth"] = parsed.map(lambda d: d["depth"] if d else np.nan)
    return out


def sweep_subset(df, required_cols, min_runs: int = MIN_SWEEP_RUNS, min_levels: int = MIN_SWEEP_LEVELS):
    """Sweep rows with all `required_cols` present, or None if there are too few / too few levels
    (the caller then falls back to the old pooled-config analysis and labels it as such)."""
    sw = add_sweep_columns(df)
    sw = sw[sw["sweep_level"].notna()]
    if sw.empty or not set(required_cols).issubset(sw.columns):
        return None
    sw = sw.dropna(subset=list(required_cols))
    if len(sw) < min_runs or sw["sweep_level"].nunique() < min_levels:
        return None
    return sw
