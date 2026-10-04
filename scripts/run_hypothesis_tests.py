"""
Reads results/all_results.csv and computes the actual H1-H5 statistical
tests (src/evaluation/statistical_tests.py -- every p-value/F-statistic
in the manuscript's Ch.4 had no corresponding computation anywhere in
the codebase before this), plus CQRS (metrics/resilience.py) per
quantum-model row. Writes results/hypothesis_test_results.json.

HONEST LIMITATION, not fixed by this script: every test below needs
MULTIPLE independent observations per condition -- a t-test comparing
one number to one number has no within-group variance to test. The
pipeline currently trains each model config exactly once (see TODO.md's
repeated-runs item), so with the data available today most of these
will report "insufficient data" rather than a real p-value. Still worth
running now: this is the harness that produces real numbers the moment
repeated runs exist, and it makes the current gap explicit and
quantified (exactly how many runs per condition exist right now)
instead of leaving it implicit.

    python -m scripts.run_hypothesis_tests
    python -m scripts.run_hypothesis_tests --results-csv results/all_results.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.evaluation.statistical_tests import (
    ancova, independent_ttest, pearson_correlation_with_ci,
)
from src.evaluation.mediation_pipelines import run_m2_dv2_mediation
from src.metrics.resilience import cqrs, qar

QGAN, CGAN = "QGAN-LLM", "Classical GAN-LLM"


def _safe(fn, *args, need_n=2, **kwargs):
    """Runs a test, reporting the "not enough data" case explicitly
    (naming exactly how many observations are missing) rather than
    letting scipy/statsmodels raise an opaque error, or -- worse --
    silently returning a p-value computed on a degenerate sample."""
    for a in args:
        if hasattr(a, "__len__") and len(a) < need_n:
            return {"error": f"insufficient data: need >= {need_n} observations, "
                              f"have {len(a)} -- repeated runs needed (see TODO.md)"}
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc)}


def test_h1_synthetic_data_fidelity(df: pd.DataFrame) -> dict:
    needed = {QGAN, CGAN}
    if not needed.issubset(set(df["model_type"])) or "fid" not in df.columns:
        return {"error": f"need both {QGAN!r} and {CGAN!r} rows with an 'fid' column"}
    sub = df[df["model_type"].isin(needed)].dropna(subset=["fid"])
    group = (sub["model_type"] == QGAN).astype(int).to_numpy()
    if "vix" in sub.columns and sub["vix"].notna().all() and len(sub) >= 4:
        result = _safe(ancova, sub["fid"].to_numpy(), group, sub["vix"].to_numpy(), need_n=4)
    else:
        a = sub.loc[sub["model_type"] == QGAN, "fid"].to_numpy()
        b = sub.loc[sub["model_type"] == CGAN, "fid"].to_numpy()
        result = _safe(independent_ttest, a, b, need_n=2)
        result["note"] = ("no usable 'vix' covariate in results/all_results.csv -- ran a plain "
                           "t-test instead of the ANCOVA Ch.3 specifies")
    return result


def test_h2_adversarial_resilience(df: pd.DataFrame) -> dict:
    cols = {"entanglement_entropy", "asr"}
    if not cols.issubset(df.columns):
        return {"error": "need entanglement_entropy and asr columns"}
    sub = df.dropna(subset=list(cols))
    return _safe(pearson_correlation_with_ci, sub["entanglement_entropy"].to_numpy(),
                 sub["asr"].to_numpy(), need_n=4)


def test_h3_poisoning_resistance(df: pd.DataFrame) -> dict:
    col = "poisoning_rmse_degradation_pct"
    if col not in df.columns:
        return {"error": f"need '{col}' column (see src/evaluation/poisoning_resistance.py)"}
    is_no_noise = df["model_type"].str.contains("no noise", case=False, na=False)
    no_noise = df.loc[is_no_noise, col].dropna().to_numpy()
    noisy = df.loc[~is_no_noise, col].dropna().to_numpy()
    result = _safe(independent_ttest, no_noise, noisy, need_n=2)
    result["interpretation"] = ("positive t (no_noise mean > noisy mean) supports Ha3: "
                                 "noise injection -> lower poisoning degradation")
    return result


def test_h4_quantum_advantage(df: pd.DataFrame) -> dict:
    needed = {QGAN, CGAN}
    if not needed.issubset(set(df["model_type"])):
        return {"error": f"need both {QGAN!r} and {CGAN!r} rows"}
    result = {}
    if "rmse" in df.columns:
        a_rmse = df.loc[df["model_type"] == QGAN, "rmse"].dropna().to_numpy()
        b_rmse = df.loc[df["model_type"] == CGAN, "rmse"].dropna().to_numpy()
        result["rmse"] = _safe(independent_ttest, a_rmse, b_rmse, need_n=2)
    else:
        result["rmse"] = {"error": "no 'rmse' column"}
    if "asr" in df.columns:
        a_asr = df.loc[df["model_type"] == QGAN, "asr"].dropna().to_numpy()
        b_asr = df.loc[df["model_type"] == CGAN, "asr"].dropna().to_numpy()
        result["asr"] = _safe(independent_ttest, a_asr, b_asr, need_n=2)
    else:
        result["asr"] = {"error": "no 'asr' column"}
    result["caveat"] = ("RMSE here now reflects the fixed one-step-ahead target "
                         "(see src/evaluation/one_step_ahead.py) rather than the old "
                         "same-row-leaked values -- numbers from before that fix are "
                         "not comparable to numbers generated after it")
    return result


def test_h5_tomography_correlation(df: pd.DataFrame) -> dict:
    cols = {"entanglement_entropy", "accuracy"}
    if not cols.issubset(df.columns):
        return {"error": "need entanglement_entropy and accuracy columns (from threat_detection.py)"}
    sub = df.dropna(subset=list(cols))
    result = _safe(pearson_correlation_with_ci, sub["entanglement_entropy"].to_numpy(),
                    sub["accuracy"].to_numpy(), need_n=4)
    result["note"] = ("'accuracy' here comes from the interim classical detector in "
                       "src/evaluation/threat_detection.py, not the manuscript's LLM-based "
                       "threat_scoring.py pathway -- see TODO.md's LLM-wiring item")
    return result


def test_m2_dv2_mediation(df: pd.DataFrame) -> dict:
    """M2->DV2: does entanglement level mediate model-configuration's
    effect on ASR category? See mediation_pipelines.py's UNIT-OF-ANALYSIS
    CAVEAT -- these rows are one per model CONFIG, not one per
    independent run, so this is still underpowered even when it runs
    without error; reported for completeness, not as a settled result."""
    if "asr" not in df.columns or "model_type" not in df.columns:
        return {"error": "need model_type and asr columns"}
    sub = df.dropna(subset=["asr"]).copy()
    if len(sub) < 4:
        return {"error": f"insufficient data: need >= 4 observations, have {len(sub)}"}
    sub["model_config"] = sub["model_type"].str.contains("QGAN|QLSTM|QAOA", case=False, na=False).astype(int)
    is_linear = sub["model_type"].str.contains("linear", case=False, na=False)
    sub["entanglement_topology"] = np.where(is_linear, "linear", "ring")
    try:
        result = run_m2_dv2_mediation(sub["model_config"].to_numpy(),
                                       sub["entanglement_topology"].tolist(),
                                       sub["asr"].to_numpy())
        result["n_observations"] = len(sub)
        result["caveat"] = ("rows here are one per model CONFIG (ablation), not one per "
                             "independent experimental run -- see mediation_pipelines.py's "
                             "UNIT-OF-ANALYSIS CAVEAT; treat as exploratory, not confirmatory")
        return result
    except Exception as exc:
        return {"error": str(exc)}


def test_qar(df: pd.DataFrame) -> dict:
    """QAR needs asr_clean (added this session -- see
    src/attacks/adversarial.py's compute_clean_asr) alongside the
    existing attacked asr, for both a classical and a quantum model."""
    needed = {QGAN, CGAN}
    cols = {"asr", "asr_clean"}
    if not needed.issubset(set(df["model_type"])) or not cols.issubset(df.columns):
        return {"error": f"need both {QGAN!r} and {CGAN!r} rows with 'asr' and 'asr_clean' columns"}
    sub = df.dropna(subset=list(cols))
    q = sub[sub["model_type"] == QGAN]
    c = sub[sub["model_type"] == CGAN]
    if len(q) == 0 or len(c) == 0:
        return {"error": "need at least one row of each model type with both columns populated"}
    try:
        result = {
            "qar": qar(
                asr_classical_clean=float(c["asr_clean"].mean()),
                asr_quantum_clean=float(q["asr_clean"].mean()),
                asr_classical_attacked=float(c["asr"].mean()),
                asr_quantum_attacked=float(q["asr"].mean()),
            ),
            "n_quantum_rows": len(q),
            "n_classical_rows": len(c),
        }
        if len(q) < 2 or len(c) < 2:
            result["caveat"] = "averaged over < 2 runs per side -- a point estimate, not yet a tested difference"
        return result
    except Exception as exc:
        return {"error": str(exc)}


def compute_cqrs_per_row(df: pd.DataFrame) -> dict:
    """QAR term omitted (NaN) -- needs a 'clean' (epsilon=0) ASR baseline
    this pipeline doesn't produce yet; cqrs() renormalizes over whatever's
    available (see metrics/resilience.py), same handling it already uses
    for a missing CTR."""
    need = {"qsfr", "eer", "nlcs"}
    if not need.issubset(df.columns):
        return {}
    rows = {}
    for _, row in df.dropna(subset=list(need)).iterrows():
        key = f"{row.get('model_type', 'unknown')}_{row.get('run_id', '')}"
        rows[key] = cqrs(row["qsfr"], row["eer"], row.get("ctr", float("nan")),
                          row["nlcs"], float("nan"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-csv", default="results/all_results.csv")
    ap.add_argument("--out", default="results/hypothesis_test_results.json")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        raise SystemExit(f"{args.results_csv} not found -- run the pipeline first.")
    df = pd.read_csv(args.results_csv)

    out = {
        "n_rows": len(df),
        "n_runs_per_model": df.groupby("model_type").size().to_dict() if "model_type" in df.columns else {},
        "H1_synthetic_data_fidelity": test_h1_synthetic_data_fidelity(df),
        "H2_adversarial_resilience": test_h2_adversarial_resilience(df),
        "H3_poisoning_resistance": test_h3_poisoning_resistance(df),
        "H4_quantum_advantage": test_h4_quantum_advantage(df),
        "H5_threat_detection_tomography_correlation": test_h5_tomography_correlation(df),
        "M2_DV2_mediation": test_m2_dv2_mediation(df),
        "QAR": test_qar(df),
        "CQRS_per_model": compute_cqrs_per_row(df),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
