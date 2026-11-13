"""
Reads results/all_results.csv and computes the actual H1-H5 statistical
tests (src/evaluation/statistical_tests.py -- every p-value/F-statistic
in the manuscript's Ch.4 had no corresponding computation anywhere in
the codebase before this), plus CQRS (metrics/resilience.py) per
quantum-model row. Writes results/hypothesis_test_results.json.

STATUS VALUES (see the "summary" block of the output and the printed table):
    ok                       a real test ran on >= 2 independent runs per condition
    exploratory_single_case  one condition has a single run; a Crawford-Howell single-case t-test
                             ran instead (weaker; not confirmatory)
    insufficient_data        a condition has no usable run; "how_to_fix" gives the command
    degenerate_by_design     every observation is identical because the quantity does not depend
                             on the model under test (e.g. forecast-head RMSE in the GAN models);
                             more seeds cannot fix this
Observations are counted per (config, seed): re-evaluating one checkpoint is not a new run, a new
seed is. Use --dedupe config for the old one-row-per-config behaviour.

Safe to run on its own at any time, with or without the rest of the pipeline:
    python -m scripts.run_hypothesis_tests
    python -m scripts.run_hypothesis_tests --results-csv results/all_results.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os

import numpy as np
import pandas as pd

from src.evaluation.statistical_tests import (
    ancova, crawford_howell, independent_ttest, pearson_correlation_with_ci,
    tost_correlation, tost_equivalence,
)
from src.evaluation.mediation_pipelines import run_m2_dv2_mediation
from src.metrics.resilience import cqrs, qar
from src.utils.run_identity import config_of_run_id, latest_per_config, latest_per_config_seed
from src.evaluation.advantage_diagnostics import find_degenerate_groups
from src.evaluation.study_status import format_budget, hypothesis_budget, study_status
from scripts import snapshot_results as snap
from src.evaluation.arm_ledger import build_ledger, format_ledger
from src.evaluation.verdicts import compute_verdicts, format_table
from src.evaluation.entanglement_design import MIN_RUNS_PER_LEVEL, sweep_subset
from src.evaluation.mediation import mediation_analysis
from src.utils.run_identity import config_of_run_id
from src.utils.require_results import require_results
from src.utils import experiment as exp
from src.evaluation.paired_tests import paired_by_seed
from src.evaluation import analysis_plan as aplan
from src.evaluation.decidability import detector_gate, poisoning_gate
from src.evaluation.equivalence_power import runs_needed_for_equivalence

QGAN, CGAN = "QGAN-LLM", "Classical GAN-LLM"
PAIRED_ENABLED = True       # main() turns this off when several experiments are pooled (seeds would be ambiguous)


# ---- Equivalence (TOST) margins: the smallest difference that would matter. FIX THESE BEFORE the confirmatory
# runs and record them in HYPOTHESES_REVISED.md; a margin picked after seeing results proves nothing.
#   ("abs", x)              x in the metric's own units
#   ("rel_ref", f)          f x |mean of the reference (classical) group|   -- reference-dependent
#   ("rel_persistence", f)  f x mean persistence RMSE of the rows used      -- model-independent scale
EQUIV_MARGINS: dict = {
    "fid": ("rel_ref", 0.10),
    "rmse": ("rel_persistence", 0.02),
    "asr": ("abs", 5.0),                  # percentage points
    "poisoning": ("abs", 5.0),            # percentage points of RMSE degradation
    "r0": ("abs", 0.30),                  # smallest correlation of interest (H2/H5)
}
EQUIV_ALPHA = 0.05


def _margin_value(key: str, ref_mean: float, persistence: float | None = None) -> tuple[float, str]:
    kind, x = EQUIV_MARGINS[key]
    if kind == "abs":
        return float(x), f"{x:g} (absolute)"
    if kind == "rel_persistence" and persistence is not None and np.isfinite(persistence) and persistence > 0:
        return float(x) * float(persistence), f"{x:g} x persistence RMSE ({persistence:.6g})"
    return float(x) * abs(float(ref_mean)), f"{x:g} x |reference mean| ({abs(ref_mean):.6g})"


def _equivalence(a, b, key: str, persistence: float | None = None) -> dict | None:
    """TOST of |mean(a) - mean(b)| < margin; None when either group has < 2 finite values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    margin, how = _margin_value(key, float(np.mean(b)), persistence)
    try:
        out = tost_equivalence(a, b, margin, EQUIV_ALPHA)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    out.update({"metric": key, "margin_rule": how,
                "reading": ("EQUIVALENT: the true difference lies within +/- margin" if out["equivalent"] else
                            "equivalence NOT shown (not the same as 'different'; the data may simply be too noisy)")})
    return out


REPEAT_CMD = 'python -m scripts.run_repeated_experiment --model "{model}" --n-runs {n} --base-seed 43'


def _insufficient(n_per_group: dict, need: int = 2, how_to_fix: str = "") -> dict:
    """Structured "not enough runs" result: says which group is short and what to run."""
    short = {g: n for g, n in n_per_group.items() if n < need}
    return {"status": "insufficient_data",
            "message": (f"insufficient data: need >= {need} observations per group, have "
                        f"{n_per_group} -- repeated runs needed"),
            "short_groups": short, "how_to_fix": how_to_fix}


def _safe(fn, *args, need_n=2, **kwargs):
    """Runs a test; reports "not enough data" explicitly instead of letting scipy raise an opaque
    error or silently returning a p-value computed on a degenerate sample."""
    for a in args:
        if hasattr(a, "__len__") and len(a) < need_n:
            return _insufficient({"observations": len(a)}, need_n)
    try:
        out = fn(*args, **kwargs)
        out.setdefault("status", "ok")
        return out
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _compare_two(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str, how_to_fix: str = "") -> dict:
    """Two-condition comparison that degrades gracefully:
      n_a >= 2 and n_b >= 2 -> Welch t-test                           status "ok"
      one side has exactly 1 -> Crawford-Howell single-case t-test    status "exploratory_single_case"
      otherwise             -> structured insufficient-data result    status "insufficient_data"
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n = {name_a: len(a), name_b: len(b)}
    if len(a) >= 2 and len(b) >= 2:
        res = _safe(independent_ttest, a, b)
        res.setdefault("n_per_group", n)   # verdicts.py reads this; without it every verdict was "< 3 runs"
        return res
    if len(a) == 1 and len(b) >= 2:
        single, many, single_name, many_name = a[0], b, name_a, name_b
    elif len(b) == 1 and len(a) >= 2:
        single, many, single_name, many_name = b[0], a, name_b, name_a
    else:
        return _insufficient(n, 2, how_to_fix)
    try:
        res = crawford_howell(single, many)
    except Exception as exc:
        return {"status": "insufficient_data", "message": f"insufficient data: {exc}",
                "n_per_group": n, "how_to_fix": how_to_fix}
    res.update({"status": "exploratory_single_case", "single_condition": single_name,
                "sample_condition": many_name, "n_per_group": n,
                "note": (f"{single_name} has ONE run, so a two-sample test is impossible. This "
                         f"Crawford-Howell test asks whether that single run lies outside the spread "
                         f"of {many_name}; it is exploratory, not confirmatory."),
                "how_to_fix": how_to_fix})
    return res


_PRIMARY_SLUG = {"QGAN-LLM": "qgan_llm", "Classical GAN-LLM": "classical_gan_llm"}
_NO_NOISE_SLUG = "qgan_llm_no_noise"


def _primary_rows(df: pd.DataFrame, model_type: str, slug: str | None = None) -> tuple[pd.DataFrame, bool]:
    """Rows of the PRIMARY config of `model_type`, plus whether that restriction actually applied.

    Every QGAN-LLM ablation (no noise, 12-qubit, linear, the entanglement-sweep levels) is stored with
    model_type == "QGAN-LLM"; only the run_id slug tells them apart. Without this filter they are
    averaged into 'the QGAN' and the comparison is not a comparison of one condition. A results file with
    no run_id, or with no row of the primary config, falls back to every row of the model_type and says so
    (second return value False) so the caller labels the result as pooled/exploratory."""
    rows = df[df["model_type"] == model_type]
    if "run_id" in rows.columns and len(rows):
        ok = rows["run_id"].map(config_of_run_id) == (slug or _PRIMARY_SLUG[model_type])
        if ok.any():
            return rows[ok], True
    return rows, False


def _n_configs(rows: pd.DataFrame):
    return int(rows["run_id"].map(config_of_run_id).nunique()) if "run_id" in rows.columns else None


def test_h1_synthetic_data_fidelity(df: pd.DataFrame) -> dict:
    needed = {QGAN, CGAN}
    if not needed.issubset(set(df["model_type"])) or "fid" not in df.columns:
        return {"error": f"need both {QGAN!r} and {CGAN!r} rows with an 'fid' column"}
    df = df.dropna(subset=["fid"])
    # One condition per side: the primary config of each model. QGAN ablations / sweep levels are NOT
    # repeats of the primary QGAN-LLM and must not be pooled into it (see _primary_rows).
    q_rows, q_primary = _primary_rows(df, QGAN)
    c_rows, c_primary = _primary_rows(df, CGAN)
    sub = pd.concat([q_rows, c_rows])
    group = (sub["model_type"] == QGAN).astype(int).to_numpy()
    fix = REPEAT_CMD.format(model=CGAN, n=2)
    if ("vix" in sub.columns and sub["vix"].notna().all() and len(sub) >= 4
            and len(q_rows) >= 2 and len(c_rows) >= 2):
        result = _safe(ancova, sub["fid"].to_numpy(), group, sub["vix"].to_numpy(), need_n=4)
        result.setdefault("n_per_group", {QGAN: int(len(q_rows)), CGAN: int(len(c_rows))})
    else:
        result = _compare_two(q_rows["fid"].to_numpy(), c_rows["fid"].to_numpy(), QGAN, CGAN, fix)
        result["note_covariate"] = ("no usable 'vix' covariate (or < 2 runs per side) in "
                                    "results/all_results.csv -- ran a plain comparison instead of "
                                    "the ANCOVA Ch.3 specifies")
    result["equivalence"] = _equivalence(q_rows["fid"].to_numpy(), c_rows["fid"].to_numpy(), "fid")
    if PAIRED_ENABLED and len(q_rows) and len(c_rows):
        result["paired_by_seed"] = paired_by_seed(
            q_rows, c_rows, "fid", margin=_margin_value("fid", float(c_rows["fid"].mean()))[0])
    n_q, n_c = _n_configs(q_rows), _n_configs(c_rows)
    result["configs_used"] = {QGAN: _PRIMARY_SLUG[QGAN] if q_primary else "all rows (no primary-config rows found)",
                              CGAN: _PRIMARY_SLUG[CGAN] if c_primary else "all rows (no primary-config rows found)"}
    result["n_per_group_used"] = {QGAN: int(len(q_rows)), CGAN: int(len(c_rows))}
    # True only if rows of different configs were really mixed within one side.
    result["pooled_qgan_configs"] = (n_q > 1) if n_q is not None else None
    result["pooled_classical_configs"] = (n_c > 1) if n_c is not None else None
    return result


def _sweep_design_info(sw: pd.DataFrame, ycol: str) -> dict:
    """Design diagnostics for the controlled entanglement sweep (src/evaluation/entanglement_design.py)."""
    from scipy import stats
    per_level = sw.groupby("sweep_level").size()
    info = {"design": "controlled_sweep", "unit": "run",
            "n_levels": int(per_level.size),
            "runs_per_level": {int(k): int(v) for k, v in per_level.items()},
            "min_runs_per_level": int(per_level.min())}
    # Manipulation check: does the designed level actually move the MEASURED entropy? If not, the
    # sweep did not manipulate entanglement and a null here says nothing about H2/H5.
    rho_e, p_e = stats.spearmanr(sw["sweep_level"], sw["entanglement_entropy"])
    info["manipulation_check_level_vs_entropy"] = {"spearman_rho": float(rho_e), "p_value": float(p_e)}
    # The designed level is an ORDINAL IV only if measured entropy rises with it. Report the per-level means so a
    # non-monotone ladder (e.g. full-d4 below ring-d4) is visible instead of being averaged into one rho.
    _means = sw.groupby("sweep_level")["entanglement_entropy"].mean()
    info["entropy_by_level"] = {int(k): float(v) for k, v in _means.items()}
    info["entropy_ladder_monotone"] = bool((_means.diff().dropna() > 0).all())
    # Dose-response against the DESIGNED level (does not depend on the measured entropy).
    rho_y, p_y = stats.spearmanr(sw["sweep_level"], sw[ycol])
    info["dose_response_level_vs_outcome"] = {"outcome": ycol, "spearman_rho": float(rho_y), "p_value": float(p_y)}
    return info


def _r_equivalence(res: dict) -> dict | None:
    """'No practically relevant correlation': is |true r| < r0? (TOST on Fisher z; see tost_correlation)"""
    try:
        out = tost_correlation(res["r"], int(res["df"]) + 2, EQUIV_MARGINS["r0"][1], EQUIV_ALPHA)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    out["reading"] = ("EQUIVALENT TO ZERO: |r| < r0 is supported" if out["equivalent"] else
                      "no-relevant-correlation NOT shown (CI too wide or r too large)")
    return out


def _entropy_correlation(df: pd.DataFrame, ycol: str) -> dict:
    """Pearson r between measured entanglement entropy and `ycol`.

    Uses the controlled entanglement sweep when it has enough runs (>= 12 runs over >= 3 levels):
    every point is then one independent (level, seed) run with everything but the entangling
    structure held fixed. Otherwise falls back to the old pooled-config analysis, labelled
    design='pooled_configs' / unit='config' so the verdict layer treats it as exploratory."""
    cols = {"entanglement_entropy", ycol}
    sw = sweep_subset(df, cols) if "run_id" in df.columns else None
    if sw is not None:
        result = _safe(pearson_correlation_with_ci, sw["entanglement_entropy"].to_numpy(),
                       sw[ycol].to_numpy(), need_n=4)
        if "r" in result:
            result.update(_sweep_design_info(sw, ycol))
            result["equivalence_r"] = _r_equivalence(result)
            # planning aid only (not a decision rule): runs the sweep would need if r stayed where it is
            result["equivalence_r"]["runs_needed_if_r_unchanged"] = runs_needed_for_equivalence(
                result["r"], result["equivalence_r"].get("r0", 0.3))
        return result
    sub = df.dropna(subset=list(cols))
    result = _safe(pearson_correlation_with_ci, sub["entanglement_entropy"].to_numpy(),
                   sub[ycol].to_numpy(), need_n=4)
    if "r" in result:
        result.update({"design": "pooled_configs", "unit": "config"})
        result["equivalence_r"] = _r_equivalence(result)
    return result


def test_h2_adversarial_resilience(df: pd.DataFrame) -> dict:
    cols = {"entanglement_entropy", "asr"}
    if not cols.issubset(df.columns):
        return {"error": "need entanglement_entropy and asr columns"}
    sub = df.dropna(subset=list(cols))
    if "asr_baseline" in sub.columns:
        sub = sub[sub["asr_baseline"] == "last_close"]  # legacy rows used a PCA component
    return _entropy_correlation(sub, "asr")


def test_h3_poisoning_resistance(df: pd.DataFrame) -> dict:
    """no-noise vs noisy poisoning degradation.

    Clean design: the primary QGAN-LLM config (noise_strength > 0) against the 'QGAN-LLM (no noise)' ablation,
    which overrides noise_strength ONLY (config/default_config.yaml). Every other QGAN-LLM variant (12-qubit,
    linear topology, entanglement-sweep levels) also has noise > 0 but differs in qubit count / topology / depth,
    so pooling them into 'noisy' confounds the noise effect. Only when one of the two configs is missing does this
    fall back to the old pooled split, and then it carries a confound_warning (the verdict layer downgrades it)."""
    col = "poisoning_rmse_degradation_pct"
    if col not in df.columns:
        return {"error": f"need '{col}' column (see src/evaluation/poisoning_resistance.py)"}
    # Group by the recorded noise_strength (model_type is "QGAN-LLM" for every
    # ablation config and never contains "no noise").
    if "noise_strength" not in df.columns:
        return {"error": "need a 'noise_strength' column"}
    q = df[(df["model_type"] == QGAN) & df["noise_strength"].notna()]
    fix = REPEAT_CMD.format(model="QGAN-LLM (no noise)", n=2)
    if "run_id" in q.columns:
        cfg = q["run_id"].map(config_of_run_id)
        nn_rows = q[(cfg == _NO_NOISE_SLUG) & (q["noise_strength"] == 0)]
        noisy_rows = q[(cfg == _PRIMARY_SLUG[QGAN]) & (q["noise_strength"] > 0)]
        if len(nn_rows) and len(noisy_rows):
            result = _compare_two(nn_rows[col].dropna().to_numpy(), noisy_rows[col].dropna().to_numpy(),
                                  "no_noise", "noisy", fix)
            result["design"] = "matched_config_noise_only"
            result["equivalence"] = _equivalence(nn_rows[col].to_numpy(), noisy_rows[col].to_numpy(), "poisoning")
            result["n_per_group_used"] = {"no_noise": int(nn_rows[col].notna().sum()),
                                          "noisy": int(noisy_rows[col].notna().sum())}
            result["configs_used"] = {"no_noise": _NO_NOISE_SLUG, "noisy": _PRIMARY_SLUG[QGAN]}
            result["interpretation"] = ("positive t (no_noise mean > noisy mean) supports Ha3: "
                                        "noise injection -> lower poisoning degradation")
            both = pd.concat([nn_rows, noisy_rows])
            pc = both["poisoning_pc_degradation_pct"].dropna().to_numpy() if "poisoning_pc_degradation_pct" in both.columns else None
            result["decidability"] = poisoning_gate(
                list(nn_rows[col].dropna()) + list(noisy_rows[col].dropna()), EQUIV_MARGINS["poisoning"][1],
                positive_control=pc if pc is not None and len(pc) else None)
            if pc is not None and len(pc):
                result["positive_control"] = {"median_degradation_pct": float(pd.Series(pc).median()), "n_runs": int(len(pc)),
                                              "protocol": str(both.get("poisoning_protocol", pd.Series(["legacy"])).iloc[0])}
            if PAIRED_ENABLED:
                result["paired_by_seed"] = paired_by_seed(nn_rows, noisy_rows, col)
            return result
    # No matched pair in this design: NO pooled / cross-design fallback (reviewer issue 4). A no-noise run from another
    # design (e.g. a head_only pilot) is not a repeat of the generator-on-path condition, and pooling 12-qubit, linear
    # or sweep configs into 'noisy' confounds the noise effect. The hypothesis is untestable until the matched run exists.
    n_nn = int(len(q[(q["run_id"].map(config_of_run_id) == _NO_NOISE_SLUG) & (q["noise_strength"] == 0)])) if "run_id" in q.columns else 0
    n_ny = int(len(q[(q["run_id"].map(config_of_run_id) == _PRIMARY_SLUG[QGAN]) & (q["noise_strength"] > 0)])) if "run_id" in q.columns else 0
    result = _insufficient({"no_noise": n_nn, "noisy": n_ny}, 1,
                           "Run the matched 'QGAN-LLM (no noise)' condition in the same design (same seeds, forecast_path generator).")
    result["design"] = "matched_pair_missing"
    result["untestable_reason"] = ("the matched no-noise condition has no completed run in the analysed design; "
                                   "pooled and cross-design comparisons are deliberately not made")
    return result


_DEGENERATE_WHY = ("In QGAN-LLM and Classical GAN-LLM the forecast head is a linear layer fitted in closed "
                   "form on REAL rows only; the generator is never on the forecast path. More seeds "
                   "cannot create a difference. A quantum-vs-classical RMSE/ASR comparison needs the "
                   "generator (or its output) to feed the forecast -- see the Ch.4 diagnosis.")


def _h4_rows(df: pd.DataFrame, classical_slug: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Rows for the H4 comparison.

    * Only the PRIMARY config of each model: every QGAN-LLM ablation (no noise, 12-qubit, the entanglement
      sweep levels) is stored with model_type == "QGAN-LLM", so without this the sweep's 36 runs would be
      averaged into 'the QGAN'. The primary config is recognised by its run_id slug; a results file with no
      run_id (or no primary rows) falls back to all rows of the model_type, as before.
    * Only rows where the generator is ON the forecast path (forecast_path == 'generator') when BOTH models
      have such rows. Otherwise the comparison is the legacy head-only one, in which the forecast cannot depend
      on the generator at all (labelled so the verdict layer and the reader know).
    """
    q, c = _primary_rows(df, QGAN)[0], _primary_rows(df, CGAN, classical_slug)[0]
    if "forecast_path" in df.columns:
        qg, cg = q[q["forecast_path"] == "generator"], c[c["forecast_path"] == "generator"]
        if len(qg) and len(cg):
            return qg, cg, "generator_on_forecast_path"
    return q, c, "head_only_or_mixed"


def test_h4_quantum_advantage(df: pd.DataFrame, classical_slug: str | None = None) -> dict:
    needed = {QGAN, CGAN}
    if not needed.issubset(set(df["model_type"])):
        return {"error": f"need both {QGAN!r} and {CGAN!r} rows"}
    q_rows, c_rows, forecast_design = _h4_rows(df, classical_slug)
    result = {"forecast_design": forecast_design,
              "classical_comparator": classical_slug or _PRIMARY_SLUG[CGAN],
              "n_runs": {QGAN: int(len(q_rows)), CGAN: int(len(c_rows))}}
    if "rmse" in df.columns:
        a_rmse = q_rows["rmse"].dropna().to_numpy()
        b_rmse = c_rows["rmse"].dropna().to_numpy()
        _deg = find_degenerate_groups({"quantum": a_rmse, "classical": b_rmse})
        if _deg["degenerate"]:
            # Identical RMSE in both groups means the forecast does not depend on the
            # quantum circuit (GAN-LLM head is fitted on real rows only). Say so
            # instead of emitting a NaN t-statistic.
            result["rmse"] = {"status": "degenerate_by_design", "reason": _deg["reason"],
                              "why": _DEGENERATE_WHY,
                              "mean_a": float(np.mean(a_rmse)) if len(a_rmse) else None,
                              "mean_b": float(np.mean(b_rmse)) if len(b_rmse) else None}
        else:
            result["rmse"] = _compare_two(a_rmse, b_rmse, QGAN, CGAN, REPEAT_CMD.format(model=QGAN, n=2))
            pers = (float(pd.concat([q_rows, c_rows])["persistence_rmse"].mean())
                    if "persistence_rmse" in q_rows.columns else None)
            result["rmse"]["equivalence"] = _equivalence(a_rmse, b_rmse, "rmse", pers)
            if PAIRED_ENABLED:
                result["rmse"]["paired_by_seed"] = paired_by_seed(
                    q_rows, c_rows, "rmse", margin=_margin_value("rmse", float(np.mean(b_rmse)), pers)[0])
    else:
        result["rmse"] = {"error": "no 'rmse' column"}
    if "asr" in df.columns:
        a_asr = q_rows["asr"].dropna().to_numpy()
        b_asr = c_rows["asr"].dropna().to_numpy()
        _dega = find_degenerate_groups({"quantum": a_asr, "classical": b_asr})
        if _dega["degenerate"]:
            result["asr"] = {"status": "degenerate_by_design", "reason": _dega["reason"],
                             "why": _DEGENERATE_WHY}
        else:
            result["asr"] = _compare_two(a_asr, b_asr, QGAN, CGAN, REPEAT_CMD.format(model=QGAN, n=2))
            result["asr"]["equivalence"] = _equivalence(a_asr, b_asr, "asr")
            if PAIRED_ENABLED:
                result["asr"]["paired_by_seed"] = paired_by_seed(
                    q_rows, c_rows, "asr", margin=_margin_value("asr", float(np.mean(b_asr)))[0])
    else:
        result["asr"] = {"error": "no 'asr' column"}
    if forecast_design == "generator_on_forecast_path":
        result["note"] = ("forecast = linear([X, G(X)]) for BOTH models, same rows, same head, same evaluation "
                          "subsample (src/utils/generator_path.py): a gap is attributable to the generator. "
                          "It is a claim about this feature-map design, not about quantum models in general.")
    result["caveat"] = ("RMSE here now reflects the fixed one-step-ahead target "
                         "(see src/evaluation/one_step_ahead.py) rather than the old "
                         "same-row-leaked values -- numbers from before that fix are "
                         "not comparable to numbers generated after it")
    return result


def test_h4_matched_capacity(df: pd.DataFrame) -> dict:
    """H4 again, against the classical generator matched to the QGAN's parameter count (config
    classical_ablations). Reported BESIDE the large-classical H4 so the result brackets the capacity confound:
    a quantum gap that survives against both is not a capacity artefact; one that appears only against the
    matched generator is a statement about small models."""
    slug = "classical_gan_llm_matched"
    if "run_id" not in df.columns or not (df["run_id"].map(config_of_run_id) == slug).any():
        return {"status": "no_matched_rows",
                "note": f"no rows with run_id slug {slug!r}; run the 'Classical GAN-LLM (matched)' condition"}
    return test_h4_quantum_advantage(df, classical_slug=slug)


# ===================================================================================================================
# Amendment 3 (12 Nov 2026): the five hypotheses H1-H5.
#   H1 FID (above)  | H2 one-step RMSE  | H3 attack success (QGAN vs classical)  | H4 noise -> poisoning  | H5 noise -> ASR
# H2 and H3 are the two halves of the old H4; H5 is the second half of the old H3 split. The old entanglement tests
# (test_h2_adversarial_resilience, test_h5_tomography_correlation, test_m2_dv2_mediation) stay below for the OPTIONAL exploratory
# sweep (--exploratory-entanglement) but are not part of the five hypotheses or of the Holm family.
# ===================================================================================================================
def _advantage_block(df: pd.DataFrame, metric: str, classical_slug: str | None = None) -> dict:
    full = test_h4_quantum_advantage(df, classical_slug)
    if "error" in full and metric not in full:
        return full
    blk = dict(full.get(metric) or {"error": f"no '{metric}' result"})
    for k in ("forecast_design", "classical_comparator", "n_runs", "note", "caveat"):
        if k in full:
            blk[k] = full[k]
    blk["metric"] = metric
    return blk


def test_h2_forecast_rmse(df: pd.DataFrame, classical_slug: str | None = None) -> dict:
    """H2: QGAN-LLM has lower one-step forecast RMSE than Classical GAN-LLM (generator on the forecast path)."""
    return _advantage_block(df, "rmse", classical_slug)


def test_h3_attack_success(df: pd.DataFrame, classical_slug: str | None = None) -> dict:
    """H3: QGAN-LLM has lower adversarial attack success rate than Classical GAN-LLM (generator on the forecast path)."""
    return _advantage_block(df, "asr", classical_slug)


def test_h4_poisoning_resistance(df: pd.DataFrame) -> dict:
    """H4: no-noise QGAN-LLM degrades more under poisoning than the primary (noisy) QGAN-LLM. (Was H3.)"""
    return test_h3_poisoning_resistance(df)


def test_h5_noise_attack_success(df: pd.DataFrame) -> dict:
    """H5: no-noise QGAN-LLM has a HIGHER adversarial attack success rate than the primary (noisy) QGAN-LLM.

    Same matched pair as H4 (the two configs differ ONLY in noise_strength, same seeds), different outcome: the evasion attacks
    (FGSM / PGD / CW) instead of data poisoning. Needs no extra runs: `asr` is already scored for every run."""
    col = "asr"
    if col not in df.columns or "noise_strength" not in df.columns or "run_id" not in df.columns:
        return {"error": "need 'asr', 'noise_strength' and 'run_id' columns"}
    q = df[(df["model_type"] == QGAN) & df["noise_strength"].notna()]
    cfg = q["run_id"].map(config_of_run_id)
    nn_rows = q[(cfg == _NO_NOISE_SLUG) & (q["noise_strength"] == 0)]
    noisy_rows = q[(cfg == _PRIMARY_SLUG[QGAN]) & (q["noise_strength"] > 0)]
    fix = REPEAT_CMD.format(model="QGAN-LLM (no noise)", n=2)
    if not (len(nn_rows) and len(noisy_rows)):
        result = _insufficient({"no_noise": len(nn_rows), "noisy": len(noisy_rows)}, 1,
                               "Run the matched 'QGAN-LLM (no noise)' condition in the same design (same seeds).")
        result["design"] = "matched_pair_missing"
        return result
    a, b = nn_rows[col].dropna().to_numpy(), noisy_rows[col].dropna().to_numpy()
    result = _compare_two(a, b, "no_noise", "noisy", fix)
    result["design"] = "matched_config_noise_only"
    result["metric"] = col
    result["equivalence"] = _equivalence(a, b, "asr")
    result["n_per_group_used"] = {"no_noise": int(len(a)), "noisy": int(len(b))}
    result["configs_used"] = {"no_noise": _NO_NOISE_SLUG, "noisy": _PRIMARY_SLUG[QGAN]}
    result["interpretation"] = "positive t (no_noise ASR > noisy ASR) supports H5: noise injection -> lower attack success"
    if PAIRED_ENABLED:
        result["paired_by_seed"] = paired_by_seed(nn_rows, noisy_rows, col)
    return result


def test_h5_tomography_correlation(df: pd.DataFrame) -> dict:
    """Prefers the LLM-based detector (llm_threat_accuracy, from src/llm/llm_pipeline.py via
    threat_scoring.py) when at least 4 configurations have it; otherwise falls back to the interim
    classical detector's 'accuracy' and says so in the result."""
    if "entanglement_entropy" not in df.columns:
        return {"error": "need an entanglement_entropy column"}
    for col, detector in (("llm_threat_accuracy_cal", "LLM threat probability, threshold calibrated on disjoint windows (threat_calibration.py)"),
                          ("llm_threat_accuracy", "LLM threat probability (threat_scoring.py)"),
                          ("accuracy", "interim classical RandomForest (threat_detection.py)")):
        if col not in df.columns:
            continue
        sub = df.dropna(subset=["entanglement_entropy", col])
        if len(sub) >= 4 or col == "accuracy":
            result = _entropy_correlation(sub, col)
            n_col = "llm_threat_cal_n" if col == "llm_threat_accuracy_cal" else "llm_threat_n"
            if col.startswith("llm_threat_accuracy") and n_col in sub.columns:
                pc_flags = (sub["llm_threat_pc_passed"].dropna().tolist()
                            if "llm_threat_pc_passed" in sub.columns else None)
                result["decidability"] = detector_gate(sub[col].to_numpy(), sub[n_col].to_numpy(),
                                                       pc_passed=pc_flags or None)
                for extra in ("llm_threat_auroc", "ref_last_jump_auroc", "llm_threat_pc_accuracy"):
                    if extra in sub.columns and sub[extra].notna().any():
                        result[f"diag_{extra}_median"] = float(sub[extra].median())
            result["detector"] = detector
            result["note"] = (f"accuracy column used: {col}. "
                              + ("" if col == "llm_threat_accuracy" else
                                 "No LLM-based detection results yet; this is the stopgap detector."))
            return result
    return {"error": "no detection-accuracy column available"}


def test_m2_dv2_mediation(df: pd.DataFrame) -> dict:
    """M2->DV2: does entanglement level mediate model-configuration's
    effect on ASR category? See mediation_pipelines.py's UNIT-OF-ANALYSIS
    CAVEAT -- these rows are one per model CONFIG, not one per
    independent run, so this is still underpowered even when it runs
    without error; reported for completeness, not as a settled result."""
    sw = sweep_subset(df, {"asr", "entanglement_entropy"}) if "run_id" in df.columns else None
    if sw is not None:
        # Controlled sweep: IV = designed entangling level (manipulated), M = MEASURED entanglement
        # entropy, DV = ASR (continuous; the sweep rows need no category approximation). One row per
        # independent (level, seed) run. Variables are z-scored so the paths are comparable.
        def z(a):
            a = np.asarray(a, dtype=float)
            sd = a.std()
            return (a - a.mean()) / sd if sd > 0 else None
        iv, m, dv = z(sw["sweep_level"]), z(sw["entanglement_entropy"]), z(sw["asr"])
        if iv is None or m is None or dv is None:
            return {"error": "a sweep variable has zero variance (level, entropy or ASR is constant) -- "
                             "mediation is undefined"}
        try:
            result = mediation_analysis(iv, m, dv, n_bootstrap=5000, seed=42)
        except Exception as exc:
            return {"error": str(exc)}
        result.update(_sweep_design_info(sw, "asr"))
        result.update({
            "n_observations": int(len(sw)),
            "pathway": "designed entangling level -> measured entanglement entropy -> ASR",
            "iv_name": "designed entangling level (L1-L6, ordinal)", "mediator_name": "entanglement_entropy",
            "dv_name": "asr", "standardized": True,
            "caveat": ("circuit depth also changes parameter count; treat the level as 'entangling "
                       "structure', not entanglement alone"),
        })
        return result

    if "asr" not in df.columns or "model_type" not in df.columns:
        return {"error": "need model_type and asr columns"}
    sub = df.dropna(subset=["asr"]).copy()
    if len(sub) < 4:
        return {"error": f"insufficient data: need >= 4 observations, have {len(sub)}"}
    sub["model_config"] = sub["model_type"].str.contains("QGAN|QLSTM|QAOA", case=False, na=False).astype(int)
    is_linear = sub["run_id"].map(config_of_run_id).str.contains("linear", na=False)
    sub["entanglement_topology"] = np.where(is_linear, "linear", "ring")
    try:
        result = run_m2_dv2_mediation(sub["model_config"].to_numpy(),
                                       sub["entanglement_topology"].tolist(),
                                       sub["asr"].to_numpy())
        result["n_observations"] = len(sub)
        result.update({"design": "pooled_configs", "unit": "config"})
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


def _guard(fn, df):
    """One failing test must never stop the others (or the output file) from being written."""
    try:
        return fn(df)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _sanitize(o):
    """NaN/inf -> None so the output is valid JSON (json.dump would emit bare NaN)."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _status_of(result: dict) -> str:
    if not isinstance(result, dict):
        return "unknown"
    if "status" in result:
        return result["status"]
    if "error" in result:
        return "error"
    if any(isinstance(v, dict) and "status" in v for v in result.values()):
        sts = [v["status"] for v in result.values() if isinstance(v, dict) and "status" in v]
        return "ok" if "ok" in sts else sts[0]
    return "ok" if ("p_value" in result or "r" in result or "indirect_effect" in result) else "unknown"


def test_naive_baseline(df: pd.DataFrame) -> dict:
    """Every configuration against the NAIVE baseline (persistence: next close = current close; ratio 1.0).
    A model whose RMSE/persistence is not below 1 has no forecasting skill, and a quantum-vs-classical gap between two such
    models is not a forecasting advantage. Per config (not pooled): runs are seeds."""
    from scipy import stats
    out = {}
    for col, label in (("rmse_over_persistence", "head / generator-on-path forecast"), ("llm_rmse_over_persistence", "LLM forecast")):
        if col not in df.columns or "run_id" not in df.columns:
            continue
        d = df.dropna(subset=[col]).copy()
        d["_cfg"] = d["run_id"].map(config_of_run_id)
        rows = {}
        for cfg, g in d.groupby("_cfg"):
            v = g[col].to_numpy(float)
            row = {"n_runs": int(len(v)), "ratio_mean": float(v.mean()), "share_below_1": float((v < 1).mean())}
            if len(v) >= 3 and v.std(ddof=1) > 0:
                t = stats.ttest_1samp(v, 1.0)
                m, se = v.mean(), v.std(ddof=1) / len(v) ** 0.5
                tc = stats.t.ppf(0.975, len(v) - 1)
                row.update({"ci95": [float(m - tc * se), float(m + tc * se)], "p_vs_1": float(t.pvalue),
                            "beats_naive": bool(m + tc * se < 1.0)})
            else:
                row["beats_naive"] = None
                row["note"] = "fewer than 3 runs (or no variance): descriptive only"
            rows[cfg] = row
        out[col] = {"label": label, "configs": rows,
                    "reading": "ratio < 1 beats persistence; beats_naive needs the whole 95% CI below 1"}
    return out or {"error": "no rmse_over_persistence column"}


def test_q1_three_arm(paths) -> dict:
    """Three-arm fair paired comparison (inputs only / classical random features / quantum circuit features) read from
    scripts/run_feature_map_experiment.py output, with the naive baseline and the balance audit attached."""
    from src.baselines.feature_map_forecaster import DEFAULT_RIDGE_GRID
    from src.evaluation import three_arm
    res = {}
    for p in paths:
        if not os.path.isfile(p):
            res[p] = {"status": "not_run", "how": "python -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8   (and again with --n-qubits 12)"}
            continue
        with open(p, encoding="utf8") as fh:
            d = json.load(fh)
        r = three_arm.verdict(d["per_run"], d["dm_quantum_vs"])
        r["audit"] = three_arm.audit(d["per_run"], d["args"], DEFAULT_RIDGE_GRID)
        r["source"] = p
        res[f"{d['args']['n_qubits']}_qubits ({os.path.basename(p)})"] = r
    return res


def authoritative_design(df: pd.DataFrame, include_pilot: bool = False) -> tuple[pd.DataFrame, dict]:
    """ONE analysed sample (reviewer issues 3 and 4): only rows run with the generator ON the forecast path
    (forecast_path == 'generator'). Pilot rows (head_only, or no forecast_path recorded) are a different experimental
    design -- the forecast could not depend on the generator -- so they are never mixed into a hypothesis test.
    --include-pilot restores the old behaviour. A file with no forecast_path column at all is a legacy file: kept whole."""
    info = {"design": "generator_on_forecast_path", "include_pilot": bool(include_pilot), "excluded_pilot_rows": {}}
    if include_pilot:
        info["design"] = "all rows (pilot included)"
        return df, info
    if "forecast_path" not in df.columns:
        info["design"] = "legacy file without forecast_path: all rows kept"
        return df, info
    keep = df["forecast_path"] == "generator"
    if "run_id" in df.columns and (~keep).any():
        info["excluded_pilot_rows"] = (df.loc[~keep, "run_id"].map(config_of_run_id).value_counts().to_dict())
    return df[keep], info


def sample_table(df: pd.DataFrame) -> dict:
    """The single authoritative n per condition. Every table and every test below uses this same dataframe."""
    return (df["run_id"].map(config_of_run_id).value_counts().sort_index().to_dict()
            if "run_id" in df.columns and len(df) else {})


def count_audit(out: dict) -> dict:
    """Cross-check the n each test reports against the one sample table; any mismatch is listed, never silent."""
    tbl, issues = out.get("n_runs_per_config", {}), []

    def check(where, name, used, slug):
        if used is not None and slug in tbl and int(used) != int(tbl[slug]):
            issues.append(f"{where}: {name} n={used} but the sample table has {slug}={tbl[slug]}")
    h1 = out.get("H1_synthetic_data_fidelity", {}) or {}
    for g, slug in ((QGAN, "qgan_llm"), (CGAN, "classical_gan_llm")):
        check("H1", g, (h1.get("n_per_group_used") or {}).get(g), slug)
    for key, slug in (("H2_forecast_rmse", "classical_gan_llm"), ("H3_attack_success_qgan_vs_classical", "classical_gan_llm"),
                      ("H2_matched_capacity", "classical_gan_llm_matched"), ("H3_matched_capacity", "classical_gan_llm_matched")):
        nr = (out.get(key, {}) or {}).get("n_runs") or {}
        check(key, QGAN, nr.get(QGAN), "qgan_llm")
        check(key, CGAN, nr.get(CGAN), slug)
    for hk, hname in (("H4_poisoning_resistance_noise", "H4"), ("H5_noise_attack_success", "H5")):
        hh = (out.get(hk, {}) or {}).get("n_per_group_used") or {}
        check(hname, "no_noise", hh.get("no_noise"), "qgan_llm_no_noise")
        check(hname, "noisy", hh.get("noisy"), "qgan_llm")
    return {"consistent": not issues, "issues": issues}


GATE_RC = 3          # exit code of the deliberate stop at the confirmatory gate (not a failure)
CONFIRM_LOG = os.path.join(snap.SNAP_DIR, "confirmatory_runs.json")


def _confirm_log() -> list:
    try:
        with open(CONFIRM_LOG, encoding="utf8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def gate_stop_message(entry: dict, status: dict) -> str:
    return "\n".join([
        "", "=" * 78,
        f"STOP AT THE CONFIRMATORY GATE: {status['label']}",
        "The pre-registered design is complete, so the next analysis is the confirmatory one. Nothing was tested.",
        f"A snapshot of the data was taken for you:  {entry['snapshot']}",
        f"   sha256 {entry['sha256']}   ({entry['rows']} rows)" + ("   [already existed]" if entry.get("reused") else ""),
        "Do the gate by hand, then run the tests ONCE:",
        "  1. python -m scripts.freeze_preregistration --check        (expect: no unrecorded changes)",
        "  2. python -m scripts.snapshot_results --verify             (the csv must still match the snapshot; do not edit it)",
        "  3. python -m scripts.run_hypothesis_tests --run-confirmatory [--experiment-id ...]",
        "     (via the driver:  python -m scripts.run_study --only hypothesis --force --run-confirmatory)",
        "Do not edit code or data after seeing the output. A second run needs --rerun-reason and is disclosed in the output.",
        "=" * 78])


def gate_checks(args, csv_path: str) -> tuple[dict | None, str]:
    """--run-confirmatory is only honoured when the code is frozen AND the data matches a snapshot. Returns (gate dict, '')
    or (None, reason)."""
    rc = subprocess.run([sys.executable, "-m", "scripts.freeze_preregistration", "--check"],
                        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if rc.returncode != 0:
        return None, "freeze check failed (python -m scripts.freeze_preregistration --check):\n" + (rc.stdout or rc.stderr)
    ok, entry = snap.verify(csv_path)
    if not ok:
        return None, ("the csv does not match any recorded snapshot (it changed after the snapshot, or none exists): "
                      "python -m scripts.snapshot_results, then disclose that the data changed")
    prior = [r for r in _confirm_log() if r.get("csv_sha256") == entry["sha256"]]
    if prior and not args.rerun_reason:
        return None, (f"a confirmatory run on this exact data already exists ({prior[-1]['utc']}); a second run must be "
                      "disclosed: pass --rerun-reason \"why\"")
    return {"freeze_check": "clean", "csv_sha256": entry["sha256"], "snapshot": entry["snapshot"],
            "rerun": bool(prior), "rerun_number": len(prior) + 1, "rerun_reason": args.rerun_reason or None}, ""


def build_summary(out: dict) -> dict:
    keys = ["H1_synthetic_data_fidelity", "H2_forecast_rmse", "H3_attack_success_qgan_vs_classical",
            "H4_poisoning_resistance_noise", "H5_noise_attack_success", "H2_matched_capacity", "H3_matched_capacity",
            "X1_entanglement_vs_asr", "X2_entanglement_vs_detection", "X3_entanglement_mediation"]
    return {k: _status_of(out[k]) for k in keys if k in out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-csv", default="results/all_results.csv")
    ap.add_argument("--out", default="results/hypothesis_test_results.json")
    ap.add_argument("--dedupe", choices=["config_seed", "config", "none"], default="config_seed",
                    help="config_seed (default): re-evaluations of one checkpoint collapse, but each "
                         "different seed counts as an independent run. config: one row per config "
                         "(the old behaviour). none: every row is an observation.")
    ap.add_argument("--equiv-margin", action="append", default=[], metavar="KEY=VALUE",
                    help="override an equivalence margin value, e.g. fid=0.15 asr=3 rmse=0.01 r0=0.25 "
                         "(keeps the margin's kind; see EQUIV_MARGINS). Fix margins BEFORE the confirmatory runs.")
    ap.add_argument("--no-dedupe", action="store_true", help="same as --dedupe none")
    ap.add_argument("--run-confirmatory", action="store_true",
                    help="opt in to the confirmatory analysis. Without it the run STOPS when the pre-registered design is "
                         "complete (after snapshotting the csv). Needs a clean freeze check and a csv that matches its snapshot")
    ap.add_argument("--rerun-reason", default=None, metavar="WHY",
                    help="required for a second confirmatory run on the same data; stored in the output (disclosure)")
    ap.add_argument("--models-dir", default="models", help="where the checkpoints are (for the three-arm ledger)")
    ap.add_argument("--no-ledger", action="store_true", help="skip the three-arm ledger / chronology section")
    ap.add_argument("--exploratory-entanglement", action="store_true",
                    help="ALSO run the old entanglement-sweep analyses as exploratory X1-X3 (they were dropped from H1-H5 on "
                         "12 Nov 2026 to save compute; they need the optional sweep conditions and never enter the Holm family)")
    ap.add_argument("--include-pilot", action="store_true",
                    help="also analyse pilot rows (forecast_path != generator); default: generator-on-path rows only")
    ap.add_argument("--walk-forward-json", default=os.path.join("results", "walk_forward_q8.json"),
                    help="walk-forward file of the three arms (scripts.run_walk_forward); the naive baseline file is looked up by "
                         "experiment id, then in results/naive_baseline/legacy")
    ap.add_argument("--feature-map-json", nargs="*", default=["results/feature_map_advantage.json"],
                    help="three-arm feature-map result file(s) from run_feature_map_experiment (one per qubit count)")
    ap.add_argument("--experiment-id", default=None,
                    help="analyse ONE experiment (EXP-...); several comma-separated ids need --pool-experiments. "
                         "Without it the results file must contain exactly one experiment.")
    ap.add_argument("--pool-experiments", action="store_true",
                    help="allow several experiments in one analysis (paired-by-seed tests are then not reported)")
    args = ap.parse_args()
    for item in args.equiv_margin:
        k, _, v = item.partition("=")
        if k not in EQUIV_MARGINS or not v:
            raise SystemExit(f"--equiv-margin: unknown key or empty value in {item!r}; keys: {sorted(EQUIV_MARGINS)}")
        EQUIV_MARGINS[k] = (EQUIV_MARGINS[k][0], float(v))

    require_results(args.results_csv, "run_hypothesis_tests")   # exits with "run src.experiments.run_all first" if empty
    df = pd.read_csv(args.results_csv)
    n_file = len(df)
    experiments_in_file = exp.experiments_in(df)
    try:
        df, used_ids = exp.select_experiments(
            df, [x.strip() for x in (args.experiment_id or "").split(",")], pool=args.pool_experiments)
    except exp.ExperimentError as err:
        raise SystemExit(f"[experiment] {err}")
    overlap = exp.seed_overlap(df) if len(used_ids) > 1 else {}
    global PAIRED_ENABLED
    PAIRED_ENABLED = len(used_ids) <= 1
    n_raw = len(df)
    mode = "none" if args.no_dedupe else args.dedupe
    if mode == "config_seed":
        df = latest_per_config_seed(df)
    elif mode == "config":
        df = latest_per_config(df)
    n_before_design = len(df)
    df_all = df                          # everything in the experiment: the chronology must also show the pilot/legacy rows
    df, design_info = authoritative_design(df, args.include_pilot)
    if "asr_baseline" in df.columns:
        legacy = df["asr_baseline"].isna()
        df.loc[legacy, ["asr", "asr_clean"]] = float("nan")  # scored against a non-price column

    tbl0 = sample_table(df)
    status0 = study_status(tbl0)
    budget0 = hypothesis_budget(tbl0)
    print("\n" + format_budget(budget0, status0), flush=True)
    ledger = None
    if not args.no_ledger:
        try:
            ledger = build_ledger(df_all, models_dir=args.models_dir,
                                  results_dir=os.path.dirname(os.path.abspath(args.results_csv)),
                                  feature_map_paths=args.feature_map_json or None,
                                  experiment_id=getattr(args, "experiment_id", None))
            print("\n" + format_ledger(ledger), flush=True)
        except Exception as exc:  # noqa: BLE001 -- a display problem must never block the analysis
            print(f"\n[ledger] unavailable: {type(exc).__name__}: {exc}", flush=True)
    gate = None
    if status0["complete"]:
        if not args.run_confirmatory:
            try:
                entry = snap.snapshot(args.results_csv, note="automatic snapshot at the confirmatory gate")
            except (OSError, RuntimeError) as err:
                raise SystemExit(f"[gate] could not snapshot {args.results_csv}: {err}")
            print(gate_stop_message(entry, status0), flush=True)
            sys.exit(GATE_RC)
        gate, why = gate_checks(args, args.results_csv)
        if gate is None:
            print(f"\n[gate] REFUSED: {why}", flush=True)
            sys.exit(GATE_RC)
        print(f"\n[gate] open: freeze clean, csv sha256 {gate['csv_sha256'][:16]}... matches {gate['snapshot']}"
              + (f"; RERUN #{gate['rerun_number']} ({gate['rerun_reason']})" if gate["rerun"] else ""), flush=True)
    else:
        print("\n[interim] the pre-registered design is incomplete: everything below is exploratory, whatever its p-values."
              + ("  (--run-confirmatory has no effect until the design is complete)" if args.run_confirmatory else ""), flush=True)

    out = {
        "run_budget": budget0,
        "three_arm_ledger": ledger,
        "confirmatory_gate": gate,
        "analysis_plan_sha": aplan.plan_sha() if os.path.isfile(aplan.PLAN_PATH) else None,
        "experiment_id": used_ids[0] if len(used_ids) == 1 else None,
        "experiments_used": used_ids,
        "experiments_in_file": experiments_in_file,
        "pooled_experiments": len(used_ids) > 1,
        "seed_overlap_between_experiments": overlap,
        "n_rows_in_file": n_file,
        "n_rows_raw": n_raw,
        "n_rows": len(df),
        "analysed_design": design_info,
        "n_rows_excluded_other_design": n_before_design - len(df),
        "dedupe": mode,
        "equivalence_margins": {k: {"kind": v[0], "value": v[1]} for k, v in EQUIV_MARGINS.items()},
        "n_runs_per_model": df.groupby("model_type").size().to_dict() if "model_type" in df.columns else {},
        "n_runs_per_config": (df["run_id"].map(config_of_run_id).value_counts().to_dict()
                              if "run_id" in df.columns else {}),
        "H1_synthetic_data_fidelity": _guard(test_h1_synthetic_data_fidelity, df),
        "H2_forecast_rmse": _guard(test_h2_forecast_rmse, df),
        "H3_attack_success_qgan_vs_classical": _guard(test_h3_attack_success, df),
        "H4_poisoning_resistance_noise": _guard(test_h4_poisoning_resistance, df),
        "H5_noise_attack_success": _guard(test_h5_noise_attack_success, df),
        # bracket against the parameter-matched classical generator (reported beside H2 / H3, not a separate hypothesis)
        "H2_matched_capacity": _guard(lambda d: test_h2_forecast_rmse(d, "classical_gan_llm_matched"), df),
        "H3_matched_capacity": _guard(lambda d: test_h3_attack_success(d, "classical_gan_llm_matched"), df),
        "naive_baseline": _guard(test_naive_baseline, df),
        "Q1_three_arm_feature_map": _guard(lambda _df: test_q1_three_arm(args.feature_map_json), df),
    }
    if args.exploratory_entanglement:       # optional add-on: needs the entanglement sweep conditions (not in the default plan)
        out["X1_entanglement_vs_asr"] = _guard(test_h2_adversarial_resilience, df)
        out["X2_entanglement_vs_detection"] = _guard(test_h5_tomography_correlation, df)
        out["X3_entanglement_mediation"] = _guard(test_m2_dv2_mediation, df)
    out["sample_table"] = sample_table(df)
    out["study_status"] = study_status(out["sample_table"])      # interim vs the pre-registered 80 runs
    out["count_audit"] = count_audit(out)
    out["summary"] = build_summary(out)          # execution status only: could the test RUN?
    out["planned_vs_ran"] = aplan.planned_vs_ran(out)   # planned analysis, what ran, which fallback fired
    out["verdicts"] = compute_verdicts(out)       # what the test SAYS, how far to trust it, and why
    # methodology figure (inputs-only / classical features / quantum features, identical chronological splits, identical metrics,
    # paired seeds x walk-forward windows): checked against the files that exist, attached to EVERY verdict
    from src.baselines.feature_map_forecaster import DEFAULT_RIDGE_GRID as _RG
    from src.evaluation import protocol_compliance as pcmp
    _wf_path = args.walk_forward_json
    if not os.path.isfile(_wf_path):          # fall back to the newest un-timestamped walk-forward file (e.g. another width)
        import glob
        _cand = [p for p in glob.glob(os.path.join("results", "walk_forward_q*.json")) if "__" not in os.path.basename(p)]
        _wf_path = max(_cand, key=os.path.getmtime) if _cand else _wf_path
    _wf = pcmp.load_json(_wf_path)
    out["walk_forward_file"] = _wf_path if _wf else None
    _checks = pcmp.check_walk_forward(_wf, pcmp.find_naive(out.get("experiment_id")), ridge_grid=_RG)
    _checks += pcmp.check_poisoning_and_detector(df.to_dict("records"))
    out["methodology_compliance"] = pcmp.summarise(_checks)
    out["verdicts"] = pcmp.annotate_verdicts(out["verdicts"], out["methodology_compliance"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from src.utils.run_identity import pipeline_run_id, save_json_timestamped, utc_stamp
    stamp = utc_stamp()
    out["pipeline_run_id"] = pipeline_run_id()
    out["generated_utc"] = stamp
    out = _sanitize(out)
    saved = save_json_timestamped(args.out, out, stamp=stamp)   # <stem>__<stamp>.json + latest copy
    out["saved_as"] = saved["versioned"]
    if gate is not None:
        os.makedirs(snap.SNAP_DIR, exist_ok=True)
        with open(CONFIRM_LOG, "w", encoding="utf8") as fh:
            json.dump(_confirm_log() + [{"utc": stamp, "csv_sha256": gate["csv_sha256"], "snapshot": gate["snapshot"],
                                         "output": saved["versioned"], "rerun": gate["rerun"],
                                         "rerun_reason": gate["rerun_reason"]}], fh, indent=2)
    print(json.dumps(out, indent=2, default=str))
    print("\nHYPOTHESIS TEST STATUS (execution):")
    for k, v in out["summary"].items():
        print(f"  {k:<46} {v}")
    print("\nNAIVE BASELINE (persistence = 1.0; < 1 beats 'no change'):")
    for col, blk in (out.get("naive_baseline") or {}).items():
        if isinstance(blk, dict) and "configs" in blk:
            for cfg, r in blk["configs"].items():
                print(f"  {col[:12]:<12} {cfg:<44} n={r['n_runs']:<3} ratio={r['ratio_mean']:.3f}  beats_naive={r.get('beats_naive')}")
    print("\nTHREE-ARM FEATURE-MAP COMPARISON (naive baseline, inputs only, classical random features, quantum):")
    for k, r in (out.get("Q1_three_arm_feature_map") or {}).items():
        if isinstance(r, dict) and "verdict" in r:
            print(f"  {k}: {r['verdict']}   balance audit {'PASSED' if r['audit']['all_passed'] else 'FAILED'}")
            for a, s in r["versus_naive"].items():
                print(f"     {a:<13} skill {s['skill_mean']:+.5f} CI [{s['skill_ci95'][0]:+.5f}, {s['skill_ci95'][1]:+.5f}]  beats naive: {s['beats_naive']}")
            for n in r["notes"]:
                print("     NOTE:", n)
        else:
            print(f"  {k}: {r}")
    ss = out["study_status"]
    print(f"\nSTUDY STATUS: {ss['label']}   (analysed design: {out['analysed_design']['design']}; "
          f"{out['n_rows_excluded_other_design']} row(s) of other designs excluded)")
    print("  n per condition:", out["sample_table"])
    print("  count audit:", "consistent" if out["count_audit"]["consistent"] else "MISMATCH -> " + "; ".join(out["count_audit"]["issues"]))
    mc = out["methodology_compliance"]
    print(f"\nMETHODOLOGY FIGURE: {mc['label']}")
    for c in mc["checks"]:
        print(f"  [{c['status']:<14}] {c['id']:<32} {c['detail']}" + (f"\n{'':19}fix: {c['fix']}" if c["fix"] else ""))
    print("\nHYPOTHESIS VERDICTS (deterministic rules in src/evaluation/verdicts.py):")
    print(format_table(out["verdicts"]))


if __name__ == "__main__":
    main()
