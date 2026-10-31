"""
Results collector.

*** MAJOR PROBLEM REMOVED FROM THE ORIGINAL CODE ***

The original ResultsCollector had this:

    self.expected = {
        'Classical LSTM': {'rmse': 0.523, 'asr': 31.0},
        'QGAN-LLM (20-qubit)': {'rmse': 0.364, 'asr': 8.7},
        ...
    }
    def verify_consistency(self):
        for _, row in self.df.iterrows():
            expected_rmse = self.expected[model_type]['rmse']
            if abs(row['rmse'] - expected_rmse) > 0.001:
                logger.warning(f"discrepancy...")

This "verifies" real results against the exact disputed numbers from
the dissertation, typed in as ground truth. Any run — even one with a
completely untrained, random-weight model — would either match those
numbers by writing them in, or get flagged as "wrong" for NOT matching
disputed figures that were never independently confirmed in the first
place. That's circular: it can only ever confirm what it was told to
expect.

This version does the opposite: it aggregates whatever the actual runs
produced, with NO hardcoded target values anywhere. Discrepancy-checking
against the dissertation's specific numbers is handled separately and
explicitly in scripts/verify_chapter4.py, where you paste in the
dissertation's claims yourself and the two are diffed transparently —
so the comparison is auditable rather than baked into a class attribute.
"""
from __future__ import annotations

import json
import os

import pandas as pd


class ResultsCollector:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        csv_path = os.path.join(results_dir, "all_results.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"No results found at {csv_path}. Run src/experiments/run_all.py first."
            )
        self.df = pd.read_csv(csv_path)

    def summary_table(self) -> pd.DataFrame:
        """One row per model_type, keeping only the LATEST run for each
        (in case a model was re-run — earlier attempts don't silently
        get averaged into the final reported number)."""
        # BUG FIXED: groupby().last() returns the last NON-NULL value of each
        # column independently, so if the newest run had asr=NaN it silently
        # inherited asr from an OLDER run -- a row stitched from different runs,
        # contradicting this docstring. Keep the single newest row instead.
        from src.utils.run_identity import config_of_run_id
        d = self.df.copy()
        d["config"] = d["run_id"].map(config_of_run_id)
        d["_ts"] = d["run_id"].astype(str).str.extract(r"(\d{8}T\d{6}Z)$")[0]
        d = d.sort_values("_ts").drop_duplicates(subset="config", keep="last")
        return d.drop(columns="_ts").reset_index(drop=True)

    def export_chapter4_tables(self) -> dict:
        """
        Reshapes the summary table into the specific comparisons Chapter 4
        reports (model-vs-model RMSE/ASR deltas, etc.), computed fresh from
        whatever is actually in all_results.csv -- not copied from any
        pre-existing table.

        BUG FIXED (ValueError: truth value of a Series is ambiguous): this method
        used to index the summary by ``model_type``. Every QGAN-LLM ablation
        (20-qubit ring, 12-qubit, linear topology, no noise) is written with
        model_type == "QGAN-LLM", so ``summary.loc["QGAN-LLM", col]`` returned a
        Series of four values and ``pd.isna(val)`` could not be used in a
        conditional. The summary is now indexed by ``config`` (the run_id slug,
        unique after summary_table()), which is the same key verify_chapter4 uses.
        """
        from src.utils.run_identity import model_slug

        summary = self.summary_table()
        if summary["config"].duplicated().any():  # defensive; summary_table() dedups
            summary = summary.drop_duplicates(subset="config", keep="last")
        summary = summary.set_index("config")
        tables = {}

        def safe_get(model, col):
            if model in summary.index and col in summary.columns:
                val = summary.loc[model, col]
                return None if pd.isna(val) else float(val)
            return None

        model_names = list(summary.index)
        # One row per config. The CSV is wide (~95 columns) and each model fills only its own metrics, so
        # drop empty cells per record: bare NaN is not valid JSON and made ~430 lines of noise.
        tables["model_comparison"] = [
            {k: v for k, v in rec.items() if not (v is None or (isinstance(v, float) and v != v))}
            for rec in summary.reset_index().to_dict(orient="records")]

        primary = "QGAN-LLM"
        baseline = "Classical GAN-LLM"
        if primary in model_names and baseline in model_names:
            qgan_rmse = safe_get(primary, "rmse")
            base_rmse = safe_get(baseline, "rmse")
            qgan_asr = safe_get(primary, "asr")
            base_asr = safe_get(baseline, "asr")

            def reduction(base, new):
                # `is not None` and a non-zero base: 0.0 is a real measurement.
                if base is None or new is None or base == 0:
                    return None
                return 100 * (base - new) / base

            # Identical metrics across different generators mean the generator
            # never reaches the metric (see CHANGES_OCT19.md, cause C1).
            gan_cfgs = [c for c in configs if c.startswith(("classical_gan_llm", "qgan_llm"))]
            rmses = {safe_get(c, "rmse") for c in gan_cfgs}
            degenerate = len(gan_cfgs) > 1 and len({r for r in rmses if r is not None}) == 1
            tables["primary_vs_baseline"] = {
                "rmse_reduction_pct": (
                    100 * (base_rmse - qgan_rmse) / base_rmse if qgan_rmse and base_rmse else None
                ),
                "asr_reduction_pct": (
                    100 * (base_asr - qgan_asr) / base_asr if qgan_asr and base_asr else None
                ),
                "qgan_rmse": qgan_rmse,
                "baseline_rmse": base_rmse,
                "qgan_asr": qgan_asr,
                "baseline_asr": base_asr,
                "degenerate_comparison": degenerate,
                "degenerate_note": (
                    "All GAN-based configurations share one RMSE, so the forecast "
                    "does not depend on the generator; the reduction above is not "
                    "evidence of quantum advantage."
                ) if degenerate else None,
            }

        # Flag the structural finding rather than hiding it: all GAN-based
        # configurations share one closed-form head, so identical RMSE is a
        # property of the architecture, not evidence about the circuit.
        gan = summary[summary.index.str.contains("gan_llm")]
        if "rmse" in gan.columns and len(gan) > 1:
            tables["gan_rmse_degenerate"] = bool(gan["rmse"].round(12).nunique() == 1)

        os.makedirs(self.results_dir, exist_ok=True)
        out_path = os.path.join(self.results_dir, "chapter4_numbers.json")
        from src.utils.run_identity import save_json_timestamped
        save_json_timestamped(out_path, tables)   # results/chapter4_numbers__<UTC stamp>.json + latest copy

        return tables
