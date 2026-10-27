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
        whatever is actually in all_results.csv — not copied from any
        pre-existing table.
        """
        # Key by configuration slug (run_id minus timestamp), NOT model_type:
        # every QGAN-LLM ablation (ring / 12-qubit / linear / no-noise) is
        # recorded as model_type "QGAN-LLM", so .loc["QGAN-LLM", col] returned
        # a Series and float() raised "The truth value of a Series is
        # ambiguous" / TypeError, which aborted verify_chapter4 (Oct 27 log).
        summary = self.summary_table().set_index("config")
        tables = {}

        def safe_get(config, col):
            if config in summary.index and col in summary.columns:
                val = summary.loc[config, col]
                if isinstance(val, pd.Series):  # defensive: duplicate index
                    val = val.iloc[-1]
                return None if pd.isna(val) else float(val)
            return None

        tables["model_comparison"] = summary.reset_index().to_dict(orient="records")

        primary = "qgan_llm"
        baseline = "classical_gan_llm"
        if primary in summary.index and baseline in summary.index:
            qgan_rmse = safe_get(primary, "rmse")
            base_rmse = safe_get(baseline, "rmse")
            qgan_asr = safe_get(primary, "asr")
            base_asr = safe_get(baseline, "asr")
            tables["primary_vs_baseline"] = {
                "rmse_reduction_pct": (
                    100 * (base_rmse - qgan_rmse) / base_rmse
                    if qgan_rmse is not None and base_rmse else None
                ),
                "asr_reduction_pct": (
                    100 * (base_asr - qgan_asr) / base_asr
                    if qgan_asr is not None and base_asr else None
                ),
                "qgan_rmse": qgan_rmse,
                "baseline_rmse": base_rmse,
                "qgan_asr": qgan_asr,
                "baseline_asr": base_asr,
            }

        # Flag the structural finding rather than hiding it: all GAN-based
        # configurations share one closed-form head, so identical RMSE is a
        # property of the architecture, not evidence about the circuit.
        gan = summary[summary.index.str.contains("gan_llm")]
        if "rmse" in gan.columns and len(gan) > 1:
            tables["gan_rmse_degenerate"] = bool(gan["rmse"].round(12).nunique() == 1)

        os.makedirs(self.results_dir, exist_ok=True)
        out_path = os.path.join(self.results_dir, "chapter4_numbers.json")
        with open(out_path, "w") as f:
            json.dump(tables, f, indent=2, default=str)

        return tables
