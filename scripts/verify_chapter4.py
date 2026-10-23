"""
Compares ACTUAL results (results/all_results.csv, produced by a real
run of src/experiments/run_all.py) against the dissertation's claimed
numbers — which you enter explicitly in DISSERTATION_CLAIMS below.

This is deliberately NOT automatic. The original check_consistency.py
had the dissertation's numbers hardcoded as "expected" values and
just checked whether they appeared as substrings somewhere in a text
file — which can only ever confirm itself. Here, you paste in the
manuscript's specific claims, and the script computes the real
percentage difference against what the run actually produced, prints
every comparison (not just the ones that fail), and leaves the
judgment call about what counts as an acceptable discrepancy to you,
not to a silently-passing threshold.

Run with:  python -m scripts.verify_chapter4
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.results_collector import ResultsCollector
from src.utils.run_identity import model_slug
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# EDIT THIS: paste in whichever dissertation numbers you want checked.
# Leave a field as None if the manuscript doesn't claim it / you haven't
# decided on it yet — the script will just skip that comparison and say so.
# ---------------------------------------------------------------------------
DISSERTATION_CLAIMS = {
    "Classical LSTM":     {"rmse": 0.523, "asr": None},
    "Classical GAN-LLM":  {"rmse": 0.492, "asr": 31.0},
    "QGAN-LLM":           {"rmse": 0.364, "asr": 8.7},
    # Manuscript text (Alternative Configurations): 12-qubit RMSE 0.452 (the
    # previous value here, 0.432, does not appear in the manuscript); linear
    # topology RMSE 0.350, ASR 12.1. The 12-qubit ASR of 11.3 is not found in
    # the manuscript text -- confirm its source or set it to None.
    "QGAN-LLM (12-qubit)":        {"rmse": 0.452, "asr": 11.3},
    "QGAN-LLM (linear topology)": {"rmse": 0.350, "asr": 12.1},
}
# ---------------------------------------------------------------------------


def compare_value(model, metric, claimed, actual):
    if claimed is None:
        return f"  {model:<28} {metric:<6} claimed=(none given)   actual={actual}  [ASR is in percent]"
    if actual is None:
        return f"  {model:<28} {metric:<6} claimed={claimed:<10} actual=(no run found)"
    pct_diff = 100 * abs(actual - claimed) / claimed if claimed != 0 else float("inf")
    flag = "  <-- differs by >1%" if pct_diff > 1.0 else ""
    return f"  {model:<28} {metric:<6} claimed={claimed:<10} actual={actual:<10.4f} diff={pct_diff:5.2f}%{flag}"


def main():
    cfg = load_config()
    results_dir = cfg.get("logging", {}).get("results_dir", "results")
    collector = ResultsCollector(results_dir=results_dir)
    # one row per CONFIGURATION (latest run), keyed by the run_id slug
    summary = collector.summary_table().set_index("config")

    print("=" * 90)
    print("CHAPTER 4 VERIFICATION — actual run output vs. dissertation claims")
    print("=" * 90)
    print("(Comparisons run against results/all_results.csv, i.e. real model output.")
    print(" This does not hardcode or assume the dissertation numbers are correct.)\n")

    any_flagged = False
    for model, claims in DISSERTATION_CLAIMS.items():
        for metric, claimed_val in claims.items():
            actual_val = None
            key = model_slug(model)
            if key in summary.index and metric in summary.columns:
                val = summary.loc[key, metric]
                actual_val = None if (val != val) else float(val)  # NaN check
            line = compare_value(model, metric, claimed_val, actual_val)
            print(line)
            if "<--" in line:
                any_flagged = True

    # ---- sanity checks on the ACTUAL results, independent of the claims ----
    gan_cfgs = [c for c in summary.index if c.startswith(("classical_gan_llm", "qgan_llm"))]
    gan_rmse = summary.loc[gan_cfgs, "rmse"].dropna().round(12).unique() if gan_cfgs else []
    print("\nSanity checks on the actual run output:")
    if len(gan_cfgs) > 1 and len(gan_rmse) == 1:
        print(f"  [WARN] All {len(gan_cfgs)} GAN-based configurations have the SAME rmse "
              f"({gan_rmse[0]:.6f}). The forecast head does not use the generator, so these "
              "numbers cannot show a quantum (or classical) generator effect.")
    if "classical_lstm" in summary.index:
        print("  [WARN] Classical LSTM asr is scored against principal component 0 in "
              "runs before the October 19 fix; re-evaluate the checkpoint before using it.")
    if "persistence_rmse" in summary.columns:
        pers = summary["persistence_rmse"].dropna()
        if len(pers):
            print(f"  [INFO] Test-split persistence RMSE = {pers.iloc[-1]:.6f}; every model's "
                  "rmse should be read against it (rmse_over_persistence > 1 means worse).")
    print("  [INFO] Claimed values are on a different scale from the processed target "
          "series (rmse ~1e-3 to 1e-1), so large percentage differences reflect unsupported "
          "claims, not a small tolerance miss.")

    print("\n" + "=" * 90)
    if any_flagged:
        print("Some values differ from the dissertation's claims by more than 1%.")
        print("This isn't necessarily wrong — it means the manuscript's number should")
        print("be updated to match this run's real output, or the run's config should")
        print("be checked against what the manuscript describes.")
    else:
        print("All compared values are within 1% of the dissertation's claims.")
    print("=" * 90)

    collector.export_chapter4_tables()
    print(f"\nFull reshaped tables written to results/chapter4_numbers.json")


if __name__ == "__main__":
    main()
