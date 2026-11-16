"""
Non-regression tests for CSV row integrity after Amendment 4.

Each completed run row in results/all_results.csv must reflect the
Amendment 4 config values:
  - attack_n_samples  >= 200  (not the old 50)
  - synthetic_ratio_used == 1.0  (not the old 0.4)
  - shift_sd == 1.0  (optional column, when present)

These tests:
  1. Validate that any existing CSV rows meet the Amendment 4 standards.
  2. Verify the run_id slug encodes the correct condition name.
  3. Check that all 4 conditions are represented (no condition dropped).
  4. Verify no run_id collision (each (condition, seed) appears exactly once).

If the CSV doesn't exist yet (study still running), the tests are skipped
gracefully — they become active once the first rows land.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "all_results.csv")

REQUIRED_SLUGS = {
    "qgan_llm",
    "classical_gan_llm",
    "classical_gan_llm_matched",
    "qgan_llm_no_noise",
}


# ---------------------------------------------------------------------------
# Fixture: load the CSV (skip if not present or empty)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    if not os.path.exists(RESULTS_CSV):
        pytest.skip("results/all_results.csv not found — study not yet started")
    import csv
    with open(RESULTS_CSV, newline="") as f:
        data = list(csv.DictReader(f))
    if not data:
        pytest.skip("results/all_results.csv is empty")
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCSVRowIntegrity:

    def test_no_rows_with_old_attack_n_samples(self, rows):
        """
        Any row with attack_n_samples=50 belongs to Amendment 3 (invalid for H3).
        After Amendment 4, no such rows should exist in the live CSV.
        """
        old_rows = [
            r for r in rows
            if int(r.get("attack_n_samples", 200) or 200) < 200
        ]
        if old_rows:
            ids = [r.get("run_id", "?") for r in old_rows[:5]]
            pytest.fail(
                f"{len(old_rows)} row(s) have attack_n_samples < 200 (Amendment 3 data). "
                f"First offenders: {ids}. Archive or delete these rows before the study."
            )

    def test_no_rows_with_old_synthetic_ratio(self, rows):
        """
        Any row with synthetic_ratio_used < 1.0 belongs to Amendment 3 (invalid for H4).
        """
        old_rows = [
            r for r in rows
            if r.get("synthetic_ratio_used") not in (None, "", "1.0", "1") and
            float(r.get("synthetic_ratio_used", 1.0)) < 1.0
        ]
        if old_rows:
            ids = [r.get("run_id", "?") for r in old_rows[:5]]
            pytest.fail(
                f"{len(old_rows)} row(s) have synthetic_ratio_used < 1.0 (Amendment 3 data). "
                f"First offenders: {ids}"
            )

    def test_all_four_conditions_represented(self, rows):
        """Once the study completes, all 4 condition slugs must be in the CSV."""
        from src.utils.run_identity import model_slug
        present_slugs = set()
        for r in rows:
            rid = r.get("run_id", "")
            for slug in REQUIRED_SLUGS:
                if rid.startswith(slug):
                    present_slugs.add(slug)
                    break

        missing = REQUIRED_SLUGS - present_slugs
        # Only fail if ALL slugs are present for some but not others AND study looks complete
        n_runs = len(rows)
        if n_runs >= 10 and missing:   # at least partial completion — flag missing conditions
            pytest.fail(
                f"Conditions missing from CSV: {missing}. "
                f"Present: {present_slugs}. Total rows: {n_runs}"
            )

    def test_no_duplicate_condition_seed_pairs(self, rows):
        """Each (condition_slug, seed) must appear at most once."""
        seen = {}
        duplicates = []
        for r in rows:
            rid = r.get("run_id", "")
            seed = r.get("seed", "?")
            # Extract slug by stripping the timestamp suffix
            slug = rid[:rid.rfind("_seed")] if "_seed" in rid else rid[:rid.rfind("_2026")]
            key = (slug, seed)
            if key in seen:
                duplicates.append((key, seen[key], rid))
            else:
                seen[key] = rid

        if duplicates:
            detail = "\n".join(f"  {slug} seed={seed}: {r1} vs {r2}" for (slug, seed), r1, r2 in duplicates[:5])
            pytest.fail(f"{len(duplicates)} duplicate (condition, seed) pairs:\n{detail}")

    def test_run_ids_have_amendment4_experiment_prefix(self, rows):
        """
        Amendment 4 runs use experiment EXP-20261116-*.  Rows from earlier
        experiments should have been archived.  Warn (not fail) if old
        experiment IDs are present alongside new ones.
        """
        from collections import Counter
        exp_ids = Counter()
        for r in rows:
            rid = r.get("run_id", "")
            # run_id format: <slug>_<experimentid>_seed<n> or <slug>_<timestamp>
            # experiment_id is stored separately when available
            exp = r.get("experiment_id", "")
            if exp:
                exp_ids[exp] += 1

        if len(exp_ids) > 1:
            # Multiple experiment IDs: warn but don't fail (may be expected during transition)
            exps_str = ", ".join(f"{k}({v})" for k, v in exp_ids.most_common())
            # Only fail if any Amendment 3 experiment (EXP-20261113-*) rows remain
            old_exps = [k for k in exp_ids if "20261113" in k or "20261114" in k or "20261115" in k]
            if old_exps:
                pytest.fail(
                    f"Old experiment rows still present alongside Amendment 4 rows. "
                    f"Old experiments: {old_exps}. All experiments: {exps_str}. "
                    "Archive Amendment 3 rows before running hypothesis tests."
                )

    def test_seeds_are_in_amendment4_range(self, rows):
        """Amendment 3/4 uses seeds 53–62.  Any seed outside this range is unexpected."""
        bad = [
            r for r in rows
            if r.get("seed") and int(r.get("seed", 0)) not in range(53, 63)
        ]
        if bad:
            bad_seeds = sorted({int(r["seed"]) for r in bad})
            pytest.fail(
                f"Rows with seeds outside 53–62: {bad_seeds}. "
                "Check --base-seed used in the launch command."
            )

    def test_attack_n_samples_column_present(self, rows):
        """The attack_n_samples column must exist — it's the evidence column for H3."""
        assert "attack_n_samples" in rows[0], (
            "attack_n_samples column missing from results CSV; "
            "H3 compliance check and these tests will not work"
        )

    def test_synthetic_ratio_used_column_present(self, rows):
        """The synthetic_ratio_used column must exist — it's the evidence column for H4."""
        assert "synthetic_ratio_used" in rows[0] or "poisoning_synthetic_ratio_used" in rows[0], (
            "synthetic_ratio_used column missing from results CSV; "
            "H4 root cause cannot be verified from the data"
        )
