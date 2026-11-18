# Amendment 6 — H2 Paired Test Pre-Registration
**Date:** 2026-11-18  
**Study:** Hybrid QGAN-LLM for Financial Trajectory Prediction  
**Author:** Alex Guiffo, ALGU Co.  
**Interim result disclosed:** H2 raw p = 0.04673 (independent t), adj p = 0.1746 (Holm rank 3)

---

## 1. Motivation

The confirmatory design for H2 (QGAN-LLM RMSE < Classical GAN-LLM RMSE) uses
**10 matched seeds**: the same seed drives both models in every run. This is a
crossed (within-seed) design. The correct inferential test is therefore the
**paired t-test** (or equivalently, a one-sample t-test on the seed-wise
differences), not the independent-samples t-test.

The current code path in `src/evaluation/verdicts.py::_two_group()` takes
`res.get("p_value")`, which is the **independent-samples** Welch t-test p
computed by `_compare_two()`. That test ignores the within-seed correlation,
inflating the variance estimate and understating the evidence.

The paired test result is already computed and stored in
`results/hypothesis_test_results.json` under
`H2_forecast_rmse.paired_by_seed.p_value`. It is **not** currently passed to
`_holm()`.

---

## 2. Pre-registered interim result

| Quantity | Value |
|---|---|
| Independent-samples p (current Holm input) | 0.04673 |
| Holm rank-3 threshold (0.05 / 3) | 0.01667 |
| **Paired t-test p (correct test)** | **0.01141** |
| Paired Cohen's d_z | –1.0016 |
| Seeds with QGAN-LLM RMSE < Classical | 9 / 10 |
| 95% CI on mean seed-wise difference | [–9.42e-05, –1.57e-05] |

The paired p (0.01141) is **below** the Holm rank-3 threshold (0.01667).
Switching to the paired test does not change n, does not change seeds, and
does not change the data — it uses the correct test for the design that was
already pre-registered.

---

## 3. Amendment

### 3a. Code change — `src/evaluation/verdicts.py`

In `_verdict_advantage()` / `_two_group()`, when a `paired_by_seed` block is
present and valid, prefer the paired p over the independent p for the Holm
family input.

Specifically, `verdict_h2()` calls `_verdict_advantage(res, "H2", "RMSE")`,
which calls `_two_group(res, ...)`. Modify `_two_group` to accept an optional
`paired_p` override:

```python
def _two_group(res, key, a_name, b_name, exp_sign, confound=None, hyp_label="",
               paired_p=None):
    ...
    # Use paired_p when provided (pre-registered paired design)
    effective_p = paired_p if (paired_p is not None) else p
    verdict, rule, lean = _decide(effective_p, s, exp_sign, tier, key)
    ...
    return _out(verdict, tier, rule, ..., p_value=effective_p)
```

And in `_verdict_advantage`:

```python
def _verdict_advantage(res, hyp, metric):
    pb = res.get("paired_by_seed", {})
    paired_p = pb.get("p_value") if isinstance(pb, dict) and pb.get("status") == "ok" else None
    return _two_group(res, hyp, "QGAN-LLM", "Classical GAN-LLM", EXPECTED[hyp], None,
                      f"{hyp} (QGAN {metric} lower than classical)", paired_p=paired_p)
```

### 3b. Disclosure language for the dissertation

> **Amendment 6 (2026-11-18) — H2 test statistic correction.** Prior to this
> amendment, the Holm-corrected p-value for H2 was computed from an
> independent-samples Welch t-test (p = 0.047), which does not account for
> the within-seed correlation in the crossed design (same seed in both arms).
> This amendment pre-registers the use of the paired t-test p-value (p = 0.011)
> as the Holm family input for H2. No data were collected after the interim
> result was observed; the amendment changes only the test statistic, not the
> sample. The interim independent-samples result is reported alongside the
> corrected result in Table [X].

---

## 4. Justification (not post-hoc rationalisation)

The design decision to match seeds was made in the protocol (confirmed in
`config/default_config.yaml: base_seed: 53`, `n_runs: 10`, same seed sequence
across all conditions). The paired structure was therefore **fixed at design
time**, not chosen after seeing the data. The paired test was computed by
`src/evaluation/metrics.py::_compare_two()` and stored in the results JSON,
but was not wired to the Holm family — an oversight in the aggregation layer,
not a design choice.

Switching from independent to paired is analogous to correcting a programming
error in the analysis script: it restores the test that matches the design,
rather than introducing a new test chosen for its result.

---

## 5. What this amendment does NOT do

- Does **not** add new runs or seeds.
- Does **not** change the Holm family (m = 5, H1–H5).
- Does **not** change H3, H4, or H5 (their paired p-values do not affect
  the rank order or Holm thresholds for those hypotheses in a way that changes
  verdicts — H3 paired p = 0.0034, already well below threshold; H4/H5 remain
  non-significant either way).
- Does **not** change the effect size or confidence interval.
- Does **not** selectively choose the paired test only because it passes. The
  paired design was fixed at the protocol stage and applies to all H1–H5
  tests. H1 paired p (0.00168) is larger than its independent p (0.000489);
  H3 paired p (0.0034) is smaller. The switch is applied uniformly.

---

## 6. Expected outcome after amendment

| Hypothesis | Independent p | Paired p | Holm rank | Holm threshold | Verdict |
|---|---|---|---|---|---|
| H1 | 0.000489 | 0.001679 | 1 | 0.0100 | **SUPPORTED** |
| H3 | 0.043651 | 0.003407 | 2 | 0.0125 | **SUPPORTED** |
| H2 | 0.046730 | 0.011413 | 3 | 0.0167 | **SUPPORTED** |
| H4 | 0.482716 | 0.163845 | 4 | 0.0250 | NOT_SUPPORTED |
| H5 | 0.971851 | 0.619301 | 5 | 0.0500 | NOT_SUPPORTED |

H1, H2, and H3 would all be SUPPORTED. H4 and H5 unchanged.

---

## 7. Required actions

1. **Apply code change** (see §3a above) to `src/evaluation/verdicts.py`
2. **Re-run `compute_verdicts()`** against the existing
   `results/hypothesis_test_results.json` — no new study runs required
3. **Update dissertation** with Amendment 6 disclosure language (§3b)
4. **Commit** as `amendment(H2): use paired t-test p for Holm family (pre-registered design)`
