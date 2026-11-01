# Revised hypothesis statements (DRAFT -- freeze before the confirmatory runs)

**Status of this draft.** Written 2026-11-01 after a first round of results in which most tests came back
UNTESTABLE or INCONCLUSIVE (n = 1 per group, head-only forecast, 4-11 model configs treated as data points).
Those results are *design/data failures*, not evidence for or against the hypotheses, so the directional
predictions below are **unchanged**. What changes is (a) the claims are scoped to what the design can show, and
(b) each between-model hypothesis gets a pre-specified **equivalence** companion, so a well-powered null is a
reportable finding instead of "p > .05". State this in the paper (see "Disclosure").

Everything marked **[FIX]** must be decided and written down *before* the confirmatory runs. Values shown are
defaults in `scripts/run_hypothesis_tests.py` (`EQUIV_MARGINS`); change them there or with `--equiv-margin`.

## Conventions
* Unit of analysis: one independent **run** = one (config, seed). Re-evaluating a checkpoint is not a new run.
* alpha = 0.05 two-sided for superiority tests; Holm correction over the confirmatory family.
* "Quantum" = the specific 12/20-qubit circuit architecture in this repository, on EUR/USD 1-min data,
  2012-2025. No claim about quantum models in general.
* Equivalence: TOST (Welch) on the mean difference; equivalent iff the 90 % CI of the difference lies inside
  +/- margin. Correlation equivalence: 90 % Fisher-z CI of r inside +/- r0.

## H1 -- Synthetic-data fidelity (QGAN-LLM vs Classical GAN-LLM)
* **H1a (superiority, directional).** The primary QGAN-LLM configuration produces synthetic data with lower
  Frechet distance (FID) than the primary Classical GAN-LLM, controlling for market volatility (VIX) in an ANCOVA.
* **H1b (equivalence).** The mean FID difference lies within +/- delta_FID.
* Design: primary configs only (`qgan_llm`, `classical_gan_llm`), >= 5 seeds each, same data and eval subsample.
* **[FIX]** delta_FID = 10 % of the classical mean FID (default). Reference-dependent; if you prefer a fixed
  number, use an absolute margin justified by what FID difference is visible in downstream use.
* Reading: H1a supported -> quantum generator better. H1b supported (H1a not) -> comparable fidelity.
  Neither -> underpowered; report the CI.

## H2 -- Entangling structure and adversarial resilience
* **H2a (directional).** Across the controlled sweep (L0 none ... L6 full x4; qubits, noise, data fixed), higher
  entangling structure (measured entanglement entropy) is associated with lower attack success rate (r < 0).
* **H2b (no practically relevant relationship).** |r| < r0.
* Design: 7 levels x >= 6 seeds = 42 independent runs. L0 (no CNOTs) is the zero-entanglement control.
  Manipulation check first: the designed level must raise the *measured* entropy, otherwise H2 is not about
  entanglement. Report as "entangling structure", not "entanglement alone" (depth also changes parameter count).
* **[FIX]** r0 = 0.30 (default): correlations smaller than this are treated as practically irrelevant.

## H3 -- Noise injection and poisoning resistance
* **H3a (directional).** The no-noise QGAN-LLM shows a larger poisoning RMSE degradation than the matched primary
  QGAN-LLM (noise > 0). The two configs differ **only** in `noise_strength`.
* **H3b (equivalence).** The degradation difference lies within +/- delta_pois.
* Design: >= 5 seeds per side; qubits, topology, depth identical.
* **[FIX]** delta_pois = 5 percentage points of degradation (default).

## H4 -- Forecasting advantage of the quantum generator (feature-map design)
* **H4a (directional).** With the generator on the forecast path (`forecast_path: generator`, forecast =
  linear([X, G(X)]) for both models, same rows, same head), QGAN-LLM has lower one-step RMSE and lower ASR than
  Classical GAN-LLM.
* **H4b (equivalence / non-inferiority).** RMSE and ASR differences lie within +/- delta_RMSE, delta_ASR.
* Scope statement (put it in the hypothesis): the claim concerns this feature-map design, not quantum models in
  general. Runs with `forecast_path=head_only` cannot test H4 (RMSE identical by construction) and are excluded.
* Also report skill vs persistence (rmse / persistence_rmse), because persistence was the strongest forecaster in
  earlier logs; a "quantum advantage" over a classical model that itself cannot beat persistence is not
  meaningful.
* **[FIX]** delta_RMSE = 2 % of persistence RMSE (default, model-independent scale); delta_ASR = 5 percentage points.

## H5 -- Entangling structure and threat-detection accuracy
* **H5a (directional).** Higher entangling structure (sweep levels) -> higher threat-detection accuracy (r > 0),
  using the LLM-based detector (`llm_threat_accuracy`), not the interim random-forest stopgap.
* **H5b (no practically relevant relationship).** |r| < r0 (as H2b).
* Design as H2 (same sweep runs).

## M2 -- Mediation by entanglement
* **M2.** The effect of entangling configuration on ASR is mediated by measured entanglement entropy
  (bootstrap 95 % CI of the indirect effect excludes 0).
* Design: sweep runs only (>= 30 independent runs). Config-level data (n = 11) are exploratory and must be
  labelled so. With the sweep the mediator path is identified only if the manipulation check passes.

## Sample size: what equivalence costs (read before choosing margins)
Approximate runs **per group** for 80 % power to show equivalence when the true difference is 0,
alpha = .05, by margin relative to the run-to-run SD (sigma) of the metric:

| margin / sigma | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| runs per group | ~69 | ~17 | ~8 | ~5 |

With 5 seeds you can only demonstrate equivalence for margins of about 2 sigma or more. Estimate sigma from a
pilot (3 seeds), then pick margins you can defend substantively *and* afford; if the margin you can defend is
smaller than 1.5 sigma and you cannot afford the seeds, report the CI and call the result inconclusive rather than
widening the margin.

## Disclosure (suggested text for the paper)
"Directional hypotheses H1-H5 and M2 were specified before data collection. After a first pilot showed the
design could not test them (single runs per condition; forecast independent of the generator; configurations
treated as observations), we (i) fixed the design (matched primary configurations, generator on the forecast
path, a controlled entanglement sweep with a zero-entanglement control, >= 5 seeds per condition) and (ii) added
pre-specified equivalence tests with margins fixed before the confirmatory runs. No hypothesis direction was
changed in light of results; pilot results are not part of the confirmatory analysis."
(Only write this if it is true: do not change directions or margins after seeing confirmatory results.)

## Checklist before the confirmatory runs
- [ ] margins (delta_FID, delta_RMSE, delta_ASR, delta_pois, r0) chosen and committed (git commit / timestamped file)
- [ ] `forecast_path: generator` in both GAN sections; Classical LSTM and QLSTM budgets documented
- [ ] sweep generated (`python -m scripts.gen_entanglement_sweep --apply`) and run with >= 6 seeds per level
- [ ] primary QGAN-LLM, Classical GAN-LLM and `QGAN-LLM (no noise)` run with >= 5 seeds
- [ ] `python -m scripts.run_hypothesis_tests` (reports superiority verdicts and, beside them, equivalence)
