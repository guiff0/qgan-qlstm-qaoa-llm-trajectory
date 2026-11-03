# Pre-registration: quantum feature maps vs matched classical controls (EUR/USD, one-step-ahead)

Freeze with `python -m scripts.freeze_preregistration` BEFORE the confirmatory runs. After the freeze,
`--check` lists every changed file/margin. Anything changed afterwards is reported as a deviation.

## Questions
- **Q1 (forecasting, feature map).** Does a fixed random PQC feature map improve one-step forecasts beyond
  (a) inputs only and (b) a classical random feature map of the same width and random-parameter count?
- **Q2 (generator on the inference path).** With forecast = linear([X, G(X)]) for both generators, same rows, same
  head, same evaluation subsample, does the QGAN generator differ from the classical generator? Run against BOTH
  the large classical generator (87,072 parameters) and the capacity-matched one (896 vs ~912).

## Design (fixed)
- Target: next-close CHANGE (`forecast_target: delta`); a model that learns nothing scores exactly persistence.
- Chronological split, no look-ahead; same test rows for every arm; ridge penalty chosen on validation rows only.
- Q1: `python -m scripts.run_feature_map_experiment --seeds 0 1 2 3 4 5 6 7 8 9 --n-qubits 8` and again with
  `--n-qubits 12`. Seeds change only the random feature-map parameters, not the data split; there is ONE test
  period, so results are conditional on that period (stated as a limitation).
- Q2: `python -m scripts.run_full_study` (conditions include `Classical GAN-LLM (matched)`), 10 seeds per condition,
  `generator_input_dims` = `qgan_llm.n_qubits` = 20 so both generators receive the same 20 features.

## Three-arm paired design (Q1) and primary metrics
Arm 1 inputs only (PCA features x_t -> linear head); Arm 2 classical random-feature map phi_classical(x_t) (same width
and same number of random parameters); Arm 3 quantum circuit feature map phi_quantum(x_t) (Pauli-Z expectation values).
Same chronological splits, same ridge/tuning protocol on validation rows only, same training budget, same random seeds
fed to all three arms, repeated walk-forward windows where the data allow. The circuit must lie on the tested
inference path: forecast = w^T phi(x_t) + b.

Primary metrics (the only ones used to declare a result): persistence skill (RMSE / persistence RMSE, < 1 beats the
naive forecast), RMSE, adversarial ASR, synthetic-data fidelity (FID), latency (classical simulation time only).
Secondary, exploratory: H2/H5 correlations and the M2 mediation chain.

## Retired on 2 Nov 2026 (do not cite)
- Manuscript numbers RMSE 0.523/0.492/0.364/0.452/0.350 and ASR 31.0%/8.7%/11.3%/12.1% (no logged run produced them).
- "ANCOVA confirmed QGAN improvement (p = 0.003)" and any VIX-covariate/chi-square result (not reproducible from the code).
- QAR and CQRS composites, QGOM, CTR and TaLIS as evidence (clamp/percent-fraction artifacts, 10x5 rough estimates).
  The functions remain in the code for audit only; they are no longer written to hypothesis_test_results.json.
- Any "regulatory latency mandate" (no SEC numeric threshold exists; 40 ms is a study-internal reference value).

## Decision rules (fixed before any confirmatory result is seen)
Margins are those in `EQUIV_MARGINS` (`scripts/run_hypothesis_tests.py`), alpha = 0.05, recorded in the freeze file:
RMSE 0.02 x persistence RMSE; ASR 5 percentage points; FID 10% of the classical reference; poisoning 5 points.
- **Quantum advantage supported** only if the quantum arm beats BOTH controls on RMSE (paired DM, two-sided
  p < 0.05, same sign in >= 8 of 10 seeds) AND its persistence skill is > 0 (RMSE/persistence < 1).
- **Equivalence supported** if the TOST for |difference| < margin rejects at alpha for the quantum arm vs each control.
- **Inconclusive** otherwise. Inconclusive is reported as inconclusive, never as a trend.
- Two comparators (large and matched classical) are both reported. No outcome is dropped, no seed excluded after seeing results.
- ASR and latency are secondary: reported with CIs, not used to declare advantage. Latency is classical simulation
  time and carries no hardware claim.

## Known limitations (stated up front)
- Quantum components are classically simulated; nothing here speaks to runtime advantage.
- Q1 uses an 8/12-qubit feature map, not the 20-qubit QGAN; claims are scoped to what was run.
- Q1 has one test period; the seed-to-seed spread reflects feature-map randomness only.
- H2 and H5 correlations over the entanglement sweep rest on few levels; reported as exploratory.
