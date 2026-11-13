# tests/research_questions/ — one file per Research Question

Five files, `test_rq1_*.py` through `test_rq5_*.py`, each exercising the
*real* statistical/ML machinery Ch.3's methodology names for a research question's
hypothesis (H1–H5, as renumbered by Amendment 3 - see the table below) — on small synthetic data, in seconds, no GPU or real
data required. Shared fixtures (a tiny synthetic dataset, tiny QGAN/attack
configs) live in `conftest.py` so each RQ file only contains what's
actually specific to that research question.

## What these DO validate

That the actual testing procedure for each hypothesis — the specific
metric, the specific statistical test, the specific comparison — is wired
correctly and can:
1. **Detect a real effect when one is deliberately injected** (a positive
   control — e.g. RQ1's test constructs one synthetic distribution close
   to real data and one far from it, and checks FID ranks them correctly).
2. **Not report an effect when there deliberately isn't one** (a negative
   control, where applicable — e.g. RQ4's ANCOVA test on data with zero
   real group effect must not report significance).

This is the standard way to validate an analysis pipeline before trusting
it on real data — the equivalent of a unit test against a known
input/output pair, applied to statistics instead of arithmetic.

## What these do NOT validate

**Whether the dissertation's actual reported numbers (FID=18.7, ASR
31.0%→8.7%, r=0.67, etc.) are correct.** That requires the real 5.5M-row
EUR/USD dataset, real GPU-scale training, and real NVIDIA API access —
none of which exist in the environment these tests run in (see the main
README.md's sandbox-constraints section). A green checkmark here means
"if you ran this exact procedure on real data and a real effect existed,
this pipeline would correctly detect it" — not "the dissertation's numbers
are confirmed." Only `scripts/verify_chapter4.py`, run against real output
logs from `python -m src.experiments.run_all`, can speak to that.

## Running

```bash
pytest tests/research_questions/ -v
```

| File | RQ | Hypothesis (Amendment 3, see HYPOTHESES_V3.md) | What it checks |
|---|---|---|---|
| `test_rq1_synthetic_fidelity.py` | RQ1 | H1 fidelity (FID) | FID/MMD/Wasserstein correctly rank distributional similarity; QGAN-vs-Classical fidelity comparison pipeline runs end-to-end |
| `test_rq4_quantum_advantage.py` | RQ2 | H2 forecast RMSE (and the pipeline for H3) | Head-to-head RMSE comparison under identical conditions; ANCOVA isolates the group effect from VIX; `results_collector` -> `verify_chapter4.compare_value()` |
| `test_rq2_adversarial_resilience.py` | RQ3 | H3 attack success, QGAN vs classical | FGSM/PGD/CW attacks produce valid ASR in [0,100]; chi-square test on attack outcomes |
| `test_rq3_noise_robustness.py` | RQ4 / RQ5 | H4 noise -> poisoning, H5 noise -> ASR | Noise-injected vs no-noise comparison on poisoning degradation (H4) and on ASR (H5) isolates the noise effect |
| `test_rq5_entanglement_correlation.py` | (dropped) | none: old H5, now exploratory X2 | Entanglement entropy responds to circuit structure; kept as a check of the optional sweep, not a research question |

(The test file names still carry the old numbers; they were not renamed so that git history and imports stay intact.)

Each file's own docstring states exactly what is and is not tested — read
that before citing a passing test suite as evidence for anything beyond
"the pipeline is mathematically sound."
