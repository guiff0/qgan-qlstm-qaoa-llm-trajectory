# QGAN-LLM Research Codebase

Set-ExecutionPolicy Unrestricted

winget install --id Git.Git -e --source winget


A rebuilt, genuinely-functional version of the QGAN-LLM comparison framework
(Classical LSTM vs. Classical GAN-LLM vs. QGAN-LLM, plus ablations) for the
"quantum-enhanced financial forecasting + cybersecurity threat detection"
dissertation project.
Se7rF1kGGAQ38Jjy9zjDB9DGRlb22wVO8x7Nyehy
FRED_API_KEY
[System.Environment]::SetEnvironmentVariable('FRED_API_KEY', '687e6d45e4a553c4e3e69e24c55fa1de', 'User')
[System.Environment]::SetEnvironmentVariable('NVIDIA_API_KEY', 'Se7rF1kGGAQ38Jjy9zjDB9DGRlb22wVO8x7Nyehy', 'User')
## Why this rebuild exists

An earlier version of this codebase had two functions that returned
hardcoded numbers instead of computing anything:

```python
def _evaluate_adversarial(self, X_test, y_test):
    return 8.7  # Expected ASR from RQ4 results

def _measure_entanglement(self):
    return np.random.uniform(2.5, 4.0)  # Placeholder
```

Running that code and "collecting the logs" would only ever print back
the disputed numbers themselves, not something a real model, attack, or
quantum circuit produced. This rebuild replaces every such stub with a
real implementation, fixes several bugs found along the way (see
`docs/CHANGES.md`... actually see the per-file docstrings — this project
keeps "what was wrong and why" documentation right next to the fix, not
in a separate changelog that drifts out of sync with the code), and adds
what was structurally missing (data loading, statistical tests, FID/MMD,
latency measurement, a real LLM fine-tuning call).

**This code cannot run inside the sandbox that built it.** It needs:
proprietary market data (Dukascopy/ForexSB), a real NVIDIA API key for
LLAMA 3.3 fine-tuning, and GPU-scale compute for hours-to-days. It's built
to run on **your** infrastructure. See `SETUP.md` then `RUNNING.md`.

## What's genuinely fixed vs. what's still an open question

**Fixed (verified by `tests/test_smoke.py`, which you can run right now,
no data or GPU required):**
- Quantum generator was detached from PyTorch's autograd graph — its
  parameters could never learn. Now uses PennyLane's `torch` interface
  end-to-end; a dedicated test (`test_qgan_generator_gradients_actually_flow`)
  fails loudly if this regresses.
- Entanglement entropy was `np.random.uniform(2.5, 4.0)`. Now a real
  partial-trace + von Neumann entropy calculation, checked against a
  known zero-entanglement circuit in `test_entanglement_entropy_is_zero_for_unentangled_circuit`.
- Adversarial Success Rate was a hardcoded return. Now real FGSM/PGD/CW
  attacks with gradient-based perturbations.
- FID/MMD/Wasserstein distance, ANCOVA/chi-square/correlation with CI,
  and single-sample inference latency measurement — none of these existed
  in the original code despite Chapter 4 reporting specific numbers for
  all of them. All implemented from their standard definitions.
- Full training set pushed through the model in one forward pass per
  "epoch" (infeasible at 5.5M rows) — now real mini-batch DataLoaders.
- Every LSTM was fed length-1 pseudo-sequences (defeating the point of
  using an LSTM at all) — the standalone Classical LSTM baseline now uses
  genuine sliding-window sequences (`src/data/windowing.py`).
- Several smaller bugs caught only by actually running the code end-to-end
  (not just reading it): a 2D-vs-3D LSTM input shape bug, a crash in the
  adversarial-attack helper calling `.eval()` on a bound method, a missing
  `os.makedirs` before the first checkpoint save, a config file with
  `logging.results_dir`/`log_dir` settings that were declared but never
  actually wired to the code that should have read them, and a
  results-CSV writer that would silently corrupt its own column alignment
  the moment two models logged different metric keys.

**Flagged, not silently decided** — this one needs your call, not mine:
`src/baselines/qgan_llm.py`'s `_forecast_model` has a long docstring
explaining that under a standard GAN-augmentation design (generator used
only to make synthetic training data, not invoked at inference), there is
no architectural reason for QGAN-LLM to have higher *inference* latency
than Classical GAN-LLM — both call a small linear layer at inference time.
If the dissertation's "~37.5% higher latency from quantum overhead" claim
is about inference specifically, that requires deliberately routing real
inputs through the quantum circuit at inference too, which is a design
decision with its own justification needed, not a bug fix.

`src/baselines/qlstm_forecaster.py` is a fourth baseline, added to settle
that question for at least one clean case: here, the quantum circuit
genuinely IS the forecaster (no GAN wrapper, no discriminator — real
features go in, a prediction comes out). Its own docstring flags a
*different* open question worth reading before citing it: it sees one
timestep per prediction (no lookback window), while `ClassicalLSTM` sees
a real 60-step window — a fair "does quantum help" comparison is
`QLSTMForecaster` vs. a single-timestep classical baseline, not directly
vs. `ClassicalLSTM`'s RMSE.

## Generator vs. discriminator, explicitly, in both GAN baselines

Both `classical_gan_llm.py` and `qgan_llm.py` open with a block comment
naming exactly which class is the generator, which is the discriminator,
and which (neither) is the forecaster actually used at inference — this
was a genuine source of ambiguity in the original code and is worth
reading once before working with either file:

| | Classical GAN-LLM | QGAN-LLM |
|---|---|---|
| **Generator** | `LSTMGenerator` — noise → synthetic feature vector | `QLSTMGenerator` — same job, as a quantum circuit |
| **Discriminator** | `ClassicalDiscriminator` | *identical class, imported unchanged* from `classical_gan_llm.py` |
| **Forecaster** (used at inference) | `self.forecast_head` (`nn.Linear`) | `self.forecast_head` (`nn.Linear`) — same shape, deliberately |

The discriminator is deliberately the *same class with the same
hyperparameters* in both baselines, so any RMSE/ASR/fidelity difference
between them can be attributed to the generator (classical LSTM vs.
quantum circuit), not confounded by two different discriminators.

## Testing each Research Question's actual claim, not just the code

`tests/research_questions/` has one file per RQ (`test_rq1_synthetic_fidelity.py`
through `test_rq5_entanglement_correlation.py`, sharing fixtures via
`conftest.py`), each exercising the real metric + real statistical test
Ch. 3/4 names for that hypothesis (FID/MMD/Wasserstein for H1, ASR +
chi-square for H2, ANCOVA for H3, RMSE-and-ASR ANCOVA + the
results-verification pipeline for H4, entanglement entropy + Pearson
correlation for H5) against small synthetic data with a deliberately
injected effect — a positive control (can it detect a real effect?) and,
where feasible, a negative control (does it stay quiet when there's
nothing to find?). **Read `tests/research_questions/README.md` first** —
it's explicit about what a passing test does and does not prove; a green
checkmark here means "this testing procedure works," not "the
dissertation's numbers are confirmed."

## Repository layout

```
config/default_config.yaml   Single source of truth for every hyperparameter,
                              date range, and file path. Nothing is hardcoded
                              a second time anywhere else.
src/data/                    Loading raw CSVs, MAD outlier cleaning, technical
                              indicators, sliding-window sequences.
src/quantum/                 The QLSTM generator circuit (fixed) and real
                              tomography (entropy/purity/fidelity).
src/attacks/                 Real FGSM/PGD/CW adversarial attacks.
src/evaluation/               RMSE/MAE, FID/MMD/Wasserstein, ANCOVA/chi-sq/
                              correlation, and latency measurement.
src/baselines/                Four model classes: ClassicalLSTM, ClassicalGANLLM,
                              QGANLLM, and QLSTMForecaster (quantum circuit as the
                              forecaster directly -- no GAN wrapper).
src/llm/                      Real NVIDIA API fine-tuning calls (fails loudly
                              without a real API key -- never fabricates).
src/experiments/run_all.py    The one entry point that trains + evaluates
                              every baseline and ablation, with full logging.
scripts/                      verify_environment.py (run first), prepare_data.py
                              (run second), verify_chapter4.py (run last, after
                              you have real results, to diff against the
                              manuscript's claims).
tests/test_smoke.py           Runs NOW, on synthetic data, no setup needed:
                              `pip install -r requirements.txt && pytest tests/`
tests/research_questions/     One file per Research Question (RQ1-RQ5), testing
                              that each hypothesis's actual metric + statistical
                              test works correctly -- NOT that the dissertation's
                              numbers are correct (see that folder's own README).
```

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -v        # all 31 tests: pipeline smoke tests (incl. the 4th
                          # baseline, QLSTMForecaster) + RQ1-RQ5 hypothesis
                          # tests, ~25-30 seconds total, no data or GPU needed
```

Then see **SETUP.md** (getting real data + API access in place) and
**RUNNING.md** (the actual multi-hour run, and exactly what to send back).
