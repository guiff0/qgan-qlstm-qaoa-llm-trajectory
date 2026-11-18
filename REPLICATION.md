# Replication Package

This document describes how to reproduce the results reported in the dissertation
"Quantum Generative Adversarial Networks for Financial Time-Series Prediction."

---

## 1. Repository layout

```
.
├── config/                  # Experiment configuration
├── docs/
│   └── m8_fix.md            # M8 / Amendment 5 attack-fix write-up (traceability)
├── results/
│   ├── snapshots/
│   │   └── all_results_20261118T071700Z.csv   # canonical result snapshot (40 rows)
│   ├── poisoning_calibration_20261118T192121Z.json
│   └── hypothesis_test_results.json
├── scripts/
│   └── audit_checkpoints.py  # design-fact verifier (see §4)
├── src/
│   └── evaluation/
│       └── verdicts.py       # Holm / verdict logic (Amendment 6 patch here)
├── amendment6_h2_paired_test.md   # H2 paired-test pre-registration
├── launch_amendment5.ps1          # Windows launch script (Amendment 5)
└── REPLICATION.md            # this file
```

---

## 2. Canonical result snapshot

| Field | Value |
|---|---|
| File | `results/snapshots/all_results_20261118T071700Z.csv` |
| SHA-256 | `77acf434415a89cf38bc4c2991b359faffc80fae6fe84a17ff2fbb2694f7557e` |
| Rows | 40 (20 Classical GAN-LLM + 20 QGAN-LLM) |
| Seeds | 53–62 (10 seeds × 2 arms) |
| Git commits included | `59e77e23` (22 runs), `905bab81` (18 runs) |

Verify integrity before any analysis:

```bash
sha256sum -c <<'EOF'
77acf434415a89cf38bc4c2991b359faffc80fae6fe84a17ff2fbb2694f7557e  results/snapshots/all_results_20261118T071700Z.csv
EOF
```

---

## 3. Reproducing the full study

### Prerequisites

- Python ≥ 3.10
- Dependencies: `pip install -r requirements.txt`
- Windows (PowerShell) or Linux/macOS (Bash)

### Run all 40 seeds (Amendment 5 attack, unidirectional adversarial shift)

**Windows (PowerShell):**
```powershell
.\launch_amendment5.ps1 -Workers 4
```

**Linux / macOS:**
```bash
bash launch_amendment5.sh
```

`-Workers 4` is recommended for a 4-core / 8-logical-thread machine.
The quantum simulator is mostly single-threaded, so 4 workers avoids
hyperthreading overhead without under-utilising the cores.

### Regenerate hypothesis test results

```bash
python -m src.evaluation.run_verdicts \
    --snapshot results/snapshots/all_results_20261118T071700Z.csv \
    --out results/hypothesis_test_results.json
```

The key outputs are:

| Hypothesis | Test | p-value | Holm threshold | Verdict |
|---|---|---|---|---|
| H1 | paired t-test (seed-matched) | — | 0.05/1 = 0.0500 | CONTRADICTED |
| H2 | paired t-test (Amendment 6) | 0.01141 | 0.05/3 = 0.0167 | SUPPORTED |
| H3 | paired t-test (seed-matched) | 0.003407 | 0.05/2 = 0.0250 | CONTRADICTED |
| H4 | — | — | — | INCONCLUSIVE |
| H5 | — | — | — | INCONCLUSIVE |

H2 uses the paired p from `results/hypothesis_test_results.json`
(`paired_by_seed.p_value`). See `amendment6_h2_paired_test.md` for
the full pre-registration rationale.

---

## 4. Design-fact verification

`scripts/audit_checkpoints.py` checks three design facts referenced
in the manuscript:

```bash
# Verify matched training budgets (next_epoch == 30 for all checkpoints)
python -m scripts.audit_checkpoints --check epochs_equal

# Verify no per-arm hyperparameter search artifacts exist
python -m scripts.audit_checkpoints --check no_per_arm_tuning

# Print per-checkpoint generator_params and pin the 155x capacity ratio
python -m scripts.audit_checkpoints --check capacity_ratio

# Run all three checks
python -m scripts.audit_checkpoints --check all
```

Expected capacity pair pinned by `capacity_ratio`:

| Model | generator_params |
|---|---|
| Classical GAN-LLM (largest) | 87,072 |
| QGAN-LLM (modal) | 560 |
| Ratio | ≈ 155× |

---

## 5. M8 / poisoning positive control (Amendment 5 traceability)

The poisoning attack was redesigned between Amendment 3 and Amendment 5.
See `docs/m8_fix.md` for the full write-up:

- Root cause: `inject_triggered_shift` used `sign(z − median(z))`,
  producing a mean-zero shift that ~50 % of seeds negated.
- Fix: unidirectional adversarial shift along `u = X^T residuals`
  (steepest-ascent direction for ridge MSE).
- Recalibration: `shift_sd` raised from 1.0 to 2.0 after the
  unidirectional fix moved the sign-flip threshold.

Calibration freeze file:
`results/poisoning_calibration_20261118T192121Z.json`
(chosen_shift_sd = 2.0, pilot_seeds = [100, 101, 102, 103, 104])

---

## 6. Amendment log

| Amendment | Commit | Summary |
|---|---|---|
| Amend. 3 | `c379116` | Replaced mean-zero sign-mixed attack with adversarial unidirectional shift |
| Amend. 5 | `6e0e366`, `d3daaa1`, `6bc88f7` | Recalibrate shift_sd 1.0 → 2.0; launch scripts |
| Amend. 6 | `494c458`, `18cfe63` | H2 Holm input: independent-samples p → paired t-test p (pre-registered) |

---

## 7. Environment

```
Python:   see requirements.txt
Platform: Windows 11 (primary); Linux-compatible
Seeds:    53–62 (main study), 100–104 (M8 calibration pilots)
```

---

## 8. Contact

Questions about this replication package:
see the dissertation manuscript or open an issue in this repository.
