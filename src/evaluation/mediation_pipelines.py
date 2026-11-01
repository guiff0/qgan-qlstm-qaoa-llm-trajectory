"""
Pathway-specific wiring for mediation.mediation_analysis(), covering the
three pathways with resolved operational definitions:

  M2 -> DV2 (decomposes H2): Qubit Entanglement Level mediates Model Configuration's
                    effect on Attack Success Rate category.
  M3 -> DV3 (decomposes H3): Quantum Noise Injection level mediates Model Configuration's
                    effect on False Positive Rate.
  M4 -> DV4 (decomposes H4): Computational Overhead mediates Model Configuration's
                    effect on Inference Latency.

M1->DV1 and M5->DV5 are deliberately NOT wired here -- per the operational
definitions still pending confirmation (Encoding Fidelity Index and
Quantum State Feature Index), building those now would mean guessing at
constructs the methodology chapter needs to define first.

UNIT-OF-ANALYSIS CAVEAT (applies to all three pathways below, not just
one of them): IV (model configuration: classical vs. quantum) is fixed
for every sample within a single trained model's evaluation -- it does
not vary sample-to-sample the way M and DV can. For a mediation model to
be meaningful, "observations" should be INDEPENDENT EXPERIMENTAL RUNS
(e.g. k different training seeds per configuration), not individual
test-set rows from one run, since IV has zero within-run variance to
mediate. Concretely: getting a real n=200 for one of these pathways
means training ~100 classical-configuration models and ~100
quantum-configuration models (varying seed and, for M2, entangling
topology) and computing one (M, DV) pair per run -- not evaluating one
trained model on 200 test-set rows. This is a real, substantial compute
requirement the paper's methodology chapter should state explicitly, not
something these wiring functions can paper over: they accept whatever
arrays they're given and cannot tell the difference between "200
independent runs" and "200 rows from one run" (the latter would run
without error but silently violate mediation's independence assumptions).
"""
from __future__ import annotations

import numpy as np

from .mediation import mediation_analysis
from .metrics import classification_metrics

ASR_CATEGORY_CODES = {"Low": 0, "Moderate": 1, "High": 2}
ENTANGLEMENT_LEVEL_CODES = {"Low": 0, "High": 1}  # Low = linear topology, High = ring/full


def encode_entanglement_level(topology_labels) -> np.ndarray:
    """
    M2's operational definition (per Table 18): categorical High/Low,
    where High = ring or full connectivity, Low = linear (nearest-neighbor)
    topology. Accepts either the topology name directly ("ring", "full",
    "linear") or an already-labeled "High"/"Low" array.
    """
    codes = []
    for label in topology_labels:
        label_str = str(label).strip()
        if label_str in ENTANGLEMENT_LEVEL_CODES:
            codes.append(ENTANGLEMENT_LEVEL_CODES[label_str])
        elif label_str.lower() in ("ring", "full"):
            codes.append(ENTANGLEMENT_LEVEL_CODES["High"])
        elif label_str.lower() in ("linear", "none"):
            codes.append(ENTANGLEMENT_LEVEL_CODES["Low"])
        else:
            raise ValueError(
                f"Unrecognized entanglement-level label {label_str!r}; expected "
                f"'ring'/'full'/'linear' (topology name) or 'High'/'Low' (already-coded)."
            )
    return np.array(codes, dtype=float)


def encode_asr_category(asr_values, low_threshold: float = 10.0, high_threshold: float = 30.0) -> np.ndarray:
    """
    DV2's operational definition (per Table 16/earlier chapters):
    ASR is categorized Low/Moderate/High. Accepts either raw ASR
    percentages (categorized here using the thresholds already used
    elsewhere in the manuscript's Table 41-style category tables:
    Low < 10%, Moderate 10-30%, High > 30%) or already-labeled strings.
    """
    codes = []
    for value in asr_values:
        if isinstance(value, str) and value.strip() in ASR_CATEGORY_CODES:
            codes.append(ASR_CATEGORY_CODES[value.strip()])
            continue
        asr_pct = float(value)
        if asr_pct < low_threshold:
            codes.append(ASR_CATEGORY_CODES["Low"])
        elif asr_pct <= high_threshold:
            codes.append(ASR_CATEGORY_CODES["Moderate"])
        else:
            codes.append(ASR_CATEGORY_CODES["High"])
    return np.array(codes, dtype=float)


def run_m2_dv2_mediation(model_config: np.ndarray, entanglement_topology, asr_values,
                          n_bootstrap: int = 5000, seed: int = 42) -> dict:
    """
    This decomposes H2: does entanglement level (M2) mediate model configuration's (IV)
    effect on ASR category (DV2)?

    model_config: 0 = classical, 1 = quantum, one value per observation.
    entanglement_topology: topology label per observation (see
        encode_entanglement_level) -- for classical-configuration rows,
        pass the topology that WOULD apply if quantum (this study's
        published ablations already run every topology on the quantum
        side; classical rows conventionally code as the reference/Low
        level unless a specific classical analog is defined).
    asr_values: raw ASR percentage (or category label) per observation.

    NOTE: both M2 and DV2 are ordinal categories here, numerically coded
    (see mediation.py's CATEGORICAL VARIABLES caveat) -- this is an OLS
    approximation to categorical mediation, not the textbook-exact
    method (which would use ordinal/logistic mediation models).
    """
    iv = np.asarray(model_config, dtype=float)
    m2 = encode_entanglement_level(entanglement_topology)
    dv2 = encode_asr_category(asr_values)

    result = mediation_analysis(iv, m2, dv2, n_bootstrap=n_bootstrap, seed=seed)
    result["pathway"] = "M2 -> DV2 (decomposes H2)"
    result["mediator_name"] = "Qubit Entanglement Level"
    result["dv_name"] = "Attack Success Rate Category"
    return result


def run_m4_dv4_mediation(model_config: np.ndarray, computational_overhead_ms: np.ndarray,
                          latency_ms: np.ndarray, n_bootstrap: int = 5000, seed: int = 42) -> dict:
    """
    This decomposes H4: does computational overhead (M4, continuous, ms) mediate model
    configuration's (IV) effect on end-to-end inference latency (DV4,
    continuous, ms)?

    computational_overhead_ms: the quantum-vs-classical latency delta
        attributable specifically to circuit execution (state
        preparation, measurement, classical-quantum handoff) per
        observation -- e.g. per-sample timing from
        src/evaluation/latency.py's measure_inference_latency(), with
        the classical baseline's own per-sample latency subtracted off.
    latency_ms: total end-to-end inference latency per observation
        (DV4, as already measured by measure_inference_latency()).

    Both M4 and DV4 are continuous here, so this pathway is a standard
    (non-approximated) OLS mediation -- no categorical-encoding caveat
    applies, unlike M2->DV2 above.
    """
    iv = np.asarray(model_config, dtype=float)
    m4 = np.asarray(computational_overhead_ms, dtype=float)
    dv4 = np.asarray(latency_ms, dtype=float)

    result = mediation_analysis(iv, m4, dv4, n_bootstrap=n_bootstrap, seed=seed)
    result["pathway"] = "M4 -> DV4 (decomposes H4)"
    result["mediator_name"] = "Computational Overhead"
    result["dv_name"] = "Inference Latency (ms)"
    return result


def compute_fpr_per_run(y_threat_true: np.ndarray, y_threat_pred_proba: np.ndarray,
                         threshold: float = 0.7) -> float:
    """
    Thin wrapper around metrics.classification_metrics() returning just
    the FPR scalar -- one run's data in, one FPR number out. Since FPR
    is inherently an aggregate over a batch of samples (FP / (FP+TN)),
    NOT a per-sample quantity, this is the function you call ONCE PER
    RUN to get that run's single DV3 value -- see this module's
    UNIT-OF-ANALYSIS CAVEAT above. Calling this once across all runs'
    pooled samples instead of once per run would collapse the very
    per-run variation the mediation model needs.

    threshold: the QACL table's threat-probability decision threshold
    (default 0.7) -- pass explicitly rather than relying on this
    default if your methodology chapter specifies a different value.
    """
    return classification_metrics(y_threat_true, y_threat_pred_proba, threshold=threshold)["fpr"]


def run_m3_dv3_mediation(model_config: np.ndarray, noise_strength: np.ndarray,
                          fpr_per_run: np.ndarray, n_bootstrap: int = 5000, seed: int = 42) -> dict:
    """
    This decomposes H3: does quantum noise injection level (M3, continuous, e.g. the
    noise_strength config parameter) mediate model configuration's (IV)
    effect on False Positive Rate (DV3)?

    model_config, noise_strength, fpr_per_run: one value per RUN (see
        this module's UNIT-OF-ANALYSIS CAVEAT above) -- fpr_per_run
        should be computed via compute_fpr_per_run() on each run's own
        threat/benign test set (built with
        src/attacks/threat_labels.build_threat_benign_dataset), not
        pooled across runs.

    Both M3 and DV3 are continuous, so -- like M4->DV4 -- this is a
    standard (non-approximated) OLS mediation.
    """
    iv = np.asarray(model_config, dtype=float)
    m3 = np.asarray(noise_strength, dtype=float)
    dv3 = np.asarray(fpr_per_run, dtype=float)

    result = mediation_analysis(iv, m3, dv3, n_bootstrap=n_bootstrap, seed=seed)
    result["pathway"] = "M3 -> DV3 (decomposes H3)"
    result["mediator_name"] = "Quantum Noise Injection Level"
    result["dv_name"] = "False Positive Rate"
    return result
