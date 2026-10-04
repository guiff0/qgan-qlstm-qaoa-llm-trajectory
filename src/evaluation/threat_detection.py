"""
H5 (Correlation with QGAN Integration) operationalization: produces one
model's threat/benign detection accuracy (FPR and friends), to be
correlated against entanglement entropy (already measured per-model by
quantum/tomography.py) ACROSS model configurations -- see
scripts/run_hypothesis_tests.py for that cross-model correlation, which
is what actually tests H5's "entanglement maps correlate with
detection-accuracy improvement" claim. This module only produces one
model's number; nothing here computes a correlation by itself.

INTERIM DETECTOR, NOT THE MANUSCRIPT'S DESCRIBED PATHWAY: the
dissertation's own design for this is src/llm/threat_scoring.py, which
parses an LLM's text output into a threat probability. No LLM is
currently wired into the forecasting path (see TODO.md's LLM-wiring
item), so there is no text output for threat_scoring.py to parse yet.
Until that lands, this module trains a lightweight, classical
(RandomForest) threat/benign classifier directly on
attacks/threat_labels.py's labeled dataset instead, purely so H5 has
SOME real, honestly-labeled detection-accuracy number to test rather
than none. Swap this for the LLM-based pathway once LLM-wiring is
done -- this is explicitly a stopgap, not the final design, and should
be reported as such in the manuscript if these numbers are used.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

from ..attacks.threat_labels import build_threat_benign_dataset
from .metrics import classification_metrics

# Per the dissertation's QACL table, which specifies 0.7 for the
# threat-detection decision -- not classification_metrics' generic 0.5
# default, which is a placeholder, not a methodology-derived choice.
QACL_DETECTION_THRESHOLD = 0.7


def evaluate_threat_detection(forward_fn, X_clean: np.ndarray, y_clean: np.ndarray,
                               attack_cfg: dict, seed: int = 0,
                               n_clean: int = 1000, n_fgsm: int = 500, n_pgd: int = 500) -> dict:
    """
    forward_fn: callable(X: torch.Tensor) -> predictions, same interface
    adversarial.py's attacks already require (e.g. a model's
    ._forecast_model bound method). Caller is responsible for eval mode,
    matching compute_attack_success_rate's own convention.
    """
    X_t = torch.as_tensor(np.asarray(X_clean), dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y_clean), dtype=torch.float32)

    n_available = len(X_t)
    n_needed = n_clean + n_fgsm + n_pgd
    if n_available < n_needed:
        # Scale the recommended 1000/500/500 split down proportionally
        # rather than failing outright -- lets this run on a smaller test
        # set (e.g. a quick smoke config) instead of hard-requiring 2000 rows.
        scale = n_available / n_needed
        n_clean = max(3, int(n_clean * scale))
        n_fgsm = max(3, int(n_fgsm * scale))
        n_pgd = max(3, int(n_pgd * scale))

    dataset = build_threat_benign_dataset(
        forward_fn, X_t, y_t,
        fgsm_epsilon=attack_cfg["fgsm_epsilon"], pgd_epsilon=attack_cfg["pgd_epsilon"],
        pgd_alpha=attack_cfg["pgd_alpha"], pgd_steps=attack_cfg["pgd_steps"],
        n_clean=n_clean, n_fgsm=n_fgsm, n_pgd=n_pgd, seed=seed,
    )
    X = dataset["X"].detach().numpy()
    y_threat = dataset["y_threat"].detach().numpy()

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    cv_folds = min(3, int(y_threat.sum()), int((1 - y_threat).sum()))
    if cv_folds < 2:
        return {"fpr": float("nan"), "detector_error": "not enough samples per class for CV"}
    y_pred_proba = cross_val_predict(clf, X, y_threat, cv=cv_folds, method="predict_proba")[:, 1]

    result = classification_metrics(y_threat, y_pred_proba, threshold=QACL_DETECTION_THRESHOLD)
    result["n_clean"] = n_clean
    result["n_fgsm"] = n_fgsm
    result["n_pgd"] = n_pgd
    result["detector"] = "interim_classical_randomforest"
    return result
