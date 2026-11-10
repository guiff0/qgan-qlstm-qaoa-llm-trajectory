"""H5 detector instrument: calibrated threshold, few-shot block, threshold-free AUROC, detectability ceiling.

WHY (results of the 45-run sweep, see the H5 line of results/hypothesis_test_results.json)
  The zero-shot LLM said "threat" for ~76 % of clean windows and ~76 % of attacked windows (FPR ~ recall, AUPRC ~ 0.5,
  pooled accuracy 0.501 over 9,168 windows). A fixed threshold of 0.70 on a score that does not separate the classes cannot
  produce accuracy above chance, and the between-run SD of accuracy (0.007) was below binomial noise (0.035). An
  outcome that is pure noise cannot be correlated with entanglement, so H5 was reported NOT_DECIDABLE.

WHAT THIS MODULE ADDS (all optional, all off unless config `llm_pipeline.threat_*` asks for them)
  * `auroc`            threshold-free separation of attacked vs clean windows from the LLM's own probabilities.
  * `pick_threshold`   threshold that maximises balanced accuracy on CALIBRATION windows; accuracy is then reported on
                       DISJOINT evaluation windows (`split_windows`), so calibration never sees the scored rows.
  * `few_shot_block`   labelled examples for the system prompt, drawn from a third disjoint set of windows that is
                       excluded from every reported metric.
  * `last_jump_z` / `reference_auroc`  detectability CEILING: a one-line statistic that needs no model. If it separates
                       attacked from clean windows and the LLM does not, the instrument is the problem, not entanglement.
  * `positive_control_summary`  the same detector on a GROSS perturbation (default 20 x the 1-minute SD). A detector that
                       cannot flag that is not a detector, and H5 is not decidable whatever the entanglement level.
None of this touches H5's directional prediction, alpha, the equivalence margin r0 or the seeds.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y_true, score) -> float:
    """Mann-Whitney AUROC with tie correction. NaN when one class is missing."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def balanced_accuracy(y_true, score, threshold: float) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(score, dtype=float) > threshold).astype(int)
    tpr = float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else float("nan")
    tnr = float(np.mean(pred[y == 0] == 0)) if (y == 0).any() else float("nan")
    return 0.5 * (tpr + tnr)


def accuracy(y_true, score, threshold: float) -> float:
    y = np.asarray(y_true, dtype=int)
    return float(np.mean((np.asarray(score, dtype=float) > threshold).astype(int) == y))


def pick_threshold(y_cal, score_cal, default: float = 0.70) -> dict:
    """Threshold maximising balanced accuracy on calibration data (ties -> the one closest to `default`).
    With one class missing, or a score that never varies, the default is kept and the reason is returned."""
    y = np.asarray(y_cal, dtype=int)
    s = np.asarray(score_cal, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(set(y.tolist())) < 2:
        return {"threshold": default, "calibrated": False, "reason": "one class missing in calibration windows"}
    u = np.unique(s)
    if len(u) < 2:
        return {"threshold": default, "calibrated": False, "reason": "the score never varies, no threshold separates"}
    cuts = (u[:-1] + u[1:]) / 2.0
    scores = np.array([balanced_accuracy(y, s, c) for c in cuts])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    pick = cuts[best[np.argmin(np.abs(cuts[best] - default))]]
    return {"threshold": float(pick), "calibrated": True, "balanced_accuracy_cal": float(scores.max()),
            "reason": "maximum balanced accuracy on calibration windows"}


def split_windows(n_windows: int, cal_fraction: float, n_shot: int, seed: int) -> dict:
    """Disjoint window positions: shots (prompt examples, never scored), calibration, evaluation."""
    rng = np.random.default_rng(seed + 7919)
    perm = rng.permutation(n_windows)
    n_shot = int(min(max(n_shot, 0), n_windows // 4))
    shots = perm[:n_shot]
    rest = perm[n_shot:]
    n_cal = int(round(cal_fraction * len(rest))) if cal_fraction > 0 else 0
    return {"shot": np.sort(shots), "cal": np.sort(rest[:n_cal]), "eval": np.sort(rest[n_cal:])}


def few_shot_block(examples: list[tuple[list[float], bool]]) -> str:
    """Labelled examples appended to the system prompt. Each is (closes, attacked?). Probabilities are the answers the
    prompt wants, not claims about the model: 0.05 for an untouched window, 0.95 for a manipulated one."""
    if not examples:
        return ""
    lines = ["", "Labelled reference windows (for calibration only; ignore them when forecasting):"]
    for closes, attacked in examples:
        lines.append("Window: " + ", ".join(f"{c:.5f}" for c in closes)
                     + f" -> Threat probability: {0.95 if attacked else 0.05:.2f}")
    return "\n".join(lines) + "\n"


def last_jump_z(window) -> float:
    """|last 1-minute change| in units of the SD of the earlier changes. The price-space attack pushes every point but
    the last one way and the last the other, so it shows up as an abnormal final step."""
    d = np.diff(np.asarray(window, dtype=float))
    if len(d) < 3:
        return float("nan")
    return float(abs(d[-1]) / (np.std(d[:-1]) + 1e-12))


def reference_auroc(clean_windows, attacked_windows) -> float:
    """Detectability ceiling: AUROC of `last_jump_z` on the very windows the LLM saw."""
    y = [0] * len(clean_windows) + [1] * len(attacked_windows)
    s = [last_jump_z(w) for w in clean_windows] + [last_jump_z(w) for w in attacked_windows]
    return auroc(y, s)


def positive_control_summary(clean_scores, pc_scores, threshold: float = 0.70, min_accuracy: float = 0.75) -> dict:
    """Same detector, gross perturbation. `passed` is the gate H5 uses: the detector must clear `min_accuracy` here."""
    c = np.asarray([x for x in clean_scores if x is not None and np.isfinite(x)], dtype=float)
    p = np.asarray([x for x in pc_scores if x is not None and np.isfinite(x)], dtype=float)
    if len(c) == 0 or len(p) == 0:
        return {"passed": None, "reason": "no positive-control scores"}
    y = np.r_[np.zeros(len(c)), np.ones(len(p))].astype(int)
    s = np.r_[c, p]
    acc = accuracy(y, s, threshold)
    return {"passed": bool(acc >= min_accuracy), "accuracy": acc, "auroc": auroc(y, s), "n": int(len(y)),
            "min_accuracy": float(min_accuracy), "threshold": float(threshold)}
