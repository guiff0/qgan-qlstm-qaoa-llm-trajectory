"""Can the outcome answer the question at all? Gates that turn 'vacuously equivalent' into 'not decidable'.

Reviewer claim 5: an analysis needs outcome variation, and a fallback when its assumptions fail. Two outcomes in this
study cannot move under the current protocol, and a test on them says nothing:
  * H3: if the poisoning attack changes RMSE by far less than the equivalence margin in EVERY run, 'no_noise equals
    noisy within +/-5 points' is true of any model, so TOST 'equivalence' is vacuous.
  * H5: if the LLM detector is at chance in every run, accuracy differences between runs are sampling noise, so a
    correlation with entanglement has nothing to correlate with.
A gate returns {"decidable": bool, "gate": ..., "reason": ..., "remedy": ...}; run_hypothesis_tests attaches it to the
result and src/evaluation/verdicts.py turns decidable=False into the verdict NOT_DECIDABLE (rule R9).
"""
from __future__ import annotations

import math

import numpy as np

POISON_FLOOR_FRACTION = 0.2      # the outcome must reach at least this share of the equivalence margin somewhere
CHANCE_ALPHA = 0.05


def poisoning_gate(values, margin: float, floor_fraction: float = POISON_FLOOR_FRACTION) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"gate": "poisoning_dynamic_range", "decidable": None, "reason": "no poisoning values"}
    peak, floor = float(np.max(np.abs(v))), floor_fraction * float(margin)
    ok = peak >= floor
    return {"gate": "poisoning_dynamic_range", "decidable": bool(ok), "largest_abs_degradation": peak,
            "required_at_least": floor, "margin": float(margin), "n_values": int(len(v)),
            "reason": (f"the poisoning attack changed RMSE by at most {peak:.3g} points in any of {len(v)} runs, "
                       f"below {floor:.3g} (= {floor_fraction:.0%} of the +/-{margin:g} equivalence margin); "
                       f"'equivalence' between conditions is then true of any model and is not evidence about noise")
              if not ok else "the poisoning outcome moves enough to compare conditions",
            "remedy": "" if ok else ("raise the poisoned fraction / strength until the un-noised control degrades "
                                     "measurably (a positive control), then re-run H3; until then report H3 as not decidable")}


def detector_gate(acc, n_windows, chance: float = 0.5, alpha: float = CHANCE_ALPHA) -> dict:
    """Is the detector better than chance across runs (pooled one-sided exact binomial test)?"""
    from scipy import stats
    acc = np.asarray(acc, dtype=float)
    n = np.asarray(n_windows, dtype=float) if np.ndim(n_windows) else np.full(len(acc), float(n_windows or 0))
    m = np.isfinite(acc) & np.isfinite(n) & (n > 0)
    acc, n = acc[m], n[m]
    if len(acc) == 0:
        return {"gate": "detector_above_chance", "decidable": None, "reason": "no detector accuracy values"}
    k, big_n = int(round(float((acc * n).sum()))), int(n.sum())
    p = float(stats.binomtest(k, big_n, chance, alternative="greater").pvalue)
    sd_obs = float(acc.std(ddof=1)) if len(acc) > 1 else float("nan")
    sd_noise = math.sqrt(chance * (1 - chance) / float(np.median(n)))
    ok = p < alpha
    return {"gate": "detector_above_chance", "decidable": bool(ok), "pooled_accuracy": k / big_n, "pooled_n": big_n,
            "p_above_chance": p, "between_run_sd": sd_obs, "sampling_noise_sd": sd_noise,
            "reason": (f"the detector's pooled accuracy is {k / big_n:.3f} over {big_n:,} windows ({len(acc)} runs), "
                       f"not above chance (p = {p:.3f}); differences in accuracy between runs "
                       f"(sd {sd_obs:.3f}) are what sampling noise alone gives (sd {sd_noise:.3f}), so a correlation "
                       f"with entanglement has no signal to find") if not ok
            else f"the detector is above chance (pooled accuracy {k / big_n:.3f}, p = {p:.3g})",
            "remedy": "" if ok else ("use a detector that beats chance (e.g. few-shot prompts or a calibrated "
                                     "threshold), verify it on a positive control, then re-run H5")}
