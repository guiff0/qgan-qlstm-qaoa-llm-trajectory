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


def poisoning_gate(values, margin: float, floor_fraction: float = POISON_FLOOR_FRACTION,
                   positive_control=None, pc_floor: float | None = None) -> dict:
    """`positive_control` (protocol v2): per-run degradation of the SAME attack on a head trained on real rows only. It
    does not involve the generator or the noise level, so it asks only "can this attack hurt anything?". Its median must
    reach `pc_floor` (default: the equivalence margin). Without it (legacy protocol) only the dynamic-range rule applies."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"gate": "poisoning_dynamic_range", "decidable": None, "reason": "no poisoning values"}
    if positive_control is not None:
        pc = np.asarray([x for x in positive_control if x is not None and np.isfinite(x)], dtype=float)
        if len(pc):
            floor_pc = float(margin if pc_floor is None else pc_floor)
            med = float(np.median(pc))
            if med < floor_pc:
                return {"gate": "poisoning_positive_control", "decidable": False, "positive_control_median": med,
                        "required_at_least": floor_pc, "n_values": int(len(pc)), "margin": float(margin),
                        "reason": (f"the positive control (the same attack on a head trained on real rows only) degraded RMSE by a "
                                   f"median of {med:.3g} points over {len(pc)} runs, below {floor_pc:g}; an attack that cannot hurt a "
                                   f"plain head cannot show that noise protects a generator"),
                        "remedy": ("raise poisoning.shift_sd / fraction using scripts/calibrate_poisoning.py on pilot seeds "
                                   "(real-only head, no generator), freeze it, then re-run H3")}
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


def detector_gate(acc, n_windows, chance: float = 0.5, alpha: float = CHANCE_ALPHA,
                  pc_passed=None, pc_min_share: float = 0.8) -> dict:
    """Is the detector better than chance across runs (pooled one-sided exact binomial test)?
    `pc_passed` (optional, per-run booleans from the positive control: the same detector on a GROSS perturbation) must be
    True in at least `pc_min_share` of the runs; a detector that misses an obvious spike has nothing to correlate."""
    from scipy import stats
    if pc_passed is not None:
        flags = [bool(x) for x in pc_passed if x is not None and not (isinstance(x, float) and np.isnan(x))]
        if flags and (sum(flags) / len(flags)) < pc_min_share:
            return {"gate": "detector_positive_control", "decidable": False, "pc_pass_share": sum(flags) / len(flags),
                    "n_runs": len(flags),
                    "reason": (f"the detector flagged the gross-perturbation positive control in only {sum(flags)} of {len(flags)} "
                               f"runs; it does not detect even an obvious manipulation, so its accuracy cannot track entanglement"),
                    "remedy": "change the detector (few-shot examples, a different model, a feature-assisted prompt), keep the positive control, re-run H5"}
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
