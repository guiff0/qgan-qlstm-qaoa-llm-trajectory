"""
Threat Latency-Impact Scoring (TaLIS) -- this study's own proposed
metric (see the dissertation's "Threat Latency-Impact Scoring (TaLIS)"
definition, Ch.1).

======================================================================
STATUS BEFORE THIS FILE
======================================================================
TaLIS is described in the manuscript as "the novel scoring framework
proposed in this study." Before this file, there was no formula, no
implementation, and no reported score anywhere in the codebase or the
results chapter. This is a first concrete operationalization of the
prose definition below -- not a transcription of a pre-existing
formula, because none existed to transcribe. If a specific formula
exists elsewhere (advisor notes, an earlier draft), replace this one --
but some formula had to exist before TaLIS could appear in a results
row at all.

======================================================================
THE FORMULA
======================================================================
The manuscript's own definition: "The operational risk of a cyber
threat is not solely determined by its potential damage but also by
the speed at which it can be identified and neutralized." That names
two ingredients -- IMPACT and LATENCY -- combined into one risk score.
This module operationalizes each from metrics this pipeline already
computes per model, rather than introducing new measurements:

  IMPACT   := Attack Success Rate (ASR), as a fraction in [0, 1].
              ASR is already this study's measure of "how often does a
              threat get through" -- a direct, already-validated proxy
              for potential damage. It is NOT a dollar figure: this
              codebase has no position-sizing/P&L model to convert a
              successful attack into an actual financial loss amount,
              so ASR stands in for damage magnitude, not damage in
              dollars. State this substitution explicitly wherever
              TaLIS is reported in the manuscript.

  LATENCY  := this model's mean single-sample inference latency
              (latency_mean_ms), relative to a configurable reference
              threshold (latency_reference_ms). This is a real
              simplification, noted here rather than silently made:
              TaLIS as defined is about DETECTION latency (time from a
              threat's occurrence to its identification in a live
              system). This pipeline doesn't run a streaming detector
              with a measurable end-to-end detection delay --
              single-sample inference latency (how long one forward
              pass takes) is the closest thing it actually measures.

    TaLIS = impact_weight * ASR_fraction
            + latency_weight * min(latency_mean_ms / latency_reference_ms, latency_cap)

Both weights default to 0.5 -- equal weighting, since the manuscript's
prose doesn't assert either ingredient matters more than the other.
latency_reference_ms defaults to 40ms, reusing the same reference this
project's own Optuna-objective description elsewhere already uses
(RMSE + 0.5*ASR + 0.1*(latency/40)) rather than inventing a new
threshold. latency_cap (default 5.0) bounds how much a single very slow
model can dominate the score.

Score range: with ASR in [0,1] and default weights/cap, TaLIS falls in
[0, 0.5 + 0.5*5.0] = [0, 3.0]. Lower is better (less operational risk).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TaLISConfig:
    impact_weight: float = 0.5
    latency_weight: float = 0.5
    latency_reference_ms: float = 40.0
    latency_cap: float = 5.0


def compute_talis(asr, latency_mean_ms, config: TaLISConfig = None) -> float:
    """
    asr: Attack Success Rate. Accepts either a fraction in [0, 1] or a
    percentage in [0, 100] -- values > 1 are assumed to be a percentage
    and divided by 100 (ASR is reported both ways across this codebase
    depending on the code path, e.g. 8.7 meaning 8.7% vs. 0.087).
    latency_mean_ms: this model's measured mean single-sample inference
    latency, in milliseconds (as produced by measure_inference_latency).

    Returns float('nan') if either input is missing/NaN -- e.g.
    Classical LSTM reports asr=NaN since the adversarial-attack
    evaluation doesn't apply to it; TaLIS is correspondingly undefined
    for that row, not silently zero.
    """
    cfg = config or TaLISConfig()
    if asr is None or latency_mean_ms is None:
        return float("nan")
    asr = float(asr)
    latency_mean_ms = float(latency_mean_ms)
    if math.isnan(asr) or math.isnan(latency_mean_ms):
        return float("nan")

    asr_fraction = asr / 100.0 if asr > 1.0 else asr
    latency_ratio = min(latency_mean_ms / cfg.latency_reference_ms, cfg.latency_cap)
    return cfg.impact_weight * asr_fraction + cfg.latency_weight * latency_ratio
