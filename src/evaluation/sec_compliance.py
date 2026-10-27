"""
SEC-oriented TECHNICAL-CONTROL SCREEN for a completed pipeline run.

WHAT THIS IS / IS NOT
---------------------
This module screens one run's logged evidence against the *kinds* of controls
that SEC Rule 15c3-5 (Market Access Rule, 17 CFR 240.15c3-5) and related
model-risk expectations call for. It is NOT legal advice and NOT a
certification. Rule 15c3-5 applies to broker-dealers with market access and
requires documented, reasonably designed financial and regulatory risk
management controls, supervisory procedures, annual review and a CEO
certification. A research model that never routes orders is not a regulated
system; the screen shows what a reviewer could and could not rely on.

Two points the dissertation draft got wrong and this screen does not repeat:
  * 15c3-5 contains NO numeric latency threshold. A 40 ms (or sub-millisecond)
    figure is at most a study-internal target (config: sec_compliance.
    internal_latency_target_ms), reported as such, never as a "mandate".
  * Reg SCI applies to SCI entities, not to a research model.

STATUSES
--------
PASS            evidence in the run supports the control
FAIL            evidence in the run contradicts the control
NOT_EVIDENCED   the run did not measure what the control needs
MANUAL          a human attestation/document is required; code cannot decide
NOT_IMPLEMENTED the codebase has no such control (e.g. pre-trade order limits)
"""
from __future__ import annotations

import math
from typing import Any

PASS, FAIL, NOT_EVIDENCED, MANUAL, NOT_IMPLEMENTED = (
    "PASS", "FAIL", "NOT_EVIDENCED", "MANUAL", "NOT_IMPLEMENTED")

DEFAULTS: dict[str, Any] = {
    "max_asr_percent": 5.0,                 # study-internal tolerance
    "min_attack_samples": 1000,             # fewer -> ASR not meaningful
    "min_detector_accuracy": 0.90,
    "max_detector_fpr": 0.05,
    "max_rmse_over_persistence": 1.0,       # must not be worse than naive
    "max_membership_accuracy": 0.60,        # 0.5 = indistinguishable
    "internal_latency_target_ms": 40.0,     # NOT an SEC requirement
    "required_attestations": [
        "risk_control_documentation", "supervisory_procedures",
        "annual_review_date", "ceo_certification"],
}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _c(cid, title, status, evidence, ref):
    return {"id": cid, "title": title, "status": status, "evidence": evidence, "reference": ref}


def evaluate_run(run: dict, cfg: dict | None = None, attestations: dict | None = None) -> dict:
    """run: the parsed run JSON ({run_id, environment, config_used, final_metrics})."""
    p = {**DEFAULTS, **(cfg or {})}
    m = run.get("final_metrics", {}) or {}
    env = run.get("environment", {}) or {}
    name = m.get("model_type", run.get("run_id", "?"))
    checks = []

    # DOC-1 provenance
    commit, dirty, seed = env.get("git_commit"), env.get("git_dirty"), m.get("seed")
    if not commit or commit == "not_a_git_repo" or seed is None:
        st, ev = FAIL, f"commit={commit!r}, seed={seed!r}: run cannot be tied to code"
    elif dirty is None:
        st, ev = NOT_EVIDENCED, f"commit {commit[:8]} recorded but no git_dirty flag (uncommitted edits cannot be ruled out); seed={seed}"
    else:
        st = FAIL if dirty else PASS
        ev = f"commit {commit[:8]}, dirty={dirty}, seed={seed}"
    checks.append(_c("DOC-1", "Run is reproducible from recorded code, seed and versions", st, ev,
                     "15c3-5(b)/(c) documented controls; record-keeping"))

    # MR-1 model skill vs naive baseline
    r = _num(m.get("rmse_over_persistence"))
    if r is None:
        st, ev = NOT_EVIDENCED, "no persistence baseline logged"
    else:
        st = PASS if r <= p["max_rmse_over_persistence"] else FAIL
        ev = f"RMSE is {r:.4g}x the one-step persistence RMSE (limit {p['max_rmse_over_persistence']})"
    checks.append(_c("MR-1", "Forecast beats a naive persistence baseline", st, ev,
                     "model-risk validation (SR 11-7 style); 15c3-5(c) financial risk controls"))

    # CTRL-1 adversarial robustness
    asr, n = _num(m.get("asr")), _num((m.get("asr_breakdown") or {}).get("n_samples"))
    if asr is None:
        st, ev = NOT_EVIDENCED, "ASR not measured"
    elif n is not None and n < p["min_attack_samples"]:
        st, ev = NOT_EVIDENCED, f"ASR={asr:.4g}% but only {int(n)} attacked samples (< {p['min_attack_samples']}): one sample moves ASR by {100/n:.2g} points"
    else:
        st = PASS if asr <= p["max_asr_percent"] else FAIL
        ev = f"ASR={asr:.4g}% over {int(n) if n else '?'} samples (tolerance {p['max_asr_percent']}%)"
    checks.append(_c("CTRL-1", "Adversarial input robustness (FGSM/PGD/CW directional flips)", st, ev,
                     "15c3-5(c)(2) regulatory/erroneous-order controls (analogue)"))

    # CTRL-2 threat detection
    acc, fpr = _num(m.get("accuracy")), _num(m.get("fpr"))
    det = m.get("detector", "")
    if acc is None:
        st, ev = NOT_EVIDENCED, "no detector result"
    else:
        ok = acc >= p["min_detector_accuracy"] and (fpr is None or fpr <= p["max_detector_fpr"])
        st = PASS if ok else FAIL
        ev = f"detector={det}, accuracy={acc:.3f}, FPR={fpr}"
        if "interim" in str(det):
            ev += "; interim classical detector, identical across models so it does not test this model"
            if st == PASS:
                st = NOT_EVIDENCED
    checks.append(_c("CTRL-2", "Threat detection performance", st, ev, "15c3-5(b) systemic risk monitoring (analogue)"))

    # PRIV-1 distinguishability / inversion
    mi, chance = _num(m.get("model_inversion_accuracy")), _num(m.get("model_inversion_chance_level")) or 0.5
    if mi is None:
        st, ev = NOT_EVIDENCED, "no membership/inversion accuracy logged"
    else:
        st = PASS if mi <= p["max_membership_accuracy"] else FAIL
        ev = f"accuracy={mi:.3f} (chance {chance}); higher = synthetic data trivially distinguishable from real, so it is not a privacy-preserving substitute"
    checks.append(_c("PRIV-1", "Synthetic data not trivially distinguishable from real (membership test)", st, ev,
                     "Reg S-P safeguards expectation (analogue); GDPR/CCPA minimisation"))

    # LAT-1 latency (internal target only)
    lat, scope = _num(m.get("latency_mean_ms")), m.get("latency_scope")
    if lat is None:
        st, ev = NOT_EVIDENCED, "latency not measured"
    else:
        scope = scope or ("quantum_circuit_in_path" if "QLSTM" in str(name) else
                          "forecast_head_only" if "GAN" in str(name) else "unknown")
        within = lat <= p["internal_latency_target_ms"]
        if scope == "forecast_head_only":
            st = NOT_EVIDENCED
            ev = f"{lat:.4g} ms times only a linear head; the quantum generator is not on the inference path, so this says nothing about quantum latency"
        else:
            st = PASS if within else FAIL
            ev = f"{lat:.4g} ms ({scope}) vs STUDY-INTERNAL target {p['internal_latency_target_ms']} ms (no SEC numeric threshold exists)"
    if (_num(m.get("concurrent_jobs")) or 1) > 1:
        ev += f"; measured with {int(m['concurrent_jobs'])} pipeline jobs running concurrently, so timing is inflated by CPU contention"
        if st == PASS:
            st = NOT_EVIDENCED
    checks.append(_c("LAT-1", "Latency vs study-internal target", st, ev, "none: internal target, not a rule requirement"))

    # MET-1 metric validity flags
    flags = []
    if (_num(m.get("qgom")) == 0.0):
        flags.append("QGOM exactly 0.0 (old clamp artifact; below noise floor, not 'no obfuscation')")
    if m.get("ctr") is not None and _num(m.get("ctr")) is None:
        flags.append("CTR undefined (no identifiable decay)")
    if "ROUGH" in str(m.get("ctr_method", "")):
        flags.append("CTR is a 10-trajectory x 5-step rough estimate")
    if _num(m.get("talis_score")) is not None and _num(m.get("asr")) is not None and m.get("talis_unit") != "percent":
        flags.append("TaLIS stored before the percent/fraction fix; recompute")
    checks.append(_c("MET-1", "Reported resilience metrics are interpretable", FAIL if flags else PASS,
                     "; ".join(flags) or "no validity flags", "supervisory review of control outputs"))

    # NI-1 controls the codebase does not contain
    checks.append(_c("NI-1", "Pre-trade order/credit limits, kill switch, erroneous-order blocks", NOT_IMPLEMENTED,
                     "The pipeline forecasts prices; it has no order path or hard limits. Required for any live market access.",
                     "15c3-5(c)(1)-(2)"))

    # MAN-1 attestations
    att = attestations or {}
    missing = [k for k in p["required_attestations"] if not att.get(k)]
    checks.append(_c("MAN-1", "Documentation, supervisory procedures, annual review and CEO certification",
                     MANUAL if missing else PASS,
                     "missing attestations: " + ", ".join(missing) if missing else "all attestations provided",
                     "15c3-5(e)(1)-(2)"))

    counts = {s: sum(c["status"] == s for c in checks) for s in (PASS, FAIL, NOT_EVIDENCED, MANUAL, NOT_IMPLEMENTED)}
    overall = ("NOT_READY" if counts[FAIL] or counts[NOT_IMPLEMENTED]
               else "INCOMPLETE" if counts[NOT_EVIDENCED] or counts[MANUAL] else "SCREEN_PASSED")
    return {"run_id": run.get("run_id"), "model": name, "overall": overall, "counts": counts, "checks": checks,
            "disclaimer": "Technical screen of logged evidence; not legal advice or a compliance certification."}


def cross_run_checks(results: list[dict]) -> list[dict]:
    """Checks that need several runs: degenerate comparisons across GAN variants."""
    out = []
    gan = [r for r in results if "GAN" in str(r["model"]).upper() or "GAN" in str(r.get("run_id", "")).upper()]
    rm = {r["rmse"] for r in gan if r.get("rmse") is not None}
    if len(gan) > 1:
        if len(rm) == 1:
            out.append(_c("X-1", "GAN-based variants produce distinguishable forecasts", FAIL,
                          f"{len(gan)} GAN-based runs share RMSE={next(iter(rm)):.9g}; forecasts come from one closed-form head, so no comparison between them (or quantum effect) is testable",
                          "model-risk validation: comparison must be able to discriminate"))
        else:
            out.append(_c("X-1", "GAN-based variants produce distinguishable forecasts", PASS, "RMSE differs across variants", "model-risk validation"))
    return out
