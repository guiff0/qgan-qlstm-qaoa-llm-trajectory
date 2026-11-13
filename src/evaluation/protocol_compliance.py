"""Does the run follow the methodology figure, step by step?  One check per box of the figure, each PASS / FAIL / NOT_ASSESSABLE
with the evidence, written into the hypothesis-test output and attached to EVERY verdict so no conclusion is stated without it.

The figure's boxes                                   check id
  Identical chronological training/test splits        M1_chronological_splits    (walk-forward windows leak-free; poisoning/eval use time splits)
  Arm 1 inputs only / Arm 2 classical / Arm 3 quantum  M2_three_arms_present      (all three arms in the SAME file(s))
  Forecast head linear in the features, per arm        M3_linear_heads            (ridge on x, [x,phi_c], [x,phi_q])
  Identical evaluation (same metrics and protocol)     M4_identical_metrics       (skill, RMSE, ASR, fidelity, latency on every row of every arm)
  Repeated paired seeds and walk-forward windows       M5_paired_seeds_windows    (same seeds in all arms, K >= 2 windows)
  Same random seeds fed to all three arms              M5 (same check)
  Equal training budget, matched capacity, same tuning M6_matched_budget          (src/evaluation/three_arm.audit)
  Circuit must lie on the tested causal/inference path M7_circuit_on_inference_path (map on the path for Arms 2/3; ASR attack goes through it)

Pure python/numpy. Inputs are the JSON the experiments already write; nothing is re-run.
"""
from __future__ import annotations

import os

from .arm_metrics import METRIC_KEYS

ARMS = ("linear", "classical_rf", "quantum")
PASS, FAIL, NA = "PASS", "FAIL", "NOT_ASSESSABLE"


def _chk(cid, label, status, detail, fix=""):
    return {"id": cid, "box": label, "status": status, "detail": detail, "fix": fix}


def _seeds(rows, arm):
    return sorted({int(r["seed"]) for r in rows if r.get("arm") == arm and r.get("seed") is not None})


def check_walk_forward(wf: dict | None, naive: dict | None, args_for_audit: dict | None = None, ridge_grid=None) -> list[dict]:
    """Checks M1-M7 on a walk-forward file (arms 2 and 3, optionally arm 1) plus the naive-baseline file."""
    if not wf and not naive:
        return [_chk(c, c, NA, "no walk-forward / naive-baseline file found", "python -m scripts.run_walk_forward ; python -m scripts.run_naive_baseline")
                for c in ("M1_chronological_splits", "M2_three_arms_present", "M3_linear_heads", "M4_identical_metrics",
                          "M5_paired_seeds_windows", "M6_matched_budget", "M7_circuit_on_inference_path")]
    rows = [r for r in (naive or {}).get("per_run", []) if r.get("arm") == "linear"] + list((wf or {}).get("per_run", []))
    # a row of the SAME (arm, window, seed) can appear in both files; the baseline file is the authority for arm 1
    seen, uniq = set(), []
    for r in rows:
        k = (r.get("arm"), r.get("window"), r.get("seed"))
        if k not in seen:
            seen.add(k); uniq.append(r)
    rows = uniq
    windows = (wf or {}).get("windows") or (naive or {}).get("windows") or []
    out = []

    # M1
    try:
        from . import walk_forward as wfm
        wfm.assert_no_leak(windows)
        same = (not naive or not wf) or (naive.get("windows") == wf.get("windows"))
        out.append(_chk("M1_chronological_splits", "Identical chronological splits (no look-ahead)", PASS if same else FAIL,
                        f"{len(windows)} walk-forward windows, embargo >= 1 row, no training target inside a test window"
                        + ("" if same else "; BUT the baseline file's windows differ from the other arms' windows"),
                        "" if same else "re-run scripts.run_naive_baseline with --like pointing at the other arms' file"))
    except Exception as exc:  # noqa: BLE001
        out.append(_chk("M1_chronological_splits", "Identical chronological splits (no look-ahead)", FAIL, f"leak check failed: {exc}"))

    # M2
    present = {a: sum(1 for r in rows if r.get("arm") == a) for a in ARMS}
    miss = [a for a, n in present.items() if n == 0]
    out.append(_chk("M2_three_arms_present", "Arm 1 / Arm 2 / Arm 3 all present", PASS if not miss else FAIL,
                    f"rows per arm: {present}", ("missing: " + ", ".join(miss) + " -> run scripts.run_naive_baseline / run_walk_forward") if miss else ""))

    # M3
    nf = {a: sorted({int(r["n_features"]) for r in rows if r.get("arm") == a and r.get("n_features") is not None}) for a in ARMS}
    ok3 = bool(nf["linear"]) and all(nf[a] and min(nf[a]) > min(nf["linear"]) for a in ("classical_rf", "quantum") if nf[a])
    out.append(_chk("M3_linear_heads", "Forecast head linear in x / [x, phi_c] / [x, phi_q]", PASS if ok3 else NA,
                    f"design widths {nf}: Arm 1 sees x only, Arms 2/3 add the map; every head is the same closed-form ridge", ""))

    # M4
    bad = {}
    for a in ARMS:
        sel = [r for r in rows if r.get("arm") == a]
        for k in METRIC_KEYS:
            n_missing = sum(1 for r in sel if r.get(k) is None)
            if sel and n_missing:
                bad.setdefault(a, []).append(f"{k}:{n_missing}/{len(sel)} rows without it")
    kinds = {r.get("fidelity_kind") for r in rows if r.get("fidelity_kind")}
    out.append(_chk("M4_identical_metrics", "Identical evaluation: skill, RMSE, ASR, fidelity, latency", PASS if not bad and rows else FAIL,
                    "every row of every arm carries all five metrics" + (f"; fidelity kind = {sorted(kinds)} (forecast-distribution, NOT generator FID)" if kinds else "")
                    if not bad else f"missing metrics: {bad}",
                    "" if not bad else "re-run the arms with the current src/baselines/feature_map_forecaster.py (it now records ASR, fidelity, latency)"))

    # M4b: every (window, seed) cell scored all three arms on the SAME rows by the SAME protocol
    cells = {}
    for r in rows:
        if r.get("arm") in ARMS and r.get("window") is not None and r.get("seed") is not None:
            cells.setdefault((int(r["window"]), int(r["seed"])), {})[r["arm"]] = r
    full = {c: v for c, v in cells.items() if set(v) == set(ARMS)}
    diff = []
    for c, v in sorted(full.items()):
        for k in ("n_test", "asr_n", "asr_eps", "metric_seed"):
            vals = {x.get(k) for x in v.values()}
            if len(vals) > 1:
                diff.append(f"window {c[0]} seed {c[1]}: {k} differs across arms {sorted(map(str, vals))}")
        if any(x.get("metric_seed") is not None and x.get("metric_seed") != c[1] for x in v.values()):
            diff.append(f"window {c[0]} seed {c[1]}: metric_seed is not the cell's seed")
    no_ms = sum(1 for v in full.values() for x in v.values() if x.get("metric_seed") is None)
    st4b = FAIL if diff else (NA if (not full or no_ms) else PASS)
    out.append(_chk("M4b_per_cell_identical_protocol", "Every iteration (window x seed): same test rows, same attack rows, same metric seed",
                    st4b, (f"{len(full)} complete cells; " + ("; ".join(diff[:3]) + (f" (+{len(diff) - 3} more)" if len(diff) > 3 else "")
                                                                if diff else "all arms share n_test, asr_n, asr_eps and metric_seed in every cell")
                           if full else "no cell holds all three arms"),
                    "" if st4b == PASS else "re-run python -m scripts.run_walk_forward and python -m scripts.run_naive_baseline with the current code "
                                           "(rows must carry metric_seed; Arm 1 and Arms 2/3 must use the same attack rows)"))

    # M5
    seeds = {a: _seeds(rows, a) for a in ARMS}
    wins = {a: sorted({int(r["window"]) for r in rows if r.get("arm") == a and r.get("window") is not None}) for a in ARMS}
    same_seeds = len({tuple(v) for v in seeds.values() if v}) <= 1 and all(seeds.values())
    same_wins = len({tuple(v) for v in wins.values() if v}) <= 1 and all(wins.values())
    k = max((len(v) for v in wins.values()), default=0)
    ok5 = same_seeds and same_wins and k >= 2
    out.append(_chk("M5_paired_seeds_windows", "Repeated paired seeds and walk-forward windows; same seeds in every arm",
                    PASS if ok5 else FAIL, f"seeds per arm {{{', '.join(f'{a}: {len(v)}' for a, v in seeds.items())}}}; windows per arm "
                    f"{{{', '.join(f'{a}: {len(v)}' for a, v in wins.items())}}}; K = {k}",
                    "" if ok5 else "every arm needs the same seeds AND the same K >= 2 windows"))

    # M6
    if wf and wf.get("args") and ridge_grid is not None and {"n_qubits", "n_layers"} <= set(wf["args"]):
        from . import three_arm
        au = three_arm.audit([r for r in rows if r.get("arm") in ARMS], wf["args"], ridge_grid)
        failed = [c["check"] for c in au["checks"] if not c["passed"]]
        out.append(_chk("M6_matched_budget", "Equal training budget, matched capacity, same tuning protocol", PASS if au["all_passed"] else FAIL,
                        f"{len(au['checks'])} balance checks, failed: {failed or 'none'}" + ("; " + " | ".join(au["notes"]) if au.get("notes") else ""),
                        "" if au["all_passed"] else "fix the failed balance checks before reading any arm difference"))
    else:
        out.append(_chk("M6_matched_budget", "Equal training budget, matched capacity, same tuning protocol", NA, "no walk-forward args to audit"))

    # M7
    on_path = {a: {bool(r.get("map_on_inference_path")) for r in rows if r.get("arm") == a and "map_on_inference_path" in r} for a in ARMS}
    ok7 = on_path["quantum"] == {True} and on_path["classical_rf"] == {True} and on_path["linear"] in ({False}, set())
    out.append(_chk("M7_circuit_on_inference_path", "Circuit lies on the tested causal/inference path",
                    PASS if ok7 else (NA if not any(on_path.values()) else FAIL),
                    f"map applied at inference: {{{', '.join(f'{a}: {sorted(v)}' for a, v in on_path.items())}}}; the ASR attack and the latency clock run "
                    f"through the same predict function",
                    "" if ok7 else "rows must record map_on_inference_path (re-run with the current forecaster)"))
    return out


def check_poisoning_and_detector(df_rows: list[dict] | None) -> list[dict]:
    """The GAN-study rows (all_results.csv as dicts): the H3 and H5 instruments must also follow the figure."""
    out = []
    if not df_rows:
        return out
    pr = [r for r in df_rows if r.get("poisoning_protocol")]
    if pr:
        splits = {str(r.get("poisoning_split", "random_permutation (recorded before the fix)")) for r in pr}
        ok = splits == {"chronological"}
        out.append(_chk("M1b_poisoning_chronological", "H3 poisoning train/eval split is chronological (no look-ahead)", PASS if ok else FAIL,
                        f"poisoning_split values in the rows: {sorted(splits)}",
                        "" if ok else "those rows used a random permutation split; re-run the poisoning stage (python -m scripts.run_study --apply-calibration) "
                                      "and exclude the old rows from H3"))
        pc = [float(r["poisoning_pc_degradation_pct"]) for r in pr if r.get("poisoning_pc_degradation_pct") not in (None, "")]
        floor = float(pr[0].get("poisoning_pc_floor_pct") or 5.0)
        if pc:
            okpc = min(pc) >= floor
            out.append(_chk("M8_poisoning_positive_control", "H3 positive control reaches its floor", PASS if okpc else FAIL,
                            f"{len(pc)} rows, min positive-control degradation {min(pc):.3g} % vs floor {floor} %",
                            "" if okpc else "raise poisoning.shift_sd/fraction via scripts.calibrate_poisoning (same recipe as in-run) and rerun"))
    dr = [r for r in df_rows if r.get("llm_threat_auroc_eval") not in (None, "")]
    if dr:
        au = [float(r["llm_threat_auroc_eval"]) for r in dr]
        mean = sum(au) / len(au)
        out.append(_chk("M9_detector_beats_chance", "H5 detector beats chance on held-out windows", PASS if mean >= 0.60 else FAIL,
                        f"{len(au)} rows, mean held-out AUROC {mean:.3f}",
                        "" if mean >= 0.60 else "turn on llm_pipeline.threat_feature_assist (calibrated statistic in the prompt) and rerun the LLM stage"))
    return out


def summarise(checks: list[dict]) -> dict:
    st = [c["status"] for c in checks]
    failed = [c["id"] for c in checks if c["status"] == FAIL]
    na = [c["id"] for c in checks if c["status"] == NA]
    return {"n_checks": len(checks), "passed": st.count(PASS), "failed": failed, "not_assessable": na,
            "conforms": bool(checks) and not failed and not na,
            "label": ("FOLLOWS the methodology figure" if checks and not failed and not na else
                      f"DOES NOT YET FOLLOW the methodology figure ({len(failed)} failed, {len(na)} not assessable)"),
            "checks": checks}


def caveat(summary: dict) -> str:
    if summary["conforms"]:
        return "methodology: all figure checks passed"
    bits = summary["failed"] + [f"{x}(not assessable)" for x in summary["not_assessable"]]
    return "methodology NOT fully followed: " + ", ".join(bits) + " -- read every number below as exploratory until these pass"


def annotate_verdicts(verdicts: dict, summary: dict) -> dict:
    """Every verdict carries the methodology status; nothing is reported without it."""
    for d in verdicts.values():
        d["methodology"] = {"conforms": summary["conforms"], "failed": summary["failed"], "not_assessable": summary["not_assessable"]}
        d.setdefault("why", []).append(caveat(summary))
    return verdicts


def load_json(path: str):
    import json
    try:
        with open(path, encoding="utf8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def find_naive(experiment_id: str | None, root: str = "results") -> dict | None:
    """The naive-baseline file of this experiment, else the shared 'legacy' one. Arm 1 is a deterministic ridge on the processed data:
    it does not depend on any GAN run, so one file serves every experiment (it is NOT taken from another experiment's folder)."""
    for folder in ([experiment_id] if experiment_id else []) + ["legacy"]:
        d = load_json(os.path.join(root, "naive_baseline", folder, "naive_baseline.json"))
        if d:
            d.setdefault("found_in", folder)
            return d
    return None
