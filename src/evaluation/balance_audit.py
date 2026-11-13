"""Is the quantum-vs-classical comparison balanced? One audit, run before launching and stored with the experiment.

Reviewer claim 2: a quantum/classical gap is not attributable to 'quantum' unless capacity, optimisation budget,
training coverage, sample counts, loss weights and input information are matched or bracketed. The code already
sets these up (config + src/utils/capacity_match.py); this module CHECKS them from the resolved config, so a later
edit that breaks the balance stops the launch instead of silently confounding 80 runs.

Pass criteria (primary pair = 'QGAN-LLM' vs 'Classical GAN-LLM (matched)')
  1. every optimisation / data / evaluation key in SHARED_KEYS is identical;
  2. loss weights: the quantum 'alpha' equals every classical 'mse_weight';
  3. rows seen per run (epochs x max_batches_per_epoch x batch_size) are identical;
  4. generator parameters within `tol` (default 5%);
  5. input information: the classical generator gets as many input dimensions as the circuit encodes qubits;
  6. no arm carries a per-arm tuning key (hpo / tuning / search): every arm uses the same fixed hyper-parameters.
Reported, never failing (they cannot be equalised and must be disclosed): the large classical generator's
capacity ratio, each entanglement-sweep circuit's parameter count against the matched control, and compute time.
"""
from __future__ import annotations

import copy
from statistics import median

from src.evaluation.advantage_diagnostics import qgan_generator_params
from src.utils.capacity_match import lstm_generator_params

SHARED_KEYS = ["forecast_path", "forecast_target", "head_fit_rows", "forecast_eval_samples", "attack_samples",
               "forecast_chunk", "discriminator_hidden", "learning_rate", "batch_size", "epochs",
               "synthetic_ratio", "n_critic", "max_batches_per_epoch", "eval_samples"]
TUNING_HINTS = ("hpo", "tuning", "search", "optuna", "grid")
N_FEATURES = 32           # PCA features = generator output width (config data.n_pca_components)


def resolve_arms(cfg: dict) -> dict:
    """name -> fully merged config, built exactly as src/experiments/run_all.py merges ablations."""
    arms = {"QGAN-LLM": copy.deepcopy(cfg["qgan_llm"]), "Classical GAN-LLM": copy.deepcopy(cfg["classical_gan_llm"])}
    for ab in cfg.get("ablations") or []:
        arms[ab["name"]] = {**cfg["qgan_llm"], **{k: v for k, v in ab.items() if k != "name"}}
    for ab in cfg.get("classical_ablations") or []:
        arms[ab["name"]] = {**cfg["classical_gan_llm"], **{k: v for k, v in ab.items() if k != "name"}}
    return arms


def _is_quantum(name: str) -> bool:
    return name.startswith("QGAN")


def n_params(arm_cfg: dict, quantum: bool) -> int:
    if quantum:
        return qgan_generator_params(arm_cfg.get("n_qubits", 20), arm_cfg.get("n_layers", 4), N_FEATURES)
    return lstm_generator_params(arm_cfg["latent_dim"], arm_cfg["generator_hidden"], N_FEATURES)


def rows_seen(arm_cfg: dict) -> int:
    return int(arm_cfg["epochs"]) * int(arm_cfg["max_batches_per_epoch"]) * int(arm_cfg["batch_size"])


def audit(cfg: dict, tol: float = 0.05, quantum: str = "QGAN-LLM", matched: str = "Classical GAN-LLM (matched)",
          large: str = "Classical GAN-LLM") -> dict:
    arms = resolve_arms(cfg)
    checks: list[dict] = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if quantum not in arms or matched not in arms:
        check("arms_present", False, f"need {quantum!r} and {matched!r}; have {sorted(arms)}")
        return {"pass": False, "checks": checks, "param_table": []}
    q, m = arms[quantum], arms[matched]
    diff = {k: (q.get(k), m.get(k)) for k in SHARED_KEYS if q.get(k) != m.get(k)}
    check("shared_keys_identical", not diff,
          "all %d optimisation/data/evaluation keys identical" % len(SHARED_KEYS) if not diff
          else "differ (quantum, classical): " + "; ".join(f"{k}={a!r} vs {b!r}" for k, (a, b) in diff.items()))
    bad_w = {n: (c.get("mse_weight"), q.get("alpha")) for n, c in arms.items()
             if not _is_quantum(n) and c.get("mse_weight") != q.get("alpha")}
    check("loss_weight_equal", not bad_w,
          f"quantum alpha = classical mse_weight = {q.get('alpha')}" if not bad_w
          else "classical arm(s) with a different weight (mse_weight, alpha): " + str(bad_w))
    rq, rm = rows_seen(q), rows_seen(m)
    check("rows_seen_equal", rq == rm, f"{rq:,} training rows per run for both ({q['epochs']} epochs x "
          f"{q['max_batches_per_epoch']} batches x {q['batch_size']})" if rq == rm else f"{rq:,} vs {rm:,}")
    pq, pm = n_params(q, True), n_params(m, False)
    gap = abs(pq - pm) / pq
    check("params_matched", gap <= tol, f"quantum {pq:,} vs matched classical {pm:,} parameters ({gap:.1%} apart, "
          f"tolerance {tol:.0%})")
    din = m.get("generator_input_dims")
    check("input_information_equal", din == q.get("n_qubits") and m.get("latent_dim") == q.get("n_qubits"),
          f"circuit encodes {q.get('n_qubits')} features; matched generator input dims={din}, latent={m.get('latent_dim')}")
    tuned = sorted(f"{n}.{k}" for n, c in arms.items() for k in c if any(h in k.lower() for h in TUNING_HINTS))
    check("no_per_arm_tuning", not tuned, "no arm has a tuning key; hyper-parameters are shared and fixed"
          if not tuned else "tuning keys found: " + ", ".join(tuned))

    table = []
    for name, c in arms.items():
        qq = _is_quantum(name)
        try:
            p = n_params(c, qq)
        except KeyError:
            continue
        table.append({"arm": name, "kind": "quantum" if qq else "classical", "generator_params": p,
                      "ratio_to_matched_classical": round(p / pm, 3), "rows_seen_per_run": rows_seen(c)})
    return {"pass": all(c["ok"] for c in checks), "checks": checks, "param_table": table,
            "disclosed_unbalanced": [
                "wall-clock compute: a quantum run costs far more than a classical run (see compute_budget)",
                "tuning effort cannot be equalised; none of the arms was tuned (check no_per_arm_tuning)",
                f"the large classical generator has {n_params(arms[large], False) / pq:.0f}x the quantum parameters; "
                f"it is a bracket, the matched control is the primary comparator",
                "entanglement-sweep circuits share the primary's 12 qubits, but all levels except ring-d4 differ in depth "
                "(so in parameter count) from the matched control, which is sized to the primary"]}


def compute_budget(runs: list[dict]) -> dict:
    """Median wall-clock minutes per condition from an experiment's runs.jsonl (successful runs only)."""
    by: dict[str, list[float]] = {}
    for r in runs:
        if r.get("status") == "ok" and r.get("minutes") is not None and not r.get("adopted"):
            by.setdefault(r["slug"], []).append(float(r["minutes"]))
    med = {k: round(median(v), 1) for k, v in by.items()}
    out = {"median_minutes_per_condition": med, "n_runs": {k: len(v) for k, v in by.items()}}
    if "qgan_llm" in med and "classical_gan_llm_matched" in med and med["classical_gan_llm_matched"] > 0:
        out["quantum_over_matched_classical"] = round(med["qgan_llm"] / med["classical_gan_llm_matched"], 1)
    return out


def format_report(rep: dict) -> str:
    lines = ["BALANCE AUDIT (quantum vs matched classical): " + ("PASS" if rep["pass"] else "FAIL")]
    for c in rep["checks"]:
        lines.append(f"  [{'ok' if c['ok'] else 'XX'}] {c['check']:<26} {c['detail']}")
    if rep.get("param_table"):
        lines.append("  generator parameters per arm:")
        for r in rep["param_table"]:
            lines.append(f"      {r['arm']:<34}{r['kind']:<10}{r['generator_params']:>8,}   x{r['ratio_to_matched_classical']:.2f} of matched")
    for d in rep.get("disclosed_unbalanced", []):
        lines.append(f"  disclosed: {d}")
    return "\n".join(lines)
