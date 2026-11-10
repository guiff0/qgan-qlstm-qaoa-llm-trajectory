"""
Is the analysed sample the pre-registered study, or an interim part of it?

WHY THIS EXISTS (reviewer feedback, Ch.4 round 5, issue 7)
    The pre-registration plans 8 conditions x 10 seeds = 80 generator-on-path runs. Reporting 25 of them is fine as an
    INTERIM analysis, but calling a verdict "confirmatory" while 55 planned runs are missing leaves the stopping rule
    ambiguous. compute_verdicts() reads this block and downgrades every CONFIRMATORY verdict to an exploratory one
    (SUPPORTED -> SUGGESTIVE_FOR, NOT_SUPPORTED -> INCONCLUSIVE ...) until every pre-registered condition has its seeds.

Pure functions, no I/O: the same counts always give the same status.
"""
from __future__ import annotations

from src.evaluation.entanglement_design import sweep_name, LEVELS
from src.utils.run_identity import model_slug

PREREG_SEEDS = 10
PREREG_SWEEP_LEVELS = (0, 2, 4, 6)       # the DEFAULT_LEVELS of scripts/run_full_study.py
PREREG_NAMES = (["QGAN-LLM", "Classical GAN-LLM", "Classical GAN-LLM (matched)", "QGAN-LLM (no noise)"]
                + [sweep_name(*lv) for lv in LEVELS if lv[0] in PREREG_SWEEP_LEVELS])


def preregistered_slugs() -> dict:
    return {model_slug(n): n for n in PREREG_NAMES}


def study_status(n_per_config: dict, seeds: int = PREREG_SEEDS) -> dict:
    """n_per_config: {config slug: independent runs analysed}. Returns the completeness of the pre-registered design."""
    slugs = preregistered_slugs()
    per = {name: int(n_per_config.get(slug, 0)) for slug, name in slugs.items()}
    planned = len(per) * seeds
    completed = sum(min(n, seeds) for n in per.values())
    short = {name: n for name, n in per.items() if n < seeds}
    extra = {s: int(n) for s, n in n_per_config.items() if s not in slugs}
    return {"design": "pre-registered generator-on-path study", "planned_runs": planned, "completed_runs": completed,
            "seeds_per_condition": seeds, "n_per_condition": per, "conditions_short": short,
            "other_configs_not_counted": extra, "complete": not short,
            "label": ("COMPLETE: pre-registered design" if not short else
                      f"INTERIM: {completed} of {planned} planned runs; {len(short)} of {len(per)} conditions short")}


# ---------------------------------------------------------------- run budget per hypothesis
from src.evaluation.verdicts import MIN_N_CORRELATION, MIN_N_H5, MIN_N_MEDIATION, MIN_RUNS_PER_LEVEL  # noqa: E402

TWO_GROUP_N = 10        # runs per condition for a two-group test to be CONFIRMATORY (config/analysis_plan.yaml: H1, H3, H4)
_SWEEP = [model_slug(sweep_name(*lv)) for lv in LEVELS if lv[0] in PREREG_SWEEP_LEVELS]

# hypothesis -> (what is compared, kind, conditions, total runs needed (corr/mediation), per-condition floor)
_BUDGET = {
    "H1": ("QGAN-LLM vs Classical GAN-LLM (FID)", "two_group", ["qgan_llm", "classical_gan_llm"], None, TWO_GROUP_N),
    "H2": ("entanglement sweep, entropy vs ASR", "sweep", _SWEEP, MIN_N_CORRELATION, MIN_RUNS_PER_LEVEL),
    "H3": ("no-noise vs noisy poisoning degradation", "two_group", ["qgan_llm_no_noise", "qgan_llm"], None, TWO_GROUP_N),
    "H4": ("QGAN-LLM vs matched + large classical (RMSE/ASR)", "two_group",
           ["qgan_llm", "classical_gan_llm_matched", "classical_gan_llm"], None, TWO_GROUP_N),
    "H5": ("entanglement sweep, entropy vs threat accuracy", "sweep", _SWEEP, MIN_N_H5, MIN_RUNS_PER_LEVEL),
    "M2": ("mediation through entropy over the sweep", "sweep", _SWEEP, MIN_N_MEDIATION, MIN_RUNS_PER_LEVEL),
}


def hypothesis_budget(n_per_config: dict, seeds: int = PREREG_SEEDS) -> dict:
    """Runs available per hypothesis, runs still missing for the hypothesis' OWN test to be adequately sized
    ('conclusive'), and runs still missing for the whole pre-registered design (the confirmatory phase)."""
    rows = {}
    for h, (what, kind, slugs, total_min, floor) in _BUDGET.items():
        have = {s: int(n_per_config.get(s, 0)) for s in slugs}
        if kind == "two_group":
            miss = sum(max(0, floor - n) for n in have.values())
            need = f"{floor} per condition"
        else:
            miss = max(max(0, total_min - sum(have.values())), sum(max(0, floor - n) for n in have.values()))
            need = f">= {total_min} runs in total and >= {floor} per level (4 levels)"
        full = sum(max(0, seeds - n) for n in have.values())
        rows[h] = {"test": what, "runs_have": have, "total_have": sum(have.values()), "needed_to_be_conclusive": need,
                   "remaining_to_be_conclusive": int(miss), "remaining_for_full_design": int(full),
                   "caveat": ("also needs the manipulation check (designed level raises measured entropy) to pass"
                              if kind == "sweep" else
                              "also needs the poisoning manipulation to move the outcome (decidability gate)" if h == "H3" else "")}
    return rows


def format_budget(budget: dict, status: dict) -> str:
    lines = [f"RUN BUDGET PER HYPOTHESIS   [{status['label']}]",
             f"{'hyp':<4} {'runs have':>10} {'to be conclusive':>17} {'to full design':>15}   test"]
    for h, b in budget.items():
        lines.append(f"{h:<4} {b['total_have']:>10} {b['remaining_to_be_conclusive']:>17} {b['remaining_for_full_design']:>15}   "
                     f"{b['test']}  (needs {b['needed_to_be_conclusive']})")
    remaining = status["planned_runs"] - status["completed_runs"]
    lines.append(f"overall: {status['completed_runs']} of {status['planned_runs']} pre-registered runs done; "
                 f"{remaining} remaining before the confirmatory phase")
    if status["conditions_short"]:
        lines.append("  conditions short: " + ", ".join(f"{k} {v}/{status['seeds_per_condition']}"
                                                         for k, v in status["conditions_short"].items()))
    return "\n".join(lines)
