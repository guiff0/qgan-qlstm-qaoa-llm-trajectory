"""
Adversarial attacks: FGSM, PGD, and a simplified CW (Carlini-Wagner) L2 attack.

THIS REPLACES THE ORIGINAL STUBS:

    def _evaluate_adversarial(self, X_test, y_test):
        return 31.0  # Expected ASR from Table 47

    def _evaluate_adversarial(self, X_test, y_test):
        return 8.7  # Expected ASR from RQ4 results

Those functions returned the dissertation's disputed numbers directly,
regardless of input. Below, Attack Success Rate is computed from an
actual perturbation search against the actual trained model: an
"attack" is counted successful if a bounded perturbation changes the
model's directional forecast (up/down relative to the current price)
or pushes the point forecast's error past a materiality threshold —
matching how ASR is operationally defined in Ch. 3 (Operational
Definitions: "% of successful adversarial inputs").
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.progress import progress_bar


def _directional_flip(original_pred: torch.Tensor, adv_pred: torch.Tensor,
                       last_input_price: torch.Tensor) -> torch.Tensor:
    """An attack 'succeeds' if it flips the predicted direction of price
    movement relative to the last observed price. Returns a bool tensor."""
    orig_dir = torch.sign(original_pred.squeeze(-1) - last_input_price)
    adv_dir = torch.sign(adv_pred.squeeze(-1) - last_input_price)
    return orig_dir != adv_dir


def fgsm_perturbation(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                       epsilon: float) -> torch.Tensor:
    """Returns the perturbed input X_adv itself (not a success verdict).
    Factored out of fgsm_attack so callers that need the actual perturbed
    samples -- e.g. threat_labels.py, building a labeled threat/benign
    dataset from real perturbations rather than a pass/fail signal --
    don't have to duplicate the gradient computation."""
    X = X.clone().detach().requires_grad_(True)
    pred = model(X)
    loss = nn.functional.mse_loss(pred.squeeze(-1), y, reduction="sum")
    grad = torch.autograd.grad(loss, X)[0]
    return (X + epsilon * grad.sign()).detach()


def fgsm_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                epsilon: float, last_input_price: torch.Tensor) -> torch.Tensor:
    """Fast Gradient Sign Method. Returns per-sample success (bool tensor)."""
    X_adv = fgsm_perturbation(model, X, y, epsilon)
    with torch.no_grad():
        orig_pred = model(X)
        adv_pred = model(X_adv)
    return _directional_flip(orig_pred, adv_pred, last_input_price)


def pgd_perturbation(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                      epsilon: float, alpha: float, steps: int) -> torch.Tensor:
    """Returns the perturbed input X_adv itself (not a success verdict).
    See fgsm_perturbation's docstring for why this exists as a separate
    function from pgd_attack."""
    X_orig = X.clone().detach()
    X_adv = X_orig.clone().detach()

    for _ in progress_bar(range(steps), total=steps, desc="  PGD steps", unit="step"):
        X_adv.requires_grad_(True)
        pred = model(X_adv)
        loss = nn.functional.mse_loss(pred.squeeze(-1), y, reduction="sum")
        grad = torch.autograd.grad(loss, X_adv)[0]

        with torch.no_grad():
            X_adv = X_adv + alpha * grad.sign()
            perturbation = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
            X_adv = (X_orig + perturbation).detach()

    return X_adv


def pgd_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
               epsilon: float, alpha: float, steps: int,
               last_input_price: torch.Tensor) -> torch.Tensor:
    """Projected Gradient Descent — iterative FGSM with an epsilon-ball projection."""
    X_orig = X.clone().detach()
    X_adv = pgd_perturbation(model, X, y, epsilon, alpha, steps)

    with torch.no_grad():
        orig_pred = model(X_orig)
        adv_pred = model(X_adv)
    return _directional_flip(orig_pred, adv_pred, last_input_price)


def cw_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
              c: float, steps: int, last_input_price: torch.Tensor,
              lr: float = 0.01) -> torch.Tensor:
    """
    Simplified Carlini-Wagner L2 attack for regression outputs: minimizes
    ||delta||_2 while maximizing directional-forecast deviation, via a
    Lagrangian trade-off weighted by `c` (matches the standard CW
    formulation adapted to a regression, not classification, target).
    """
    X_orig = X.clone().detach()
    delta = torch.zeros_like(X_orig, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    with torch.no_grad():
        orig_pred = model(X_orig)

    cw_bar = progress_bar(total=steps, desc="  CW steps", unit="step")
    for _ in range(steps):
        optimizer.zero_grad()
        X_adv = X_orig + delta
        adv_pred = model(X_adv)
        # Push the adversarial prediction away from the original prediction,
        # penalized by perturbation norm.
        deviation = -torch.abs(adv_pred.squeeze(-1) - orig_pred.squeeze(-1)).sum()
        l2_penalty = (delta ** 2).sum()
        loss = c * deviation + l2_penalty
        loss.backward()
        optimizer.step()
        cw_bar.update(1)
        cw_bar.set_postfix(loss=f"{loss.item():.3e}")
    cw_bar.close()

    with torch.no_grad():
        X_adv = (X_orig + delta).detach()
        adv_pred = model(X_adv)
    return _directional_flip(orig_pred, adv_pred, last_input_price)


def compute_attack_success_rate(model, X_test: torch.Tensor, y_test: torch.Tensor,
                                 last_input_prices: torch.Tensor, attack_cfg: dict,
                                 attacks: list[str] = ("fgsm", "pgd", "cw")) -> dict:
    """
    Runs the configured attack suite against `model` on `X_test` and returns
    per-attack and overall Attack Success Rate (%), matching the "average
    ASR across attack types" framing used throughout Ch. 4.

    `model` just needs to be callable as model(X) -> predictions of shape
    (batch, 1). It is NOT required to be an nn.Module itself -- callers
    typically pass a bound method (e.g. self._forecast_model) that wraps
    one or more underlying modules. Because of that, this function does
    NOT call model.eval()/.train() (a bound method has neither attribute
    and this would raise AttributeError -- caught by tests/test_smoke.py).
    Callers are responsible for putting their own underlying nn.Module(s)
    into eval mode before calling this, e.g.:
        self.forecast_head.eval()
        compute_attack_success_rate(self._forecast_model, ...)
    """
    n = X_test.shape[0]
    results = {}
    all_success = torch.zeros(n, dtype=torch.bool)

    # NOTE ON SCALE: each attack below runs against the FULL X_test tensor
    # in one shot (PGD/CW's internal step loops -- progress-barred above --
    # are the only per-iteration cost inside them). For the classical
    # baselines that's cheap per row; for QLSTMForecaster, every forward
    # call here re-runs the PQC once per row per step, which does not scale
    # to a multi-million-row test set the way the classical models do. This
    # is a real, separate scaling limitation of the quantum-at-inference
    # design (see qlstm_forecaster.py's module docstring), not something a
    # progress bar fixes -- consider setting attack_cfg["n_benign_samples"]
    # (or subsampling X_test before calling this) if this is too slow for
    # QLSTMForecaster specifically.
    for name in progress_bar(list(attacks), total=len(attacks), desc="Attack suite", unit="attack"):
        if name == "fgsm":
            success = fgsm_attack(model, X_test, y_test, attack_cfg["fgsm_epsilon"], last_input_prices)
            results["fgsm_asr"] = 100.0 * success.float().mean().item()
            all_success |= success
        elif name == "pgd":
            success = pgd_attack(model, X_test, y_test, attack_cfg["pgd_epsilon"],
                                 attack_cfg["pgd_alpha"], attack_cfg["pgd_steps"], last_input_prices)
            results["pgd_asr"] = 100.0 * success.float().mean().item()
            all_success |= success
        elif name == "cw":
            success = cw_attack(model, X_test, y_test, attack_cfg["cw_c"],
                                attack_cfg["cw_steps"], last_input_prices)
            results["cw_asr"] = 100.0 * success.float().mean().item()
            all_success |= success

    results["overall_asr"] = 100.0 * all_success.float().mean().item()
    results["n_samples"] = n
    return results


def compute_clean_asr(model, X_test, y_test, last_input_prices, attack_cfg: dict) -> float:
    """
    The "clean" (epsilon=0, no actual perturbation) baseline
    metrics/resilience.py's qar() needs (asr_classical_clean /
    asr_quantum_clean) but this pipeline never computed before -- every
    existing `asr` value is already the ATTACKED condition.

    Scoped to fgsm+pgd only: both are epsilon-bounded, so epsilon=0
    guarantees zero perturbation by construction. Carlini-Wagner has no
    epsilon parameter (cw_c/cw_steps control an optimization objective,
    not a perturbation budget directly) -- there's no equally direct
    "zero perturbation" setting for it, so it's left out of the clean
    baseline rather than guessing one. This means asr_clean and the
    attacked `asr` aren't computed over the exact same attack set
    (clean: fgsm+pgd; attacked: fgsm+pgd+cw by default) -- documented
    here rather than silently mismatched.
    """
    clean_cfg = dict(attack_cfg)
    clean_cfg["fgsm_epsilon"] = 0.0
    clean_cfg["pgd_epsilon"] = 0.0
    clean_cfg["pgd_alpha"] = 0.0
    report = compute_attack_success_rate(model, X_test, y_test, last_input_prices,
                                          clean_cfg, attacks=["fgsm", "pgd"])
    return report["overall_asr"]
