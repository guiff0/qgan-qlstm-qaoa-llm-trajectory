"""Parameter counts for the classical LSTM generator, so a capacity-matched control can be chosen by rule.

The 20-qubit QGAN generator has ~912 trainable parameters; the default classical generator
(latent 32, hidden 128) has 87,072, about 95x more. A quantum-vs-classical gap measured against the big
one cannot be attributed to "quantum". `match_hidden` picks the LSTM hidden size whose parameter count
is closest to a target, for a given latent width and output width.
"""
from __future__ import annotations


def lstm_generator_params(latent: int, hidden: int, out: int) -> int:
    """nn.LSTM(latent, hidden) (weight_ih, weight_hh, bias_ih, bias_hh) + nn.Linear(hidden, out)."""
    lstm = 4 * hidden * (latent + hidden) + 8 * hidden
    fc = hidden * out + out
    return int(lstm + fc)


def match_hidden(target_params: int, latent: int, out: int, max_hidden: int = 512) -> int:
    """Hidden size whose generator parameter count is closest to `target_params` (ties -> smaller)."""
    best, best_gap = 1, None
    for h in range(1, max_hidden + 1):
        gap = abs(lstm_generator_params(latent, h, out) - target_params)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
    return best
