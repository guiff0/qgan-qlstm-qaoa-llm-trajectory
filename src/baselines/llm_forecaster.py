"""
LLM Forecaster: the first baseline in this pipeline whose forecast
actually comes from an LLM, rather than a classical nn.Linear head
dressed up with an LLM-sounding name. Every other "LLM" model
(Classical GAN-LLM, QGAN-LLM) names a `forecast_head` that is a plain
linear layer trained on real features -- src/llm/nvidia_finetune.py
existed as a real, working fine-tuning client, but nothing ever called
it or used an LLM's output as a prediction.

SCOPE, read before using this for real:
  - ZERO-SHOT / FEW-SHOT, not fine-tuned, by default. Fine-tuning is a
    separate, hours-long, asynchronous job
    (nvidia_finetune.submit_finetune_job + poll_finetune_status) that
    doesn't fit this codebase's synchronous train()/evaluate() contract
    -- train() here does NOT submit or wait for a fine-tuning job. Run
    that manually, get a fine-tuned model_id, then pass it as
    config["model_id"] to use it here, instead of a base model.
  - Input mismatch with every other baseline: this model needs actual
    recent CLOSE PRICES to build a sensible text prompt, but
    evaluate()'s X_test argument is PCA-transformed features, not
    interpretable prices. Reuses the SAME convention
    src/experiments/run_all.py already applies for last_input_prices
    (X[:, 0] as a stand-in for "the last observed close") -- an
    existing, documented proxy in this codebase, not a new assumption
    invented here, but still a proxy, not real price history.
  - Each prediction is a REAL network call (see src/llm/llm_inference.py)
    -- evaluating on the full ~750k-row test set would mean ~750k live
    API calls. This model only evaluates on a small subsample
    (config["max_llm_calls"], default 20) and reports metrics computed
    over just that subsample, with n_llm_samples recorded explicitly so
    it is never confused with the full-test-set metrics every other
    baseline in this pipeline reports.
  - UNVERIFIED AGAINST A LIVE ENDPOINT: see llm_inference.py's module
    docstring -- this sandbox has no NVIDIA_API_KEY and no network
    route to NVIDIA's API at all. Verify against a real account before
    trusting any number this produces.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseForecastingModel
from ..evaluation.metrics import mae as mae_fn
from ..evaluation.metrics import rmse as rmse_fn
from ..evaluation.one_step_ahead import shift_for_one_step_ahead
from ..llm.llm_inference import query_llm_forecast


class LLMForecaster(BaseForecastingModel):
    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            "model_id": "meta/llama-3.3-70b-instruct",  # base model; see module docstring re: fine-tuning
            "api_base": "https://api.nvidia.com/v1",
            "window_size": 10,   # how many recent closes go into the prompt
            "horizon_minutes": 1,
            "max_llm_calls": 20,  # real network calls cost real time/money -- keep this small by default
            "query_fn": None,     # dependency injection point for tests -- see evaluate()
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("LLM Forecaster", cfg)
        self.seed = seed

    def build(self):
        pass  # nothing to build locally -- inference happens against a hosted model

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        # Deliberately a no-op -- see module docstring. Fine-tuning (if
        # you want it) is run manually via nvidia_finetune.py, separately
        # from this pipeline's synchronous train() contract.
        self.is_trained = True

    def predict(self, X):
        raise NotImplementedError(
            "LLMForecaster.predict(X) isn't implemented standalone -- unlike every other "
            "baseline, it needs a raw CLOSE-PRICE window to build a prompt, not just a "
            "feature matrix. Use evaluate(), which builds that window itself from X[:, 0] "
            "(see module docstring)."
        )

    def evaluate(self, X_test, y_test, attack_cfg: Dict = None, last_input_prices=None, **kwargs):
        X_test, y_test = shift_for_one_step_ahead(np.asarray(X_test), np.asarray(y_test))
        query_fn = self.config["query_fn"] or query_llm_forecast

        window = self.config["window_size"]
        n_calls = min(self.config["max_llm_calls"], len(X_test) - window)
        if n_calls <= 0:
            self.results = {"error": f"not enough rows ({len(X_test)}) for window_size={window}"}
            return self.results

        rng = np.random.default_rng(self.seed)
        start_indices = rng.choice(range(window, len(X_test)), size=n_calls, replace=False)

        predictions, actuals, errors = [], [], []
        for idx in start_indices:
            recent_closes = X_test[idx - window:idx, 0].tolist()  # see module docstring re: this proxy
            try:
                pred = query_fn(recent_closes, self.config["model_id"],
                                 horizon_minutes=self.config["horizon_minutes"],
                                 api_base=self.config["api_base"])
                predictions.append(pred)
                actuals.append(float(y_test[idx]))
            except Exception as exc:
                errors.append(str(exc))

        self.results = {
            "n_llm_samples": len(predictions),
            "n_llm_errors": len(errors),
            "llm_errors_sample": errors[:5],
            "model_id": self.config["model_id"],
        }
        if predictions:
            preds_arr = np.array(predictions)
            actuals_arr = np.array(actuals)
            self.results["rmse"] = rmse_fn(actuals_arr, preds_arr)
            self.results["mae"] = mae_fn(actuals_arr, preds_arr)
        else:
            self.results["rmse"] = float("nan")
            self.results["mae"] = float("nan")
        return self.results
