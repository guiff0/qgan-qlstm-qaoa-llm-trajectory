"""
LLM models whose prompt carries a quantum component, so the quantum part can actually influence
a forecast that the LLM produces (the proposal's "-LLM" design). Both reuse src/llm/llm_pipeline.py:
same stratified windows, price-space attacks, dual output (forecast + threat probability).

QLSTMLLM
    A QLSTMForecaster is trained as usual (its own checkpoint/resume logic). At evaluation, its forecast
    for the real row is added to the prompt ("a quantum LSTM forecasts ..."). The LLM decides the final
    forecast and the threat probability. Control arm: the LLM Forecaster (same prompt, no quantum text).
    Expect no benefit: the QLSTM's own RMSE (0.115) is hundreds of times the persistence RMSE.

FeatureLLM (selector = "qaoa" | "classical")
    A selector picks `target_k` NAMED engineered features and the prompt lists their values for the real
    row. QAOA runs the QUBO selector of src/quantum/qaoa.py; the control picks the top-k by |correlation|
    with the next-close change. Named features are approximate: they are reconstructed from the 32 PCA
    components with the preprocessors saved by scripts/prepare_data.py (scaler.joblib, pca.joblib,
    feature_cols.json). Raw price levels (open, high, low, close) are excluded: they would be selected
    because they correlate with the price level and would add nothing the close window does not show.

Neither model has a forecast head: rmse/mae/asr in the results ARE the LLM's, on the sampled windows.
UNVERIFIED against a live endpoint, like the rest of the LLM stage.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import numpy as np

from .base import BaseForecastingModel
from ..evaluation.one_step_ahead import shift_for_one_step_ahead


class _ContextLLMBase(BaseForecastingModel):
    context_kind = "none"

    def build(self):
        pass

    def predict(self, X):
        raise NotImplementedError(f"{self.name} produces forecasts only through evaluate() (one LLM call per window).")

    def make_context_fn(self, X_rows: np.ndarray) -> Callable[[int], str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def evaluate(self, X_test, y_test, attack_cfg: Dict = None, last_input_prices=None, **kwargs):
        from ..llm.llm_pipeline import evaluate_llm_pipeline

        if not os.environ.get("NVIDIA_API_KEY") and self.config.get("chat_fn") is None:
            self.results = {"error": f"{self.name} needs NVIDIA_API_KEY"}
            return self.results
        if last_input_prices is None:
            self.results = {"error": f"{self.name} needs last_input_prices (the real close series)"}
            return self.results
        X_test, y_test = shift_for_one_step_ahead(np.asarray(X_test), np.asarray(y_test))
        closes = np.asarray(last_input_prices, dtype=float).flatten()[:-1]
        ctx = self.make_context_fn(X_test)
        cfg = {**self.config, "use_synthetic_context": False}
        self.results = evaluate_llm_pipeline(closes, np.asarray(y_test, dtype=float).flatten(), X_test, cfg,
                                             seed=self.seed, generator_fn=None, mapper=None,
                                             chat_fn=self.config.get("chat_fn"), context_fn=ctx,
                                             cache_dir=self.config.get("cache_dir", "results/llm_cache"))
        self.results.update(rmse=self.results.get("llm_rmse", float("nan")), mae=self.results.get("llm_mae", float("nan")),
                            asr=self.results.get("llm_asr", float("nan")), n_llm_samples=self.results.get("llm_n_forecasts", 0),
                            n_llm_errors=self.results.get("llm_n_errors", 0), model_id=self.config["model_id"],
                            context_kind=self.context_kind)
        self.results.update(self.extra_results())
        return self.results

    def extra_results(self) -> Dict:
        return {}


class QLSTMLLM(_ContextLLMBase):
    context_kind = "qlstm_forecast"

    def __init__(self, config: Dict = None, seed: int = 42, inner=None):
        super().__init__("QLSTM-LLM", dict(config or {}))
        self.seed = seed
        if inner is None:
            from .qlstm_forecaster import QLSTMForecaster
            inner = QLSTMForecaster(self.config["qlstm"], seed=seed)
        self.inner = inner

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        self.inner.train(X_train, y_train, X_val, y_val, run_logger=run_logger)
        self.resumed_from_checkpoint = bool(getattr(self.inner, "resumed_from_checkpoint", False))
        self.is_trained = True

    def make_context_fn(self, X_rows):
        def ctx(idx: int) -> str:
            f = float(np.asarray(self.inner.predict(X_rows[idx:idx + 1])).flatten()[0])
            return (f"A quantum LSTM model trained on the engineered market features forecasts the next close "
                    f"as {f:.5f}.")
        return ctx


class FeatureLLM(_ContextLLMBase):
    context_kind = "selected_features"

    def __init__(self, config: Dict = None, seed: int = 42, mapper=None, selector_cls=None):
        cfg = {"selector": "qaoa", "n_candidates": 16, "target_k": 8, "max_selection_rows": 200000,
               "exclude_features": ["open", "high", "low", "close"], "processed_dir": "data/processed", **(config or {})}
        name = "QAOA-LLM" if cfg["selector"] == "qaoa" else "Classical-Features-LLM"
        super().__init__(name, cfg)
        self.seed = seed
        self._mapper = mapper
        self._selector_cls = selector_cls
        self.selected_names: list = []
        self.selected_idx: Optional[np.ndarray] = None

    def _get_mapper(self):
        if self._mapper is None:
            from ..llm.llm_pipeline import SyntheticPriceMapper
            self._mapper = SyntheticPriceMapper(self.config["processed_dir"])
        if not getattr(self._mapper, "available", False):
            raise RuntimeError(
                f"{self.name} needs the preprocessors saved by scripts/prepare_data.py in "
                f"{self.config['processed_dir']} (scaler.joblib, pca.joblib, feature_cols.json); re-run prepare_data. "
                f"Reason: {getattr(self._mapper, 'reason', 'unknown')}")
        return self._mapper

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        mapper = self._get_mapper()
        X = np.asarray(X_train); y = np.asarray(y_train, dtype=float).flatten()
        n = min(len(X) - 1, int(self.config["max_selection_rows"]))
        rows = np.linspace(0, len(X) - 2, n).astype(int)          # evenly spaced, needs row k+1 for the target
        target = y[rows + 1] - y[rows]                              # next-close CHANGE, not the price level
        raw = mapper.to_raw(X[rows])
        names = list(mapper.feature_cols)
        keep = [i for i, nm in enumerate(names) if nm not in set(self.config["exclude_features"])]
        raw_k = raw[:, keep]
        if self.config["selector"] == "qaoa":
            if self._selector_cls is None:
                from ..quantum.qaoa import QAOAFeatureSelector as sel_cls, QAOAFeatureSelectorConfig as cfg_cls
            else:
                sel_cls, cfg_cls = self._selector_cls
            qc = self.config
            sel_cfg = cfg_cls(n_candidates=qc["n_candidates"], target_k=qc["target_k"], n_layers=qc.get("qaoa_layers", 3),
                              budget_penalty=qc.get("budget_penalty", 2.0), redundancy_penalty=qc.get("redundancy_penalty", 1.0),
                              redundancy_edge_threshold=qc.get("redundancy_edge_threshold", 0.3),
                              quantum_device=qc.get("quantum_device", "lightning.qubit"), n_steps=qc.get("qaoa_steps", 150),
                              learning_rate=qc.get("qaoa_learning_rate", 0.1), seed=self.seed)
            mask = np.asarray(sel_cls(sel_cfg).fit(raw_k, target).selected_mask, dtype=bool)
            chosen = np.where(mask)[0]
        else:
            c = np.array([abs(np.corrcoef(raw_k[:, j], target)[0, 1]) if raw_k[:, j].std() > 0 else 0.0
                          for j in range(raw_k.shape[1])])
            chosen = np.argsort(-np.nan_to_num(c))[: int(self.config["target_k"])]
        if len(chosen) == 0:
            raise RuntimeError(f"{self.name}: the selector returned no features")
        self.selected_idx = np.array([keep[j] for j in chosen])
        self.selected_names = [names[i] for i in self.selected_idx]
        if run_logger:
            run_logger.info(f"{self.name} selected features: {self.selected_names}")
        self.is_trained = True

    def make_context_fn(self, X_rows):
        mapper = self._get_mapper()

        def ctx(idx: int) -> str:
            vals = mapper.to_raw(X_rows[idx:idx + 1])[0, self.selected_idx]
            items = ", ".join(f"{n}={v:.4g}" for n, v in zip(self.selected_names, vals))
            return f"Selected market indicators at the last bar (approximate): {items}."
        return ctx

    def extra_results(self) -> Dict:
        return {"selector": self.config["selector"], "selected_features": ",".join(self.selected_names)}
