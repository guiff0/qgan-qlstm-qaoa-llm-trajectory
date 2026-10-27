"""
LLM pipeline: the LLM produces BOTH the price forecast and the threat
probability, as the dissertation's design requires.

What this module does (and what the old code did not):
  * One chat call per window returns two numbers, "Forecast: <price>" and
    "Threat probability: <0..1>". The forecast replaces the linear
    forecast_head as the reported forecast on the evaluation sample; the
    threat probability is parsed by src/llm/threat_scoring.py (which no code
    previously called) and used for FPR / accuracy (DV3, DV5).
  * Prompts use REAL close prices (the unshifted y series), not X[:, 0]. The
    old LLMForecaster used PCA component 0 as a "close", which is not a price.
  * The generator reaches the LLM: when synthetic context is available, the
    prompt includes the closes of the generator's nearest synthetic rows
    (see SyntheticPriceMapper). Without it, the LLM sees real history only,
    which is the control condition (the plain "LLM Forecaster" baseline).
  * Adversarial inputs for the LLM are crafted in PRICE space against a
    surrogate (ridge AR model on the closes), then sent to the LLM. A
    directional flip of the LLM forecast relative to the last real close is an
    attack success, the same definition src/attacks/adversarial.py uses.
    Epsilons are new design parameters (llm_pipeline.price_attack_k etc.),
    not the feature-space values of the proposal.

Scope and cost: every window costs 2 live calls (a clean query and an attacked
query), so this runs on a stratified SAMPLE of the test set (default 102
windows), never on all ~745,000 rows. Responses are cached on disk by
prompt hash so a re-run does not pay twice.

NOT VERIFIED AGAINST A LIVE ENDPOINT from the development sandbox (no key, no
route). Unit tests use injected fakes. Run `python scripts/check_llm_endpoint.py`
once with a real key before a full run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import requests

from .nvidia_finetune import NvidiaApiKeyMissingError
from .threat_scoring import parse_threat_probability

DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1"
QACL_THRESHOLD = 0.70

SYSTEM_PROMPT = (
    "You are a EUR/USD market analyst and a market-data security monitor. "
    "Answer with exactly two lines and nothing else:\n"
    "Forecast: <predicted next 1-minute close, 5 decimals>\n"
    "Threat probability: <number between 0 and 1>\n"
    "The threat probability is the probability that the price window was "
    "adversarially manipulated (small crafted perturbations), not that the market is risky."
)


class LLMModelUnavailableError(RuntimeError):
    """The hosted model is gone (HTTP 404/410) or not in the account's model list."""


class LLMAuthError(RuntimeError):
    """The endpoint rejected the API key (HTTP 401/403)."""


def get_api_key() -> str:
    """NVIDIA_API_KEY with the usual copy/paste damage removed: surrounding whitespace and quotes
    (PowerShell and cmd keep quotes that are typed inside `set X="..."`) and a pasted 'Bearer ' prefix."""
    key = os.environ.get("NVIDIA_API_KEY", "").strip().strip("\"'").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if not key:
        raise NvidiaApiKeyMissingError("NVIDIA_API_KEY is not set (or is empty after removing quotes and spaces).")
    return key


def describe_key() -> str:
    """Shape of the key for diagnostics. Never prints the secret itself."""
    raw = os.environ.get("NVIDIA_API_KEY", "")
    key = get_api_key()
    notes = [f"length {len(key)}"]
    notes.append("starts with 'nvapi-'" if key.startswith("nvapi-") else "does NOT start with 'nvapi-' (keys from build.nvidia.com do)")
    if raw != key:
        notes.append("the environment value had quotes, spaces or a 'Bearer ' prefix (removed before use)")
    if any(c.isspace() for c in key):
        notes.append("contains whitespace inside the key")
    return "; ".join(notes)


# ----------------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------------
def list_models(api_base: str = DEFAULT_API_BASE, timeout: int = 20, get: Callable = requests.get) -> List[str]:
    """Model ids the endpoint currently serves (GET {api_base}/models)."""
    r = get(f"{api_base}/models", headers={"Authorization": f"Bearer {get_api_key()}"}, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"GET /models returned {r.status_code}: {r.text[:300]}")
    return [m["id"] for m in r.json().get("data", []) if "id" in m]


def verify_auth(model_id: str, api_base: str = DEFAULT_API_BASE, **kw) -> None:
    """One 1-token authenticated call. Run it before any expensive work: /models is public, so
    resolve_model_id() alone passes with an invalid key."""
    chat([{"role": "user", "content": "ping"}], model_id, api_base, max_tokens=1, retries=1, **kw)


def probe_model(model_id: str, api_base: str = DEFAULT_API_BASE, timeout: int = 45, **kw) -> tuple:
    """(ok, detail). /models lists the whole catalog, including models this account cannot call
    (HTTP 404 "Function ... Not found for account"), so only a real 1-token call proves a model is usable.
    A 429 counts as usable (it exists, it is just rate-limited). A rejected key raises LLMAuthError."""
    try:
        chat([{"role": "user", "content": "ping"}], model_id, api_base, max_tokens=1, retries=1, timeout=timeout,
             backoff=kw.pop("backoff", 2.0), **kw)
        return True, "ok"
    except LLMAuthError:
        raise
    except LLMModelUnavailableError as exc:
        return False, str(exc)[:200]
    except RuntimeError as exc:
        if "HTTP 429" in str(exc):
            return True, "rate limited (exists)"
        if "network error" in str(exc):
            return False, f"no answer ({str(exc)[:120]}); cold start or overloaded, retry later"
        return False, str(exc)[:200]


_NOT_CHAT = ("embed", "guard", "safety", "retriev", "rerank", "reward", "clip", "parse", "topic-control", "content-safety")


def discover_models(api_base: str = DEFAULT_API_BASE, list_fn: Callable = None, probe_fn: Callable = None,
                    needle: str = "llama", limit: int = 40, max_workers: int = 4,
                    on_result: Optional[Callable] = None) -> list:
    """Chat-capable models whose name contains `needle` that this account can actually call. Probed in parallel;
    `on_result(model, (ok, detail))` is called as each finishes so a long run shows progress. A failure on one
    model never stops the others; a rejected key (LLMAuthError) does."""
    list_fn = list_fn or list_models
    probe_fn = probe_fn or (lambda m: probe_model(m, api_base))
    cands = sorted(m for m in list_fn(api_base) if needle in m.lower() and not any(x in m.lower() for x in _NOT_CHAT))[:limit]

    def one(m):
        try:
            res = probe_fn(m)
        except LLMAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            res = (False, f"error: {type(exc).__name__}: {str(exc)[:120]}")
        if on_result:
            on_result(m, res)
        return m, res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, cands))


def resolve_model_id(preferred: str, fallbacks: Sequence[str] = (), api_base: str = DEFAULT_API_BASE,
                     list_fn: Callable = list_models, probe_fn: Optional[Callable] = None) -> str:
    """First of [preferred, *fallbacks] that is listed AND (when probe_fn is given) callable by this account.
    Fails BEFORE any expensive work. History: meta/llama-3.3-70b-instruct reached end of life on 2026-08-26
    and returned 410; nvidia/llama-3.1-nemotron-70b-instruct is listed but returned 404 for this account."""
    available = list_fn(api_base)
    tried = []
    for mid in [preferred, *fallbacks]:
        if mid not in available:
            tried.append(f"{mid}: not listed")
            continue
        if probe_fn is None:
            return mid
        ok, detail = probe_fn(mid)
        if ok:
            return mid
        tried.append(f"{mid}: listed but not callable ({detail})")
    llama_like = sorted(m for m in available if "llama" in m.lower())
    raise LLMModelUnavailableError(
        f"No usable model among {[preferred, *fallbacks]} at {api_base}:\n  " + "\n  ".join(tried) +
        f"\nLlama-family ids in the catalog: {llama_like[:20]}\n"
        f"Run `python scripts/check_llm_endpoint.py --probe-all` to find which ones THIS account can call, then set "
        f"llm_pipeline.model_id (and record the change in the manuscript; the proposal names Llama 3.3 70B)."
    )


def chat(messages: List[dict], model_id: str, api_base: str = DEFAULT_API_BASE, timeout: int = 60,
         max_tokens: int = 64, temperature: float = 0.0, retries: int = 3, backoff: float = 2.0,
         post: Callable = requests.post, sleep: Callable = time.sleep) -> str:
    """One chat completion. Retries 429 and 5xx with increasing waits; 404/410 raise
    LLMModelUnavailableError; any other 4xx raises RuntimeError. Never fabricates text."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    last = ""
    for attempt in range(retries + 1):
        try:
            r = post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            # A first call to a large hosted model can take longer than the timeout (cold start); retry.
            last = f"{type(exc).__name__} after {timeout}s"
            if attempt < retries:
                sleep(backoff * (attempt + 1))
                continue
            raise RuntimeError(f"network error after {retries + 1} attempt(s): {last}") from exc
        if r.status_code in (401, 403):
            raise LLMAuthError(
                f"HTTP {r.status_code}: the key was rejected ({r.text[:200]}). Key shape: {describe_key()}. "
                f"Create a key at build.nvidia.com (it starts with 'nvapi-') and set it WITHOUT quotes inside the value: "
                f"PowerShell  $env:NVIDIA_API_KEY = 'nvapi-...'   (the quotes are PowerShell's, not part of the key). "
                f"Note that GET /models needs no key, so a model listing succeeding does not prove the key works.")
        if r.status_code in (404, 410):
            raise LLMModelUnavailableError(f"HTTP {r.status_code} for model {model_id!r}: {r.text[:300]}")
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if attempt < retries:
                sleep(backoff * (attempt + 1))
                continue
            raise RuntimeError(f"giving up after {retries + 1} attempts ({last})")
        if r.status_code >= 400:
            raise RuntimeError(f"NVIDIA API returned {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Unexpected response shape: {r.text[:300]!r}") from exc
    raise RuntimeError(last)


# ----------------------------------------------------------------------------
# Prompt / parsing
# ----------------------------------------------------------------------------
def reasoning_prefix(model_id: str) -> str:
    """System-prompt switch that turns a reasoning model's thinking off. Plain instruct models get none."""
    m = model_id.lower()
    if "nemotron-super" in m or "nemotron-nano" in m:
        return "/no_think\n"
    if "nemotron-ultra" in m:
        return "detailed thinking off\n"
    return ""


def build_dual_messages(closes: Sequence[float], synthetic_closes: Optional[Sequence[float]] = None,
                        horizon_minutes: int = 1, prefix: str = "", extra_context: Optional[str] = None) -> List[dict]:
    hist = ", ".join(f"{c:.5f}" for c in closes)
    user = f"The last {len(closes)} 1-minute EUR/USD closes, oldest first: {hist}.\n"
    if synthetic_closes is not None and len(synthetic_closes):
        syn = ", ".join(f"{c:.5f}" for c in synthetic_closes)
        user += (f"A generative model of this market produced {len(synthetic_closes)} nearby synthetic "
                 f"market states with closes: {syn}.\n")
    if extra_context:
        user += extra_context.strip() + "\n"
    user += f"Predict the close {horizon_minutes} minute(s) ahead and the threat probability."
    sys_prompt = prefix + SYSTEM_PROMPT
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


_FORECAST_RE = re.compile(r"forecast[^0-9+\-.]*([-+]?\d*\.?\d+)", re.IGNORECASE)


def parse_dual_response(text: str) -> dict:
    """Returns {'forecast': float|None, 'threat_probability': float|None}. A missing value is
    None, never a default. <think>...</think> blocks are removed first."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    m = _FORECAST_RE.search(text)
    forecast = float(m.group(1)) if m else None
    try:
        threat = parse_threat_probability(text)
    except ValueError:
        threat = None
    return {"forecast": forecast, "threat_probability": threat}


# ----------------------------------------------------------------------------
# Synthetic context: generator -> price space
# ----------------------------------------------------------------------------
class SyntheticPriceMapper:
    """Maps generator rows (PCA space) to close prices with the preprocessors that
    scripts/prepare_data.py now saves (scaler.joblib, pca.joblib, feature_cols.json).
    Absent files => available == False and the pipeline reports synthetic_context_used=False."""

    def __init__(self, processed_dir: str):
        self.available = False
        try:
            import joblib
            self.scaler = joblib.load(os.path.join(processed_dir, "scaler.joblib"))
            self.pca = joblib.load(os.path.join(processed_dir, "pca.joblib"))
            with open(os.path.join(processed_dir, "feature_cols.json"), encoding="utf8") as fh:
                cols = json.load(fh)
            self.feature_cols = cols
            self.close_idx = cols.index("close")
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"{type(exc).__name__}: {exc}"

    def to_raw(self, pca_rows: np.ndarray) -> np.ndarray:
        """Approximate named engineered features (columns = feature_cols) reconstructed from PCA rows.
        32 of ~54 components are kept, so these are approximations, not the original values."""
        return self.scaler.inverse_transform(self.pca.inverse_transform(np.asarray(pca_rows, dtype=float)))

    def to_close(self, pca_rows: np.ndarray) -> np.ndarray:
        return self.to_raw(pca_rows)[:, self.close_idx]


def nearest_synthetic_closes(x_row: np.ndarray, pool_rows: np.ndarray, pool_closes: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(pool_rows - x_row[None, :], axis=1)
    return pool_closes[np.argsort(d)[:k]]


# ----------------------------------------------------------------------------
# Price-space adversarial inputs (surrogate transfer attack)
# ----------------------------------------------------------------------------
def fit_ar_surrogate(closes: np.ndarray, window: int, ridge: float = 1e-8, max_rows: int = 50000):
    """Ridge AR(window) forecaster of the next close, fitted on mean-centred windows. Returns (coef, bias)."""
    c = np.asarray(closes, dtype=float)[:max_rows + window + 1]
    idx = np.arange(window, len(c) - 1)
    W = np.stack([c[i - window + 1:i + 1] for i in idx]); y = c[idx + 1]
    mu = W.mean(axis=1, keepdims=True)
    A = W - mu; t = y - mu[:, 0]
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(window), A.T @ t)
    return coef, 0.0


def attack_window(window_closes: np.ndarray, last_close: float, coef: np.ndarray, kind: str, eps: float,
                  alpha: float, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Push the surrogate forecast across the last close (a directional flip). The surrogate is linear,
    so FGSM and PGD share the gradient sign; PGD adds a random start, `steps` projected steps of size alpha."""
    w = np.asarray(window_closes, dtype=float)
    f = float(coef @ (w - w.mean()) + w.mean())
    direction = np.sign(f - last_close) or 1.0
    g = -direction * np.sign(coef)               # gradient sign of the loss "forecast moves toward the other side"
    if kind == "fgsm":
        return w + eps * g
    adv = w + rng.uniform(-eps, eps, size=w.shape)
    for _ in range(steps):
        adv = np.clip(adv + alpha * g, w - eps, w + eps)
    return adv


# ----------------------------------------------------------------------------
# Evaluation on a stratified sample
# ----------------------------------------------------------------------------
def _cache_key(model_id: str, messages: list) -> str:
    return hashlib.sha1(json.dumps([model_id, messages], sort_keys=True).encode()).hexdigest()


def stratified_indices(closes: np.ndarray, window: int, n_windows: int, n_regimes: int, seed: int,
                       max_candidates: int = 200000) -> tuple[np.ndarray, list]:
    """Sample windows evenly across realised-volatility regimes (terciles of the window's std of
    1-minute changes). The proposal asks for >=100 observations per regime for regime-level estimates."""
    rng = np.random.default_rng(seed)
    last = len(closes) - 2          # need closes[idx + 1] as the target
    cand = np.arange(window, last)
    if len(cand) > max_candidates:
        cand = np.sort(rng.choice(cand, max_candidates, replace=False))
    vol = np.array([np.std(np.diff(closes[i - window + 1:i + 1])) for i in cand])
    edges = np.quantile(vol, np.linspace(0, 1, n_regimes + 1)[1:-1])
    regime = np.digitize(vol, edges)
    per = max(1, n_windows // n_regimes)
    chosen, counts = [], []
    for r in range(n_regimes):
        pool = cand[regime == r]
        take = min(per, len(pool))
        chosen.append(rng.choice(pool, take, replace=False)); counts.append(int(take))
    return np.sort(np.concatenate(chosen)), counts


def evaluate_llm_pipeline(closes_aligned: np.ndarray, targets: np.ndarray, X_rows: Optional[np.ndarray],
                          cfg: Dict, seed: int = 42, generator_fn: Optional[Callable[[int], np.ndarray]] = None,
                          mapper: Optional[SyntheticPriceMapper] = None, chat_fn: Optional[Callable] = None,
                          cache_dir: Optional[str] = "results/llm_cache",
                          context_fn: Optional[Callable[[int], str]] = None) -> Dict:
    """
    closes_aligned[k] = last observed close at row k; targets[k] = the NEXT close (already shifted, so
    targets[k] == closes_aligned[k + 1]); X_rows[k] = PCA features of row k (for nearest-synthetic lookup).
    Returns a flat dict of llm_* metrics; raises nothing for per-call failures (counted), but
    LLMModelUnavailableError propagates so the caller can stop early.
    """
    from ..evaluation.metrics import classification_metrics  # numpy/scipy only

    window = int(cfg.get("window_size", 10))
    horizon = int(cfg.get("horizon_minutes", 1))
    model_id = cfg["model_id"]
    api_base = cfg.get("api_base", DEFAULT_API_BASE)
    closes = np.asarray(closes_aligned, dtype=float).flatten()
    tgt = np.asarray(targets, dtype=float).flatten()
    idxs, regime_counts = stratified_indices(closes, window, int(cfg.get("n_windows", 102)),
                                             int(cfg.get("n_regimes", 3)), seed)
    rng = np.random.default_rng(seed + 1)

    use_syn = bool(generator_fn is not None and mapper is not None and getattr(mapper, "available", False)
                   and X_rows is not None and cfg.get("use_synthetic_context", True))
    pool_rows = pool_closes = None
    if use_syn:
        pool_rows = np.asarray(generator_fn(int(cfg.get("synthetic_pool", 2000))))
        pool_closes = mapper.to_close(pool_rows)
    k_syn = int(cfg.get("synthetic_neighbors", 5))

    diffs = np.diff(closes[: min(len(closes), 200000)])
    sd = float(np.std(diffs)) or 1e-4
    eps = float(cfg.get("price_attack_k", 2.0)) * sd
    alpha = eps * float(cfg.get("pgd_alpha_frac", 0.1)); steps = int(cfg.get("pgd_steps", 10))
    coef, _ = fit_ar_surrogate(closes, window)

    if chat_fn is None:
        chat_fn = lambda msgs: chat(msgs, model_id, api_base, max_tokens=int(cfg.get("max_tokens", 64)),  # noqa: E731
                                    timeout=int(cfg.get("timeout", 60)), retries=int(cfg.get("retries", 6)),
                                    backoff=float(cfg.get("backoff", 5.0)))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def ask(msgs):
        key = _cache_key(model_id, msgs)
        path = os.path.join(cache_dir, key + ".json") if cache_dir else None
        if path and os.path.isfile(path):
            with open(path, encoding="utf8") as fh:
                return parse_dual_response(json.load(fh)["text"]), True
        text = chat_fn(msgs)
        if path:
            with open(path, "w", encoding="utf8") as fh:
                json.dump({"text": text}, fh)
        return parse_dual_response(text), False

    jobs = []   # (window position j, kind, messages)
    attack_kind = {}
    for j, idx in enumerate(idxs):
        w = closes[idx - window + 1: idx + 1]
        syn = nearest_synthetic_closes(np.asarray(X_rows)[idx], pool_rows, pool_closes, k_syn) if use_syn else None
        kind = "fgsm" if j % 2 == 0 else "pgd"; attack_kind[j] = kind
        w_adv = attack_window(w, closes[idx], coef, kind, eps, alpha, steps, rng)
        prefix = reasoning_prefix(model_id) if cfg.get("nothink", True) else ""
        ctx = context_fn(int(idx)) if context_fn is not None else None   # from the REAL row; never attacked
        jobs.append((j, "clean", build_dual_messages(w, syn, horizon, prefix, ctx)))
        jobs.append((j, kind, build_dual_messages(w_adv, syn, horizon, prefix, ctx)))

    out = {}; errors = []; n_cached = 0

    def run(job):
        j, kind, msgs = job
        try:
            res, cached = ask(msgs)
            return j, kind, res, cached, None
        except (LLMModelUnavailableError, LLMAuthError):
            raise
        except Exception as exc:  # noqa: BLE001
            return j, kind, None, False, str(exc)[:200]

    with ThreadPoolExecutor(max_workers=int(cfg.get("max_workers", 4))) as ex:
        for j, kind, res, cached, err in ex.map(run, jobs):
            n_cached += int(cached)
            if err:
                errors.append(err)
            else:
                out[(j, kind)] = res

    # ---- forecast accuracy on clean windows ----
    clean = [(j, out[(j, "clean")]) for j in range(len(idxs)) if (j, "clean") in out and out[(j, "clean")]["forecast"] is not None]
    parse_fail = sum(1 for v in out.values() if v["forecast"] is None or v["threat_probability"] is None)
    res = {"llm_model_id": model_id, "llm_n_windows": len(idxs), "llm_n_calls": len(jobs), "llm_n_errors": len(errors),
           "llm_n_cached": n_cached, "llm_parse_failures": parse_fail, "llm_errors_sample": errors[:3],
           "llm_regime_counts": regime_counts, "llm_synthetic_context_used": use_syn,
           "llm_extra_context_used": context_fn is not None,
           "llm_price_attack_eps": eps, "llm_threat_threshold": QACL_THRESHOLD}
    if clean:
        p = np.array([v["forecast"] for _, v in clean]); y = tgt[idxs[[j for j, _ in clean]]]
        base = closes[idxs[[j for j, _ in clean]]]
        res["llm_rmse"] = float(np.sqrt(np.mean((p - y) ** 2))); res["llm_mae"] = float(np.mean(np.abs(p - y)))
        res["llm_persistence_rmse"] = float(np.sqrt(np.mean((base - y) ** 2)))
        res["llm_rmse_over_persistence"] = (res["llm_rmse"] / res["llm_persistence_rmse"]
                                            if res["llm_persistence_rmse"] > 0 else float("nan"))
        res["llm_n_forecasts"] = len(clean)
    else:
        res.update(llm_rmse=float("nan"), llm_mae=float("nan"), llm_n_forecasts=0)

    # ---- attack success: directional flip vs the last real close ----
    for kind in ("fgsm", "pgd"):
        flips = []
        for j in range(len(idxs)):
            if attack_kind[j] != kind or (j, "clean") not in out or (j, kind) not in out:
                continue
            a, b = out[(j, "clean")]["forecast"], out[(j, kind)]["forecast"]
            if a is None or b is None:
                continue
            last = closes[idxs[j]]
            flips.append(np.sign(a - last) != np.sign(b - last))
        res[f"llm_asr_{kind}"] = 100.0 * float(np.mean(flips)) if flips else float("nan")
        res[f"llm_n_{kind}"] = len(flips)
    vals = [res[k] for k in ("llm_asr_fgsm", "llm_asr_pgd") if not np.isnan(res[k])]
    res["llm_asr"] = float(np.mean(vals)) if vals else float("nan")

    # ---- threat detection from the LLM's own probability (FPR / accuracy: DV3, DV5) ----
    yt, yp = [], []
    for (j, kind), v in out.items():
        if v["threat_probability"] is None:
            continue
        yt.append(0 if kind == "clean" else 1); yp.append(v["threat_probability"])
    if len(set(yt)) == 2:
        m = classification_metrics(np.array(yt), np.array(yp), threshold=QACL_THRESHOLD)
        res.update({f"llm_threat_{k}": m[k] for k in ("fpr", "recall", "precision", "f1", "accuracy", "auprc")})
        res["llm_threat_n"] = len(yt)
    else:
        res.update(llm_threat_fpr=float("nan"), llm_threat_accuracy=float("nan"), llm_threat_n=len(yt))
    res["llm_detector"] = "llm_threat_probability"
    return res


def evaluate_if_enabled(llm_cfg: Optional[Dict], last_prices: np.ndarray, y_next: np.ndarray, X_rows: np.ndarray,
                        seed: int, generator_fn: Optional[Callable[[int], np.ndarray]] = None,
                        processed_dir: str = "data/processed", **kw) -> Dict:
    """Hook called from each model's evaluate(). Returns {} when the pipeline is disabled, a status
    entry when it cannot run, or the llm_* metrics. A retired model stops the LLM stage only, with
    the reason recorded; it does not discard the rest of the model's evaluation."""
    if not llm_cfg or not llm_cfg.get("enabled", False):
        return {}
    if last_prices is None:
        return {"llm_pipeline_status": "skipped: no real close series (last_input_prices) was passed"}
    if not os.environ.get("NVIDIA_API_KEY"):
        return {"llm_pipeline_status": "skipped: NVIDIA_API_KEY not set"}
    mapper = SyntheticPriceMapper(processed_dir) if generator_fn is not None else None
    try:
        res = evaluate_llm_pipeline(last_prices, y_next, X_rows, llm_cfg, seed=seed,
                                    generator_fn=generator_fn, mapper=mapper, **kw)
        res["llm_pipeline_status"] = "ok"
        if generator_fn is not None and not res["llm_synthetic_context_used"]:
            res["llm_synthetic_context_note"] = getattr(mapper, "reason", "synthetic context disabled")
        return res
    except LLMModelUnavailableError as exc:
        return {"llm_pipeline_status": f"model unavailable: {str(exc)[:300]}"}
    except LLMAuthError as exc:
        return {"llm_pipeline_status": f"key rejected: {str(exc)[:300]}"}
