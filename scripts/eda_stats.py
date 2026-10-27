"""
Exploratory-data-analysis statistics for Chapter 3/4 (Figures 3-5, 7, 9-13; Tables 26, 27, 29-31).

Re-runs the SAME functions prepare_data.py uses (load_dukascopy, merge_all_sources, clean_prices,
add_technical_indicators, clean_indicators, expand_feature_space), but year by year with a warm-up
overlap so a 5.5M-row, 54-feature frame never has to sit in memory at once. Global MAD statistics
are computed on the full series first, so the cleaning decisions are identical to the one-shot run.

Run:  python -m scripts.eda_stats            (writes results/eda_out/*)
"""
from __future__ import annotations
import json, os, pickle, sys, time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_loader import load_dukascopy, load_fred, load_vix, merge_all_sources
from src.data.preprocessing import (clean_prices, add_technical_indicators, mad_outlier_mask,
                                    domain_implausible_mask)
from scripts.prepare_data import expand_feature_space, INDICATOR_COLS, PRICE_COLS
from src.utils.config import load_config

OUT = "eda_out"; os.makedirs(OUT, exist_ok=True)
OVERLAP = 2000
cfg = load_config(); dc = cfg["data"]
t0 = time.time()

raw = load_dukascopy(dc["dukascopy_file"])
fred = load_fred(dc["fred_file"])
vix = load_vix(dc["yfinance_ticker"], start="2011-12-18", end="2025-12-31")
stats = {"raw_rows": int(len(raw)), "raw_first": str(raw.timestamp.iloc[0]), "raw_last": str(raw.timestamp.iloc[-1])}
raw = raw[raw.timestamp <= pd.Timestamp(dc["test_end"], tz="UTC") + pd.Timedelta(days=1)].reset_index(drop=True)

# ---------------- flagged-row accounting on the raw prices (global statistics)
dom = domain_implausible_mask(raw, PRICE_COLS, dc.get("max_abs_return", 0.5))
flag = {"domain": int(dom.sum())}
tmp = raw[PRICE_COLS].copy(); tmp.loc[dom, :] = np.nan
mad_any = pd.Series(False, index=raw.index)
for c in PRICE_COLS:
    m = mad_outlier_mask(tmp[c], dc["outlier_threshold_price"]); flag[f"mad_{c}"] = int(m.sum()); mad_any |= m
flag["mad_any_price"] = int(mad_any.sum()); flag["any_price_flag"] = int((mad_any | dom).sum())
month = raw.timestamp.dt.tz_convert(None).dt.to_period("M")
monthly = pd.DataFrame({"month": month, "domain": dom.values, "mad": mad_any.values}).groupby("month").sum()
monthly.index = monthly.index.astype(str); monthly.to_csv(f"{OUT}/flagged_by_month.csv")
stats["flag"] = flag
stats["dup_ts"] = int(raw.timestamp.duplicated().sum())
stats["nonpositive_volume"] = int((raw.volume <= 0).sum())
stats["zero_range_bars"] = int((raw.high == raw.low).sum())
gaps = raw.timestamp.diff().dt.total_seconds().div(60)
stats["gap_gt_60min"] = int((gaps > 60).sum()); stats["gap_gt_1day"] = int((gaps > 1440).sum())
stats["rows_by_year"] = raw.timestamp.dt.year.value_counts().sort_index().to_dict()
stats["flag_by_year"] = pd.DataFrame({"y": raw.timestamp.dt.year, "f": (mad_any | dom).values}).groupby("y").f.sum().to_dict()
del tmp, month

# ---------------- clean prices globally (identical to prepare_data.py, memory-light: 5 columns)
clean = clean_prices(raw[["timestamp"] + PRICE_COLS + ["volume"]], PRICE_COLS,
                     n_mads_price=dc["outlier_threshold_price"], max_abs_return=dc.get("max_abs_return", 0.5))
del raw
clean["year"] = clean.timestamp.dt.year
years = sorted(clean.year.unique())
print(f"[{time.time()-t0:.0f}s] prices cleaned: {len(clean):,} rows")

IND_FLAGS = []
YEAR_AFTER = {}
def build(y, ind_stats=None):
    """Return the engineered frame for calendar year y (warm-up rows trimmed) and its raw indicators."""
    idx = np.where(clean.year.values == y)[0]
    lo = max(idx[0] - OVERLAP, 0); hi = idx[-1] + 1
    ch = clean.iloc[lo:hi].drop(columns="year").reset_index(drop=True)
    ch = merge_all_sources(ch, fred, vix, release_lags=dc.get("fred_release_lags"),
                           vix_lag_days=dc.get("vix_lag_days", 1), drop_leading_incomplete=False)
    ch = add_technical_indicators(ch)
    raw_ind = ch[INDICATOR_COLS].copy()
    if ind_stats is not None:
        anyf = pd.Series(False, index=ch.index)
        for c in INDICATOR_COLS:
            med, mad = ind_stats[c]
            if mad > 0: anyf |= (ch[c] - med).abs() / mad > dc["outlier_threshold_indicator"]
        yk = (ch.timestamp.dt.year == y).values
        IND_FLAGS.append(pd.DataFrame({"month": ch.timestamp[yk].dt.tz_convert(None).dt.to_period("M").astype(str),
                                       "any_indicator_flag": anyf[yk].values}).groupby("month").sum())
        for c in INDICATOR_COLS:
            med, mad = ind_stats[c]
            m = (ch[c] - med).abs() / mad > dc["outlier_threshold_indicator"] if mad > 0 else pd.Series(False, index=ch.index)
            ch.loc[m, c] = np.nan
            ch[c] = ch[c].interpolate(method="linear", limit_direction="both")
        ch, feats = expand_feature_space(ch)
        ch["next_logret"] = np.log(ch["close"].shift(-1) / ch["close"])
        keep = (ch.timestamp.dt.year == y).values
        return ch[keep], raw_ind[keep], feats
    return None, raw_ind[(ch.timestamp.dt.year == y).values], None

# ---------------- pass 1: global indicator median / MAD
import contextlib, io
buf = io.StringIO()
inds = []
for y in years:
    with contextlib.redirect_stdout(buf):
        _, ri, _ = build(y)
    inds.append(ri.astype("float64"))
IND = pd.concat(inds); del inds
ind_stats = {}; ind_flag = {}
for c in INDICATOR_COLS:
    s = IND[c]; med = float(s.median()); mad = float((s - med).abs().median() * 1.4826)
    ind_stats[c] = (med, mad); ind_flag[c] = int(((s - med).abs() / mad > dc["outlier_threshold_indicator"]).sum()) if mad > 0 else 0
stats["indicator_flag"] = ind_flag
print(f"[{time.time()-t0:.0f}s] pass 1 done; indicator flags {ind_flag}")
del IND

# ---------------- pass 2: features, accumulators
feature_cols = json.load(open("data/processed/feature_cols.json")) if os.path.exists("data/processed/feature_cols.json") else None
rng = np.random.default_rng(0)
acc = {}      # (split, regime) -> counts
rs = {}       # regime -> sums for mean/std/min/max
cross = None; sx = None; ntr = 0
samples = []; daily = []
split_edges = {"train": (dc["train_start"], dc["train_end"]), "val": (dc["val_start"], dc["val_end"]), "test": (dc["test_start"], dc["test_end"])}
dropped_warm = 0; total_after = 0; nan_before = {}
for y in years:
    with contextlib.redirect_stdout(buf):
        ch, _, feats = build(y, ind_stats)
    if feature_cols is None: feature_cols = feats
    n0 = len(ch)
    for c in feature_cols:
        nan_before[c] = nan_before.get(c, 0) + int(ch[c].isna().sum())
    ch = ch.dropna(subset=feature_cols).reset_index(drop=True)
    dropped_warm += n0 - len(ch); total_after += len(ch); YEAR_AFTER[int(y)] = [int(n0), int(len(ch))]
    ts = ch.timestamp.dt.tz_convert(None)
    split = np.where(ts <= pd.Timestamp(split_edges["train"][1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), "train",
             np.where(ts <= pd.Timestamp(split_edges["val"][1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), "val", "test"))
    v = ch["vix"].values
    regime = np.where(v < 15, "low", np.where(v < 25, "normal", "high"))
    for s_, r_ in zip(*np.unique(np.char.add(np.char.add(split.astype(str), "|"), regime.astype(str)), return_counts=True)):
        acc[s_] = acc.get(s_, 0) + int(r_)
    X = ch[feature_cols].to_numpy(dtype="float64")
    for r_ in ("low", "normal", "high"):
        m = regime == r_
        if not m.any(): continue
        d = rs.setdefault(r_, dict(n=0, s=np.zeros(len(feature_cols)), ss=np.zeros(len(feature_cols)),
                                   mn=np.full(len(feature_cols), np.inf), mx=np.full(len(feature_cols), -np.inf)))
        Xm = X[m]; d["n"] += len(Xm); d["s"] += Xm.sum(0); d["ss"] += (Xm ** 2).sum(0)
        d["mn"] = np.minimum(d["mn"], Xm.min(0)); d["mx"] = np.maximum(d["mx"], Xm.max(0))
    tr = split == "train"
    if tr.any():
        Xt = X[tr]; sx = Xt.sum(0) if sx is None else sx + Xt.sum(0)
        cross = Xt.T @ Xt if cross is None else cross + Xt.T @ Xt; ntr += len(Xt)
    k = min(25000, len(ch)); sel = np.sort(rng.choice(len(ch), k, replace=False))
    smp = ch.iloc[sel][["timestamp"] + feature_cols + ["next_logret"]].copy()
    smp["split"] = split[sel]; smp["regime"] = regime[sel]; samples.append(smp)
    dm = ch[(ts.dt.hour == 12) & (ts.dt.minute == 0)]
    daily.append(dm[["timestamp", "close"] + INDICATOR_COLS].assign(split=split[dm.index]))
    print(f"[{time.time()-t0:.0f}s] {y}: {len(ch):,} rows")

mean = sx / ntr; cov = cross / ntr - np.outer(mean, mean); sd = np.sqrt(np.diag(cov))
corr = cov / np.outer(sd, sd)
np.save(f"{OUT}/corr_train.npy", corr)
pd.concat(samples).to_pickle(f"{OUT}/sample.pkl")
pd.concat(daily).to_pickle(f"{OUT}/daily_indicators.pkl")
regime_stats = {}
for r_, d in rs.items():
    mu = d["s"] / d["n"]; var = d["ss"] / d["n"] - mu ** 2
    regime_stats[r_] = dict(n=d["n"], mean=dict(zip(feature_cols, mu)), std=dict(zip(feature_cols, np.sqrt(np.maximum(var, 0)))),
                            min=dict(zip(feature_cols, d["mn"])), max=dict(zip(feature_cols, d["mx"])))
stats.update(feature_cols=feature_cols, split_regime_counts=acc, regime_stats=regime_stats,
             rows_after_dropna=total_after, dropped_warmup_or_nan=dropped_warm, nan_before_dropna=nan_before,
             train_rows_used_for_corr=ntr, elapsed_s=time.time() - t0)
pd.concat(IND_FLAGS).to_csv(f"{OUT}/indicator_flags_by_month.csv")
stats["year_before_after_dropna"] = YEAR_AFTER
stats["zero_volume_by_year"] = pd.DataFrame({"y": clean.year, "z": (clean.volume <= 0)}).groupby("y").z.sum().to_dict()
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1, default=str)
print("DONE", f"{time.time()-t0:.0f}s", "rows after dropna:", total_after)
