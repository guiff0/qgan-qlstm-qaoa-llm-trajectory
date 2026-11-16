"""Why does the in-run H3 positive control degrade by ~0.3 % at shift_sd 0.5 while calibrate_poisoning reports ~660 %?

    python -m scripts.diagnose_poisoning_gap                      # real data: data/processed/X_test.npy, y_test.npy
    python -m scripts.diagnose_poisoning_gap --synthetic          # self-test on built-in data (no files needed)
    python -m scripts.diagnose_poisoning_gap --grid 0.5 1 2 4 8 16 32 --run-seeds 43 44 45

Both numbers come from the SAME function (poisoning_protocol.degradation = (poisoned RMSE - control RMSE) / control RMSE
of a ridge head), so the gap cannot be the attack or the metric. It must be the SETUP around that function. The code shows
three differences, and this script varies them one at a time while everything else is held fixed:

  chronology  calibrate_poisoning picks a random subset, SORTS it, trains on the earlier half and evaluates on the LATER half
              (a time split). evaluate_poisoning_resistance permutes ALL rows and splits at random (an i.i.d. split).
              Under a time split the evaluation rows can lie outside the training feature range (volume spikes, regime
              change); a head carrying even a small spurious slope then EXTRAPOLATES and RMSE explodes. That measures
              distribution shift, not poisoning.
  size        200 000 rows (100k/100k) in the calibration vs the full test set (100k training rows for the control, ~half
              the test set to evaluate) in-run.
  seeds       pilot seeds 100-104 vs the run seeds (43...). The attack direction is drawn from the seed.

It runs FIVE recipes x TWO seed sets x the shift grid, then:
  * reproduces the calibration curve (latest results/poisoning_calibration_*.json) -> tells you if the arrays or the code changed;
  * reproduces the in-run values recorded in results/all_results.csv (poisoning_pc_degradation_pct) -> tells you if the
    in-run control saw the arrays you are looking at;
  * attributes the gap: which single change moves the in-run recipe to the calibration number;
  * reports a SCALE-FREE alternative (extra RMSE as a fraction of sd(y)) and the evaluation-set drift, so you can see
    whether a large "degradation %" is the head learning the poison or the head extrapolating.
It also prints the smallest shift_sd at which EACH recipe reaches the pre-set floor and target.

Exit code 0: the in-run and calibration recipes agree within --agree-factor at the configured shift. 1: they do not (the
calibrated strength is not validated for the in-run protocol). 2: cannot run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.attacks.poisoning import inject_triggered_shift  # noqa: E402
from src.evaluation.poisoning_protocol import fit_ridge, predict, rmse  # noqa: E402

DEFAULT_GRID = (0.5, 1, 2, 3, 4, 6, 8, 16, 32)
RECIPES = {
    "inrun": "evaluate_poisoning_resistance: all rows, random half split, control trained on the first pc_max_rows",
    "inrun_chrono": "same rows as in-run but a TIME split (earlier half trains, later half evaluates)",
    "calib_random": "calibration's 2*max_rows subset, but a RANDOM half split",
    "calibration": "calibrate_poisoning: 2*max_rows random subset, sorted, earlier half trains / later half evaluates",
}
# one row per recipe above; the fifth "recipe" is the seed set, handled as a second axis below.


# ---------------------------------------------------------------- recipes (each returns Xtr, ytr, Xev, yev)
def split_inrun(X, y, seed, pc_max_rows, **_):
    n = len(y)
    half = n // 2
    perm = np.random.default_rng(seed).permutation(n)
    tr, ev = perm[:half], perm[half:]
    k = min(pc_max_rows, half)
    return X[tr[:k]], y[tr[:k]], X[ev], y[ev]


def split_inrun_chrono(X, y, seed, pc_max_rows, **_):
    n = len(y)
    half = n // 2
    k = min(pc_max_rows, half)
    return X[:k], y[:k], X[half:], y[half:]


def _subset(X, y, subset_seed, max_rows):
    n = min(len(y), 2 * max_rows)
    sel = np.sort(np.random.default_rng(subset_seed).choice(len(y), n, replace=False))
    return X[sel], y[sel], n


def split_calibration(X, y, seed, pc_max_rows, subset_seed=None, **_):
    # calibrate_poisoning draws the subset ONCE from the first pilot seed; the per-seed part is only the attack.
    Xs, ys, n = _subset(X, y, subset_seed if subset_seed is not None else seed, pc_max_rows)
    half = n // 2
    return Xs[:half], ys[:half], Xs[half:], ys[half:]


def split_calib_random(X, y, seed, pc_max_rows, subset_seed=None, **_):
    Xs, ys, n = _subset(X, y, subset_seed if subset_seed is not None else seed, pc_max_rows)
    half = n // 2
    perm = np.random.default_rng(seed).permutation(n)
    return Xs[perm[:half]], ys[perm[:half]], Xs[perm[half:]], ys[perm[half:]]


BUILD = {"inrun": split_inrun, "inrun_chrono": split_inrun_chrono, "calib_random": split_calib_random,
         "calibration": split_calibration}


# ---------------------------------------------------------------- one probe
def prepare_control(Xtr, ytr, Xev, ridge):
    """The un-poisoned head does not depend on shift_sd, so fit it and predict ONCE per split (9x faster over a grid)."""
    control = fit_ridge(Xtr, ytr, ridge)
    return predict(control, Xev)


def probe(Xtr, ytr, Xev, yev, *, fraction, shift, seed, ridge, control_pred=None):
    """Same arithmetic as poisoning_protocol.degradation, plus diagnostics that explain a large value."""
    atk = inject_triggered_shift(Xtr, ytr, fraction=fraction, shift_sd=shift, seed=seed)
    poisoned = fit_ridge(Xtr, atk["y_poisoned"], ridge)
    pc = control_pred if control_pred is not None else prepare_control(Xtr, ytr, Xev, ridge)
    pp = predict(poisoned, Xev)
    c, p = rmse(pc, yev), rmse(pp, yev)
    sdy = float(np.std(yev)) or float("nan")
    return {"degradation_pct": (p - c) / c * 100.0 if c > 0 else float("nan"),
            "control_rmse": c, "poisoned_rmse": p,
            "extra_rmse_over_sd_y": (p - c) / sdy,                       # scale-free: how much worse, in units of sd(y)
            "control_rmse_over_sd_y": c / sdy,
            "pred_shift_over_sd_y": float(np.sqrt(np.mean((pp - pc) ** 2))) / sdy}   # how far the poison moves predictions


def drift(Xtr, Xev) -> dict:
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = np.abs((Xev.mean(axis=0) - mu) / sd)
    r = Xev.std(axis=0) / sd
    out_frac = float(np.mean(np.any(np.abs((Xev - mu) / sd) > 6.0, axis=1)))
    return {"max_abs_mean_shift_in_train_sd": float(z.max()), "max_sd_ratio_eval_over_train": float(r.max()),
            "share_of_eval_rows_beyond_6_train_sd": out_frac}


# ---------------------------------------------------------------- main experiment
def run_grid(X, y, *, grid, seed_sets, fraction, ridge, pc_max_rows, subset_seed):
    res = {}
    for rname, fn in BUILD.items():
        for sname, seeds in seed_sets.items():
            per_shift = {float(s): [] for s in grid}
            for sd in seeds:
                Xtr, ytr, Xev, yev = fn(X, y, sd, pc_max_rows, subset_seed=subset_seed)     # split ONCE per seed
                cpred = prepare_control(Xtr, ytr, Xev, ridge)                                  # control ONCE per seed
                for s in grid:
                    per_shift[float(s)].append(probe(Xtr, ytr, Xev, yev, fraction=fraction, shift=float(s), seed=int(sd),
                                                     ridge=ridge, control_pred=cpred))
                del Xtr, ytr, Xev, yev, cpred
            for s, vals in per_shift.items():
                agg = {k: float(np.nanmedian([v[k] for v in vals])) for k in vals[0]}
                agg["min_degradation_pct"] = float(np.nanmin([v["degradation_pct"] for v in vals]))
                agg["per_seed_degradation_pct"] = [float(v["degradation_pct"]) for v in vals]
                res[(rname, sname, s)] = agg
    return res


def first_reaching(res, rname, sname, grid, target):
    for s in grid:
        if res[(rname, sname, float(s))]["degradation_pct"] >= target:
            return float(s)
    return None


def attribute(res, shift, run_name="run", pilot_name="pilot"):
    """Which single change moves the in-run recipe toward the calibration number? Ratios are of median degradation."""
    g = lambda r, sn: res[(r, sn, shift)]["degradation_pct"]
    base = g("inrun", run_name)
    cal = g("calibration", pilot_name)
    steps = {
        "chronology (random split -> time split, same rows)": (g("inrun", run_name), g("inrun_chrono", run_name)),
        "chronology (random split -> time split, calibration's subset)": (g("calib_random", run_name), g("calibration", run_name)),
        "size (full test set -> 2*max_rows subset, random split)": (g("inrun", run_name), g("calib_random", run_name)),
        "seeds (run seeds -> pilot seeds, in-run recipe)": (g("inrun", run_name), g("inrun", pilot_name)),
        "seeds (run seeds -> pilot seeds, calibration recipe)": (g("calibration", run_name), g("calibration", pilot_name)),
    }
    ratios = {k: (b / a if a and np.isfinite(a) and a > 0 else float("nan")) for k, (a, b) in steps.items()}
    total = cal / base if base and base > 0 else float("nan")
    return {"in_run_pct": base, "calibration_pct": cal, "total_ratio": total, "factor_ratios": ratios}


def narrative(att, drift_by_recipe, agree_factor):
    tot = att["total_ratio"]
    if np.isfinite(tot) and 1 / agree_factor <= tot <= agree_factor:
        return ["The in-run and calibration recipes AGREE at this shift: the calibrated strength carries over to the in-run protocol."]
    lines = [f"The recipes DISAGREE: calibration {att['calibration_pct']:.3g} % vs in-run {att['in_run_pct']:.3g} % "
             f"(x{tot:.3g}). Single-change ratios (>1 moves toward the calibration number):"]
    ranked = sorted(((k, v) for k, v in att["factor_ratios"].items() if np.isfinite(v)), key=lambda kv: -abs(np.log(kv[1])))
    lines += [f"    x{v:<10.3g} {k}" for k, v in ranked]
    if ranked and abs(np.log(ranked[0][1])) >= np.log(3):
        lines.append(f"  Dominant factor: {ranked[0][0]}.")
    else:
        lines.append("  No single factor moves it by 3x: check the feature set (--processed-dir arrays vs what the run scored) "
                     "and the recorded-vs-replica table below.")
    d = drift_by_recipe.get("calibration")
    if d and (d["share_of_eval_rows_beyond_6_train_sd"] > 0.01 or d["max_sd_ratio_eval_over_train"] > 3):
        lines.append("  The calibration's evaluation half lies OUTSIDE the training feature range "
                     f"({d['share_of_eval_rows_beyond_6_train_sd']:.1%} of rows beyond 6 sd, max sd ratio "
                     f"{d['max_sd_ratio_eval_over_train']:.1f}): its large degradation is extrapolation under distribution shift, "
                     "not the head learning the poison. It is not evidence that the attack is strong in-run.")
    return lines


def latest_calibration_json():
    files = sorted(glob.glob(os.path.join(ROOT, "results", "poisoning_calibration_*.json")))
    if not files:
        return None, None
    with open(files[-1], encoding="utf8") as fh:
        return files[-1], json.load(fh)


def recorded_rows(csv_path, limit):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001 - optional input
        return []
    need = {"poisoning_pc_degradation_pct", "poisoning_shift_sd", "seed"}
    if not need.issubset(df.columns):
        return []
    d = df.dropna(subset=list(need))
    return d.tail(limit)[["run_id", "seed", "poisoning_shift_sd", "poisoning_pc_degradation_pct"]].to_dict("records") \
        if "run_id" in d.columns else []


def synthetic_data(n=60000, d=20, seed=0, spike=8.0):
    """Small random-walk-like data with a weak signal, plus volume-spike drift in the LATER part (what a time split sees)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] = np.cumsum(rng.normal(size=n)) / 50.0
    late = np.arange(n) > int(0.6 * n)
    X[late, 1:4] *= 1.0 + spike * rng.random((late.sum(), 3)) ** 3          # heavy-tailed spikes only late in the series
    y = 1e-4 * (0.05 * X[:, 0] + rng.normal(size=n))
    return X, y


def main(argv=None) -> int:
    from src.utils.config import load_config
    cfg = load_config().get("poisoning", {}) or {}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use built-in data (self-test; proves the diagnostic works)")
    ap.add_argument("--synthetic-spike", type=float, default=8.0,
                    help="size of the late-period volume spikes in the synthetic data (a larger value makes a time split blow up)")
    ap.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    ap.add_argument("--run-seeds", type=int, nargs="+", default=[43, 44, 45, 46, 47])
    ap.add_argument("--pilot-seeds", type=int, nargs="+", default=cfg.get("pilot_seeds", [100, 101, 102, 103, 104]))
    ap.add_argument("--fraction", type=float, default=float(cfg.get("fraction", 0.2)))
    ap.add_argument("--ridge", type=float, default=float(cfg.get("ridge", 1.0)))
    ap.add_argument("--pc-max-rows", type=int, default=int(cfg.get("pc_max_rows", 100000)))
    ap.add_argument("--shift", type=float, default=float(cfg.get("shift_sd", 0.5)), help="configured shift_sd to attribute at")
    ap.add_argument("--floor", type=float, default=float(cfg.get("pc_floor_pct", 5.0)))
    ap.add_argument("--target", type=float, default=float(cfg.get("pc_target_pct", 10.0)))
    ap.add_argument("--agree-factor", type=float, default=3.0, help="recipes agree when within this factor of each other")
    ap.add_argument("--results-csv", default=os.path.join("results", "all_results.csv"))
    ap.add_argument("--max-csv-rows", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.synthetic:
        X, y = synthetic_data(spike=a.synthetic_spike)
        a.pc_max_rows = min(a.pc_max_rows, 15000)
        src = "synthetic"
    else:
        pdir = a.processed_dir or load_config()["data"]["processed_dir"]
        try:
            X, y = np.load(os.path.join(pdir, "X_test.npy")), np.load(os.path.join(pdir, "y_test.npy"))
        except FileNotFoundError as exc:
            print(f"cannot read the processed test arrays in {pdir}: {exc}\nrun scripts/prepare_data.py first, or use --synthetic")
            return 2
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64)
        src = pdir
    grid = sorted(set(a.grid) | {a.shift})
    seed_sets = {"run": a.run_seeds, "pilot": a.pilot_seeds}
    print(f"data: {src}  X{tuple(X.shape)}  fraction {a.fraction}  ridge {a.ridge}  pc_max_rows {a.pc_max_rows}  "
          f"run seeds {a.run_seeds}  pilot seeds {a.pilot_seeds}")
    res = run_grid(X, y, grid=grid, seed_sets=seed_sets, fraction=a.fraction, ridge=a.ridge, pc_max_rows=a.pc_max_rows,
                   subset_seed=a.pilot_seeds[0])

    # ---- table: median degradation % by recipe x seed set x shift
    print("\nMEDIAN POSITIVE-CONTROL DEGRADATION (%)  -- same attack, same head, different setup")
    head = f"{'recipe':<14}{'seeds':<7}" + "".join(f"{s:>11g}" for s in grid)
    print(head + "   <- shift_sd\n" + "-" * len(head))
    for r in BUILD:
        for sn in seed_sets:
            print(f"{r:<14}{sn:<7}" + "".join(f"{res[(r, sn, float(s))]['degradation_pct']:>11.3g}" for s in grid))
    print("\nSCALE-FREE VIEW at the configured shift: extra RMSE as a share of sd(y), and how far the poison moves predictions")
    print(f"{'recipe':<14}{'seeds':<7}{'control rmse/sd(y)':>20}{'extra rmse/sd(y)':>19}{'pred move/sd(y)':>18}")
    for r in BUILD:
        for sn in seed_sets:
            v = res[(r, sn, float(a.shift))]
            print(f"{r:<14}{sn:<7}{v['control_rmse_over_sd_y']:>20.4g}{v['extra_rmse_over_sd_y']:>19.4g}{v['pred_shift_over_sd_y']:>18.4g}")

    # ---- drift of the evaluation half under each recipe (first run seed)
    drift_by = {}
    for r, fn in BUILD.items():
        Xtr, ytr, Xev, yev = fn(X, y, a.run_seeds[0], a.pc_max_rows, subset_seed=a.pilot_seeds[0])
        drift_by[r] = drift(Xtr, Xev)
    print("\nEVALUATION-SET DRIFT vs the training rows (first run seed)")
    print(f"{'recipe':<14}{'max |mean shift| (train sd)':>29}{'max sd ratio':>14}{'rows > 6 sd':>13}")
    for r, d in drift_by.items():
        print(f"{r:<14}{d['max_abs_mean_shift_in_train_sd']:>29.3g}{d['max_sd_ratio_eval_over_train']:>14.3g}"
              f"{d['share_of_eval_rows_beyond_6_train_sd']:>13.2%}")

    # ---- minimum shift reaching floor / target
    print(f"\nSMALLEST shift_sd IN THE GRID reaching the floor ({a.floor:g} %) / the target ({a.target:g} %)")
    reach = {}
    for r in BUILD:
        for sn in seed_sets:
            fl, tg = first_reaching(res, r, sn, grid, a.floor), first_reaching(res, r, sn, grid, a.target)
            reach[f"{r}/{sn}"] = {"floor": fl, "target": tg}
            print(f"  {r:<14}{sn:<7} floor: {fl if fl is not None else 'not reached':<12} target: {tg if tg is not None else 'not reached'}")

    # ---- reproduce the calibration json
    cal_note = "no results/poisoning_calibration_*.json found"
    cal_ok = None
    cpath, cal = (None, None) if a.synthetic else latest_calibration_json()
    if a.synthetic:
        cal_note = "skipped (synthetic data cannot reproduce a calibration made on your real arrays)"
    if cal:
        diffs = []
        for row in cal["curve"]:
            k = ("calibration", "pilot", float(row["shift_sd"]))
            if k in res:
                mine, theirs = res[k]["degradation_pct"], row["median_degradation_pct"]
                diffs.append(abs(mine - theirs) / max(abs(theirs), 1e-9))
        cal_ok = bool(diffs) and max(diffs) < 1e-3
        cal_note = (f"{os.path.basename(cpath)}: " + ("REPRODUCED (calibration recipe matches the saved curve)" if cal_ok else
                    f"NOT reproduced (max relative difference {max(diffs):.2g}): the arrays or the code changed since that "
                    "calibration" if diffs else "grid does not overlap the saved curve"))
    print(f"\nCALIBRATION CURVE CHECK: {cal_note}")

    # ---- reproduce recorded in-run values
    rec = [] if a.synthetic else recorded_rows(
        os.path.join(ROOT, a.results_csv) if not os.path.isabs(a.results_csv) else a.results_csv, a.max_csv_rows)
    rec_out = []
    if rec:
        print("\nRECORDED IN-RUN POSITIVE CONTROL vs the in-run recipe replayed on these arrays (same seed, same shift)")
        print(f"{'run_id':<44}{'seed':>6}{'shift':>7}{'recorded %':>13}{'replayed %':>13}")
        for r in rec:
            Xtr, ytr, Xev, yev = split_inrun(X, y, int(r["seed"]), a.pc_max_rows)
            rp = probe(Xtr, ytr, Xev, yev, fraction=a.fraction, shift=float(r["poisoning_shift_sd"]), seed=int(r["seed"]),
                       ridge=a.ridge)["degradation_pct"]
            rec_out.append({**{k: r[k] for k in ("run_id", "seed", "poisoning_shift_sd")},
                            "recorded_pct": float(r["poisoning_pc_degradation_pct"]), "replayed_pct": rp})
            print(f"{str(r['run_id'])[:43]:<44}{int(r['seed']):>6}{float(r['poisoning_shift_sd']):>7g}"
                  f"{float(r['poisoning_pc_degradation_pct']):>13.4g}{rp:>13.4g}")
        close = [abs(x["replayed_pct"] - x["recorded_pct"]) <= max(0.25 * abs(x["recorded_pct"]), 0.02) for x in rec_out]
        print("  -> " + ("the replay matches the recorded values: the in-run number IS what this recipe gives on these arrays."
                         if all(close) else "the replay does NOT match: the run scored different rows or features than "
                                            "X_test.npy (feature selection, a different split, or an older code version)."))

    # ---- attribution + verdict
    att = attribute(res, float(a.shift))
    print(f"\nGAP ATTRIBUTION at shift_sd = {a.shift:g}")
    for line in narrative(att, drift_by, a.agree_factor):
        print("  " + line)
    agree = np.isfinite(att["total_ratio"]) and 1 / a.agree_factor <= att["total_ratio"] <= a.agree_factor
    inrun_fl = reach["inrun/run"]["floor"]
    print("\nCONSEQUENCE FOR H3")
    if agree:
        print(f"  The recipes agree, so shift_sd {a.shift:g} is validated for the in-run protocol.")
    else:
        print(f"  shift_sd {a.shift:g} is NOT validated: calibration chose it on a recipe that does not match the in-run one.")
        print("  The in-run positive control needs shift_sd >= "
              + (f"{inrun_fl:g}" if inrun_fl is not None else f"more than {max(grid):g} (no grid value reaches the floor)")
              + f" to clear the {a.floor:g} % floor. Re-calibrate with the IN-RUN recipe (random split, in-run row count) "
                "or make calibrate_poisoning use the same split as evaluate_poisoning_resistance, then re-freeze.")
        print("  Until then report H3 as NOT DECIDABLE; do not start more QGAN runs at this shift_sd.")

    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "data": src, "grid": grid,
           "fraction": a.fraction, "ridge": a.ridge, "pc_max_rows": a.pc_max_rows, "run_seeds": a.run_seeds,
           "pilot_seeds": a.pilot_seeds, "configured_shift_sd": a.shift, "floor_pct": a.floor, "target_pct": a.target,
           "recipes": RECIPES,
           "table": [{"recipe": r, "seeds": sn, "shift_sd": s, **{k: v for k, v in res[(r, sn, s)].items()}}
                     for (r, sn, s) in res],
           "drift": drift_by, "smallest_shift_reaching": reach, "calibration_check": cal_note,
           "recorded_vs_replayed": rec_out, "attribution": att, "recipes_agree": bool(agree)}
    path = a.out or os.path.join(ROOT, "results", f"poisoning_gap_diagnostic_{out['generated_utc']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwritten: {path}")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
