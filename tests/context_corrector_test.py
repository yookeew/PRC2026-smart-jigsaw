"""Context corrector on the final chain (reports/context_corrector_preregistration.md).

Input p = v33's method on the Jan+Jul 2025 holdout (v31 chain + LOBT window + NM-unmatched
lines). A LightGBM learns (y - p), cross-fit Jan <-> Jul, twice: control (p + a few basics)
and treatment (control + six context blocks). Decision on body RMSE (NM-matched, y <= 1 h).

Run:  .venv/Scripts/python.exe tests/context_corrector_test.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from post import nm_lines as NL  # noqa: E402

RAW = ROOT / "data" / "raw"
HOLDOUT = ROOT / "cache" / "eval" / "v31_chain_holdout.parquet"
REFS = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]
CELL_MIN, CELL_CLIP = 30, 7200.0
PARAMS = dict(objective="regression", num_leaves=63, learning_rate=0.05, min_data_in_leaf=200,
              feature_fraction=0.9, lambda_l2=10.0, seed=42, deterministic=True, force_row_wise=True,
              verbose=-1, max_bin=127)
TRIM = 18000


def ep(c: str) -> pl.Expr:
    return pl.col(c).dt.epoch("ms") / 1000.0


def movements(src) -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = pl.scan_parquet(src)
    dep = (lf.filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), apt=pl.col("ADEP_mvt"), rwy=pl.col("RUNWAY_mvt").fill_null("?"),
        stand=pl.col("STAND_mvt").fill_null("?"), T=ep("MVT_TIME_UTC_mvt"),
        y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), m=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        nm=pl.col("FLIGHT_ID_mvt").is_null(),
        **{f"t_{c}": ep(c) for c in REFS},
        arvt1=ep("ARVT_1_flt"), arvt3=ep("ARVT_3_flt"),
        ades_eq=(pl.col("ADES_mvt") == pl.col("ADES_flt")).cast(pl.Float64),
        div=(pl.col("ADES_flt") != pl.col("ADES_FILED_flt")).cast(pl.Float64),
        type_eq=(pl.col("AIRCRAFT_TYPE_mvt") == pl.col("AIRCRAFT_TYPE_flt")).cast(pl.Float64)).collect())
    arr = (lf.filter(pl.col("PHASE_mvt") == "ARR").select(
        apt=pl.col("ADES_mvt"), L=ep("MVT_TIME_UTC_mvt"), B=ep("BLOCK_TIME_UTC_mvt")).collect())
    return dep, arr


def _cnt(sorted_arr: np.ndarray, q: np.ndarray, side="right") -> np.ndarray:
    return np.searchsorted(sorted_arr, q, side=side).astype(np.float64)


def time_blocks(dep: pl.DataFrame, arr: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Blocks 5 (retention at AOBT_3) and 6 (surface at T - p), per airport and runway."""
    n = dep.height
    out = {k: np.full(n, np.nan) for k in ("ret_overdue_apt", "ret_overdue_rwy", "ret_active_rwy", "ret_delay15",
                                           "sur_dep_taxi", "sur_arr_taxi", "sur_to_during", "sur_ld_during")}
    apt, rwy = dep["apt"].to_numpy(), dep["rwy"].to_numpy()
    T, A3, E1, S = (dep[c].to_numpy() for c in ("T", "t_AOBT_3_flt", "t_EOBT_1_flt", "t_SCHED_TIME_UTC_mvt"))
    o = T - p
    for a in np.unique(apt):
        ia = np.flatnonzero(apt == a)
        ar = arr.filter(pl.col("apt") == a)
        L = np.sort(ar["L"].drop_nulls().to_numpy())
        # arrivals taxiing in: only arrivals with both landing and in-block, so the two cumulative
        # counts cover the same set (a count over unequal sets drifts with time -- fixed 2026-10-04)
        lb = ar.filter(pl.col("L").is_not_null() & pl.col("B").is_not_null() & (pl.col("B") >= pl.col("L")))
        Lb, B = np.sort(lb["L"].to_numpy()), np.sort(lb["B"].to_numpy())
        Ts = np.sort(T[ia]); os_ = np.sort(o[ia][~np.isnan(o[ia])])
        e = np.sort(E1[ia][~np.isnan(E1[ia])]); a3 = np.sort(A3[ia][~np.isnan(A3[ia])])
        t = A3[ia]
        out["ret_overdue_apt"][ia] = (_cnt(e, t) - _cnt(e, t - 21600)) - (_cnt(a3, t) - _cnt(a3, t - 21600))
        # mean AOBT3 - SCHED of departures with AOBT3 in [t - 900, t)
        ok = ~np.isnan(A3[ia]) & ~np.isnan(S[ia])
        order = np.argsort(A3[ia][ok]); a3o = A3[ia][ok][order]; dlo = (A3[ia][ok] - S[ia][ok])[order]
        cs = np.concatenate([[0.0], np.cumsum(dlo)])
        lo, hi = np.searchsorted(a3o, t - 900, "left"), np.searchsorted(a3o, t, "left")
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ret_delay15"][ia] = np.where(hi > lo, (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1), np.nan)
        out["sur_dep_taxi"][ia] = _cnt(os_, o[ia]) - _cnt(Ts, o[ia])
        out["sur_arr_taxi"][ia] = _cnt(Lb, o[ia]) - _cnt(B, o[ia])
        out["sur_to_during"][ia] = _cnt(Ts, T[ia]) - _cnt(Ts, o[ia]) - 1
        out["sur_ld_during"][ia] = _cnt(L, T[ia]) - _cnt(L, o[ia])
        for r in np.unique(rwy[ia]):
            ir = ia[rwy[ia] == r]
            er = np.sort(E1[ir][~np.isnan(E1[ir])]); a3r = np.sort(A3[ir][~np.isnan(A3[ir])])
            tr = A3[ir]
            out["ret_overdue_rwy"][ir] = (_cnt(er, tr) - _cnt(er, tr - 21600)) - (_cnt(a3r, tr) - _cnt(a3r, tr - 21600))
            # active on the runway (AOBT_3 <= t < T): both counts over departures that have an AOBT_3 at
            # or before their takeoff (unequal sets drifted with time -- fixed 2026-10-04)
            both = ~np.isnan(A3[ir]) & (A3[ir] <= T[ir])
            a3b, Tb = np.sort(A3[ir][both]), np.sort(T[ir][both])
            out["ret_active_rwy"][ir] = _cnt(a3b, tr) - _cnt(Tb, tr)
    for k in ("ret_overdue_apt", "ret_overdue_rwy", "ret_active_rwy", "ret_delay15"):
        out[k][np.isnan(A3)] = np.nan
    return pl.DataFrame(out)


def cell_table(train: pl.DataFrame) -> list[tuple[list[str], pl.DataFrame]]:
    t = train.filter(pl.col("y").is_not_null()).with_columns(yc=pl.col("y").clip(upper_bound=CELL_CLIP))
    out = []
    for keys in (["apt", "stand", "rwy"], ["apt", "rwy"], ["apt"]):
        g = (t.group_by(keys).agg(cel_p50=pl.col("yc").median(), cel_p90=pl.col("yc").quantile(0.9),
                                  cel_sd=pl.col("yc").std(), cel_n=pl.len().cast(pl.Float64))
             .filter(pl.col("cel_n") >= CELL_MIN))
        out.append((keys, g))
    return out


def apply_cells(df: pl.DataFrame, tables) -> pl.DataFrame:
    cols = ["cel_p50", "cel_p90", "cel_sd", "cel_n"]
    res = df.select("MVT_ID_mvt", "apt", "stand", "rwy").with_columns([pl.lit(None, pl.Float64).alias(c) for c in cols])
    for keys, g in tables:
        j = res.join(g, on=keys, how="left", suffix="_g")
        res = j.with_columns([pl.coalesce(pl.col(c), pl.col(f"{c}_g")).alias(c) for c in cols]).select(res.columns)
    return res.select(cols)


def build(dep: pl.DataFrame, arr: pl.DataFrame, p: np.ndarray, cells) -> tuple[pl.DataFrame, list[str], list[str]]:
    T = dep["T"]
    g = {c: T - dep[f"t_{c}"] for c in REFS}
    base = pl.DataFrame({"p": p, "apt_c": dep["apt"].cast(pl.Categorical).to_physical().cast(pl.Float64),
                         "hour": ((dep["T"] % 86400) / 3600.0), "g_aobt3": g["AOBT_3_flt"], "g_sched": g["SCHED_TIME_UTC_mvt"]})
    ctrl = list(base.columns)
    prefix = dep["apt"] + "|" + dep["stand"].str.extract(r"^([A-Za-z]+)", 1).fill_null(dep["stand"].str.slice(0, 1))
    b1 = pl.DataFrame({"stand_pfx": prefix.cast(pl.Categorical).to_physical().cast(pl.Float64)})
    b2 = apply_cells(dep, cells).with_columns(cel_rel=pl.Series(p) - pl.col("cel_p50"))
    dur_p = dep["arvt1"] - dep["t_EOBT_1_flt"]; dur_r = dep["arvt3"] - dep["t_AOBT_3_flt"]
    b3 = pl.DataFrame({"nm_dur_plan": dur_p, "nm_dur_real": dur_r, "nm_dur_dif": dur_r - dur_p,
                       "nm_dur_ratio": dur_r / dur_p.replace(0, None), "nm_ades_eq": dep["ades_eq"],
                       "nm_div": dep["div"], "nm_type_eq": dep["type_eq"]})
    dist = {f"dist_{c.split('_')[0].lower()}": pl.Series(p) - g[c] for c in REFS}
    absd = pl.DataFrame(dist).select(pl.min_horizontal([pl.col(k).abs() for k in dist]).alias("dist_min"))
    b4 = pl.DataFrame(dist).with_columns(absd["dist_min"],
                                         n_plans=pl.sum_horizontal([g[c].is_not_null().cast(pl.Float64) for c in REFS]))
    b56 = time_blocks(dep, arr, p)
    X = pl.concat([base, b1, b2, b3, b4, b56], how="horizontal")
    return X, ctrl, ["apt_c", "stand_pfx"]


def fit_apply(X: np.ndarray, y: np.ndarray, fit: np.ndarray, inner: np.ndarray, app: np.ndarray, cats: list[int]):
    tr = lgb.Dataset(X[fit & ~inner], y[fit & ~inner], categorical_feature=cats, free_raw_data=False)
    va = lgb.Dataset(X[fit & inner], y[fit & inner], categorical_feature=cats, reference=tr, free_raw_data=False)
    m = lgb.train(PARAMS, tr, 3000, valid_sets=[va], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(m.best_iteration, 1)
    full = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit], categorical_feature=cats), best)
    return full.predict(X[app]), best


PARAMS_APT = {**PARAMS, "min_data_in_leaf": 100}
MIN_APT_ROWS = 20000


def fit_apply_ens(X, y, fit, inner, app, cats, apt):
    """Equal-weight mean of a global LightGBM and one LightGBM per airport (airports with fewer
    than MIN_APT_ROWS training rows use the global one) -- reports/ctx_ensemble_preregistration.md."""
    c_glob, best = fit_apply(X, y, fit, inner, app, cats)
    c_apt = c_glob.copy()
    apt_app = apt[app]
    for a in np.unique(apt):
        k = fit & (apt == a)
        if k.sum() < MIN_APT_ROWS:
            continue
        tr = lgb.Dataset(X[k & ~inner], y[k & ~inner], categorical_feature=cats, free_raw_data=False)
        va = lgb.Dataset(X[k & inner], y[k & inner], categorical_feature=cats, reference=tr, free_raw_data=False)
        mm = lgb.train(PARAMS_APT, tr, 3000, valid_sets=[va], callbacks=[lgb.early_stopping(100, verbose=False)])
        full = lgb.train(PARAMS_APT, lgb.Dataset(X[k], y[k], categorical_feature=cats), max(mm.best_iteration, 1))
        sel = apt_app == a
        if sel.any():
            c_apt[sel] = full.predict(X[app][sel])
    return (c_glob + c_apt) / 2, best


def main() -> None:
    dep_all, arr_all = movements(str(RAW / "training_*.parquet"))
    cells = cell_table(dep_all.filter(~pl.col("m").is_in([1, 7])))
    raw = NL.frame(str(RAW / "training_*.parquet"))
    prm = NL.fit(raw.filter(~pl.col("m").is_in([1, 7])))
    h = (pl.read_parquet(HOLDOUT).select("MVT_ID_mvt", "pred_v31").join(raw, on="MVT_ID_mvt"))
    lt = h["lt"].to_numpy(); p = h["pred_v31"].to_numpy()
    win = lambda q: np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))
    p, line_rows, rome = NL.apply(h, win(p), prm)
    line_rows = line_rows | rome
    dep = h.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="left")
    assert dep.height == h.height and dep["apt"].null_count() == 0
    arr = arr_all.filter(pl.col("L").is_not_null())
    X, ctrl, catn = build(dep, arr, p, cells)
    print(f"rows {X.height:,}, treatment features {X.width}, control {len(ctrl)}")
    print("null share:", {c: round(X[c].null_count() / X.height, 3) for c in X.columns if X[c].null_count()})
    y = dep["y"].to_numpy(); m = dep["m"].to_numpy(); nm = dep["nm"].to_numpy(); apt = dep["apt"].to_numpy()
    day = (dep["T"].to_numpy() // 86400).astype(int)
    trainable = (~nm) & (y <= TRIM) & ~line_rows
    inner = (day % 5 == 0)
    res = {}
    for arm, cols in (("control", ctrl), ("treatment", X.columns)):
        Xa = X.select(cols).to_numpy().astype(np.float64)
        cats = [cols.index(c) for c in catn if c in cols]
        q = p.copy()
        for fm, am in ((1, 7), (7, 1)):
            app = (m == am) & ~line_rows
            c, best = fit_apply(Xa, y - p, trainable & (m == fm), inner, app, cats)
            q[app] = p[app] + c
            print(f"  {arm} fit month {fm}: best_iter {best}", flush=True)
        res[arm] = win(np.clip(q, 0, None))
    body = (~nm) & (y <= 3600)
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))
    print(f"\n{'slice':22s} {'input p':>9s} {'control':>9s} {'treatment':>10s} {'t - c':>7s}")
    rows = [("body pooled", body), ("body Jan", body & (m == 1)), ("body Jul", body & (m == 7)),
            ("trimmed", y <= TRIM), ("full", np.ones_like(body))]
    rows += [(f"  body {a}", body & (apt == a)) for a in sorted(set(apt))]
    for name, k in rows:
        print(f"{name:22s} {r(p, k):9.2f} {r(res['control'], k):9.2f} {r(res['treatment'], k):10.2f} "
              f"{r(res['treatment'], k) - r(res['control'], k):+7.2f}")
    g = r(res["control"], body) - r(res["treatment"], body)
    gj = r(res["control"], body & (m == 1)) - r(res["treatment"], body & (m == 1))
    gl = r(res["control"], body & (m == 7)) - r(res["treatment"], body & (m == 7))
    ok = g >= 1.5 and gj > 0 and gl > 0 and r(res["treatment"], y <= TRIM) <= r(res["control"], y <= TRIM)
    print(f"\nDECISION: body gain vs control {g:.2f} (Jan {gj:.2f}, Jul {gl:.2f}) -> {'ADOPT' if ok else 'REJECT'}")
    pl.DataFrame({"MVT_ID_mvt": h["MVT_ID_mvt"], "p": p, "control": res["control"], "treatment": res["treatment"]}).write_parquet(
        ROOT / "cache" / "eval" / "context_corrector_holdout.parquet")


if __name__ == "__main__":
    main()
