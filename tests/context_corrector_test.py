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
import scope  # noqa: E402  (PRC_SCOPE=final4m: final_ranking rows, final4m caches; PROGRESS.md §132)

RAW = ROOT / "data" / "raw"
RANKING = scope.RANKING
HOLDOUT = ROOT / "cache" / "eval" / "v31_chain_holdout.parquet"
REFS = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]
CELL_MIN, CELL_CLIP = 30, 7200.0
PARAMS = dict(objective="regression", num_leaves=63, learning_rate=0.05, min_data_in_leaf=200,
              feature_fraction=0.9, lambda_l2=10.0, seed=42, deterministic=True, force_row_wise=True,
              verbose=-1, max_bin=127)
TRIM = 18000
RWY = os.environ.get("CTX_RWY") == "1"    # block 7, reports/ctx_runway_preregistration.md
AIR = os.environ.get("CTX_AIR") == "1"    # block 8, reports/ctx_airline_preregistration.md
ADSB = os.environ.get("CTX_ADSB") == "1"  # block 9, reports/ctx_adsb_preregistration.md
RO = os.environ.get("CTX_RO") == "1"      # block 10, stand re-occupation (PROGRESS.md §123; src/features/reocc_feats.py)
HOUR = os.environ.get("CTX_HOUR") == "1"  # block 11, hour-window state (reports/ctx_hour_preregistration.md; PROGRESS.md §134)
HOUR2 = os.environ.get("CTX_HOUR2") == "1"  # block 12, multi-scale / directional windows + ADS-B window (reports/ctx_hour2_preregistration.md)
RO_FILE = os.environ.get("CTX_RO_FILE", str(scope.out_dir(ROOT / "cache" / "v50") / "reocc_feats3.parquet"))
RO_COLS = ["ro_gap_last", "ro_tm_last", "ro_om_last", "ro_two_dep", "ro_tm_next", "ro_gap_prev", "ro_tm_prev",
           "ro_n_since_s", "ro_first_since_s", "ro_last_tin_x", "ro_last_land_gap"]
DET4 = ROOT / "cache" / "adsb_pushback_v4"
AB_TIERS = ["none", "appear", "dwell", "pass", "fb_dwell", "fb_appear"]
AB_DROP = [c for c in os.environ.get("CTX_ADSB_DROP", "").split(",") if c]   # shift-check drops (pre-registration rule)
_DET: dict = {}
FEATS = ROOT / "cache" / "features_nb"
AIR_MIN = 30
# months kept out of the airline table (gates: "1,7" so holdout labels never enter it, as for the cells);
# production leaves it empty and uses all 12 months
AIR_EXCL = [int(x) for x in os.environ.get("CTX_AIR_EXCL", "").split(",") if x]
_AIR_TAB: dict = {}                        # airline table from the labelled frame, reused for the ranking frame


_VOCAB: dict = {}


def _prefix(apt: pl.Series, stand: pl.Series) -> pl.Series:
    return apt + "|" + stand.str.extract(r"^([A-Za-z]+)", 1).fill_null(stand.str.slice(0, 1))


def code(values: pl.Series, key: str) -> pl.Series:
    """Stable integer codes from a sorted vocabulary over all 2025 + 2026 departures. Categorical
    physical codes follow order of appearance, so the training and ranking frames disagreed
    (stand prefixes: 17 of 140 matched; found 2026-10-05). Unseen values -> null."""
    if key not in _VOCAB:
        d = pl.concat([pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
                       .select(apt=pl.col("ADEP_mvt"), stand=pl.col("STAND_mvt").fill_null("?"),
                               op=pl.col("AIRCRAFT_OPERATOR_flt").fill_null("?")).collect()
                       for src in (str(RAW / "training_*.parquet"), str(RANKING))])
        v = {"apt": d["apt"], "stand_pfx": _prefix(d["apt"], d["stand"]), "op": d["op"]}[key]
        _VOCAB[key] = {x: i for i, x in enumerate(sorted(v.unique().to_list()))}
    return values.replace_strict(_VOCAB[key], default=None, return_dtype=pl.Int64).cast(pl.Float64)


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
        type_eq=(pl.col("AIRCRAFT_TYPE_mvt") == pl.col("AIRCRAFT_TYPE_flt")).cast(pl.Float64),
        heavy=pl.col("WK_TBL_CAT_flt").is_in(["H", "J"]).fill_null(False),
        op=pl.col("AIRCRAFT_OPERATOR_flt").fill_null("?")).collect())
    arr = (lf.filter(pl.col("PHASE_mvt") == "ARR").select(
        apt=pl.col("ADES_mvt"), L=ep("MVT_TIME_UTC_mvt"), B=ep("BLOCK_TIME_UTC_mvt"),
        rwy=pl.col("RUNWAY_mvt").fill_null("?")).collect())
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


def runway_block(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Block 7: runway state at takeoff T from takeoff/landing times, runways, AOBT_3 and wake
    category only (reports/ctx_runway_preregistration.md)."""
    n = dep.height
    out = {k: np.full(n, np.nan) for k in ("rw_cad20", "rw_cad20_apt", "rw_gap_prev", "rw_gap_next", "rw_share60",
                                           "rw_arr30", "rw_dom_change", "rw_heavy_q")}
    apt, rwy = dep["apt"].to_numpy(), dep["rwy"].to_numpy()
    T, A3, hv = dep["T"].to_numpy(), dep["t_AOBT_3_flt"].to_numpy(), dep["heavy"].to_numpy()
    for a in np.unique(apt):
        ia = np.flatnonzero(apt == a)
        Ts = np.sort(T[ia])
        k = np.searchsorted(Ts, T[ia], "right") - 1                     # index of this takeoff (last tie)
        out["rw_cad20_apt"][ia] = np.where(k >= 20, (Ts[k] - Ts[np.maximum(k - 20, 0)]) / 20.0, np.nan)
        n60 = _cnt(Ts, T[ia]) - _cnt(Ts, T[ia] - 3600)
        ar = arr.filter(pl.col("apt") == a)
        # dominant departure runway per 30-min bin of this airport
        b = (T[ia] // 1800).astype(np.int64)
        bins = pl.DataFrame({"b": b, "r": rwy[ia]}).group_by("b", "r").agg(c=pl.len()).sort("c", descending=True).unique("b", keep="first")
        dom = dict(zip(bins["b"].to_list(), bins["r"].to_list()))
        out["rw_dom_change"][ia] = [float(dom.get(x) != dom.get(x - 2)) if dom.get(x - 2) is not None else np.nan for x in b]
        for r in np.unique(rwy[ia]):
            ir = ia[rwy[ia] == r]
            Tr = np.sort(T[ir])
            kr = np.searchsorted(Tr, T[ir], "right") - 1
            out["rw_cad20"][ir] = np.where(kr >= 20, (Tr[kr] - Tr[np.maximum(kr - 20, 0)]) / 20.0, np.nan)
            lo = np.searchsorted(Tr, T[ir], "left")                     # first takeoff at this time
            out["rw_gap_prev"][ir] = np.where(lo >= 1, T[ir] - Tr[np.maximum(lo - 1, 0)], np.nan)
            out["rw_gap_next"][ir] = np.where(kr + 1 < len(Tr), Tr[np.minimum(kr + 1, len(Tr) - 1)] - T[ir], np.nan)
            out["rw_share60"][ir] = (_cnt(Tr, T[ir]) - _cnt(Tr, T[ir] - 3600)) / np.maximum(n60[np.searchsorted(ia, ir)], 1)
            Lr = np.sort(ar.filter(pl.col("rwy") == r)["L"].drop_nulls().to_numpy())
            out["rw_arr30"][ir] = _cnt(Lr, T[ir]) - _cnt(Lr, T[ir] - 1800)
            # heavy departures queued on the runway: AOBT_3 <= t < takeoff, both counts over the same set
            both = hv[ir] & ~np.isnan(A3[ir]) & (A3[ir] <= T[ir])
            a3h, Th = np.sort(A3[ir][both]), np.sort(T[ir][both])
            out["rw_heavy_q"][ir] = _cnt(a3h, T[ir], "left") - _cnt(Th, T[ir], "left")
    # the ranking frame jumps from 31 Jan to 1 Jul: gaps across that seam (and dead runways) are not state
    for k, cap in (("rw_gap_prev", 21600), ("rw_gap_next", 21600), ("rw_cad20", 3600), ("rw_cad20_apt", 3600)):
        out[k][out[k] > cap] = np.nan
    return pl.DataFrame(out)


def airline_block(dep: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Block 8: airline statistics of the stack residual and the copy pattern, leave-one-month-out, plus the
    stand-rotation columns of the base feature cache (reports/ctx_airline_preregistration.md)."""
    d = dep.select("MVT_ID_mvt", "apt", "op", "m", "y", "nm", "T", "t_SCHED_TIME_UTC_mvt").with_columns(
        r=(pl.col("y") - pl.Series(p)).clip(-3600, 3600),
        cp=((pl.col("y") - (pl.col("T") - pl.col("t_SCHED_TIME_UTC_mvt"))).abs() <= 60).cast(pl.Float64))
    labelled = d["y"].null_count() < d.height
    if labelled:
        src = d.filter(pl.col("y").is_not_null() & (pl.col("y") <= TRIM) & ~pl.col("nm") & ~pl.col("m").is_in(AIR_EXCL))
        g = src.group_by("apt", "op", "m").agg(sr=pl.col("r").sum(), sc=pl.col("cp").sum(), n=pl.len())
        _AIR_TAB["g"] = g
    g = _AIR_TAB["g"]
    tot = g.group_by("apt", "op").agg(Sr=pl.col("sr").sum(), Sc=pl.col("sc").sum(), N=pl.col("n").sum())
    x = d.join(tot, on=["apt", "op"], how="left")
    if labelled:   # remove the row's own month
        x = x.join(g, on=["apt", "op", "m"], how="left").with_columns(
            Sr=pl.col("Sr") - pl.col("sr").fill_null(0), Sc=pl.col("Sc") - pl.col("sc").fill_null(0),
            N=pl.col("N") - pl.col("n").fill_null(0))
    ok = pl.col("N") >= AIR_MIN
    out = x.select(al_res=pl.when(ok).then(pl.col("Sr") / pl.col("N")), al_copy=pl.when(ok).then(pl.col("Sc") / pl.col("N")),
                   al_n=pl.col("N").cast(pl.Float64),
                   al_op=code(x["op"], "op"))
    rot = (pl.concat([pl.scan_parquet(f) .select("MVT_ID_mvt", "inbound_arr_delay", "sched_ground", "actual_ground", "link_confidence")
                      for f in (FEATS / "train2025.parquet", scope.out_dir(FEATS) / "ranking.parquet")], how="vertical_relaxed")
           .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).unique("MVT_ID_mvt").collect())
    sec = lambda c: (pl.col(c).dt.total_seconds() if rot.schema[c] == pl.Duration else pl.col(c)).cast(pl.Float64)
    rot = rot.select("MVT_ID_mvt", ro_inb_delay=sec("inbound_arr_delay"), ro_sched_ground=sec("sched_ground"),
                     ro_actual_ground=sec("actual_ground"),
                     ro_link_conf=pl.col("link_confidence").cast(pl.Utf8).replace({"high": "2", "med": "1", "low": "0"}).cast(pl.Float64, strict=False))
    rr = dep.select("MVT_ID_mvt").join(rot, on="MVT_ID_mvt", how="left")
    assert rr.height == dep.height
    return pl.concat([out, rr.drop("MVT_ID_mvt")], how="horizontal")


def adsb_block(dep: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Block 9: ADS-B track state from the v4 detector (reports/ctx_adsb_preregistration.md). Rows of
    days that were never fetched have ab_avail = 0 and null track columns."""
    if "d" not in _DET:
        _DET["d"] = (pl.concat([pl.read_parquet(f) for f in scope.day_files(DET4)], how="diagonal")
                     .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).unique("MVT_ID_mvt"))
    x = dep.select("MVT_ID_mvt", "T").join(_DET["d"], on="MVT_ID_mvt", how="left")
    assert x.height == dep.height
    f = lambda c: pl.col(c).cast(pl.Float64)
    out = x.select(
        ab_avail=pl.col("airport").is_not_null().cast(pl.Float64),
        ab_cov=f("adsb_day_coverage"), ab_matched=f("adsb_matched"), ab_fallback=f("adsb_fallback"),
        ab_tier=pl.when(pl.col("airport").is_not_null()).then(
            pl.col("adsb_tier").fill_null("none").replace_strict({t: i for i, t in enumerate(AB_TIERS)}, default=0,
                                                                 return_dtype=pl.Float64)),
        ab_taxi=pl.col("T") - pl.col("adsb_pushback_ts"), ab_L=pl.col("T") - pl.col("adsb_first_ts"),
        ab_first_gs="adsb_first_gs", ab_first_own_m="adsb_first_own_m", ab_min_own_m="adsb_min_own_m",
        ab_n_pts=f("adsb_n_pts"), ab_takeoff_gap_s="adsb_takeoff_gap_s",
        ab_pb_gap_s="adsb_pb_gap_s", ab_pb_dist_m="adsb_pb_dist_m", ab_pb_gs="adsb_pb_gs",
    ).with_columns(ab_taxi_rel=pl.col("ab_taxi") - pl.Series(p), ab_L_rel=pl.col("ab_L") - pl.Series(p))
    return out.drop(AB_DROP)


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


def reocc_block(dep: pl.DataFrame) -> pl.DataFrame:
    """Label-free stand re-occupation features (arrival in-blocks, departure takeoffs, stands, types), row order of dep."""
    ro = pl.read_parquet(RO_FILE).select(pl.col("MVT_ID_mvt").cast(pl.Int64), *RO_COLS)
    k = dep.select(pl.col("MVT_ID_mvt").cast(pl.Int64)).with_row_index("_i")
    out = k.join(ro, on="MVT_ID_mvt", how="left").sort("_i")
    assert out.height == dep.height
    return out.select([pl.col(c).cast(pl.Float64) for c in RO_COLS])


_TINMED: dict = {}


def _win(t, v, w_lo, w_hi, min_n=3):
    """Mean of v over the other rows with t_j in [t_i - w_lo, t_i + w_hi] (t sorted); NaN v skipped; count of used rows."""
    ok = np.isfinite(v)
    v0 = np.where(ok, v, 0.0)
    cs, cn = np.concatenate([[0.0], np.cumsum(v0)]), np.concatenate([[0], np.cumsum(ok)])
    lo, hi = np.searchsorted(t, t - w_lo, "left"), np.searchsorted(t, t + w_hi, "right")
    s_, n = cs[hi] - cs[lo] - v0, cn[hi] - cn[lo] - ok
    return np.where(n >= min_n, s_ / np.maximum(n, 1), np.nan), n.astype(float)


def hour2_block(dep: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Block 12: other departures at the same airport -- (T - LOBT - p) and (T - AOBT_3 - p) over +-15 / +-60 min and
    over the 30 min before / after T; and the ADS-B taxi of tracked neighbours (tiers appear, dwell; v4 detector)
    minus their p, clipped to +-600 s, over +-30 min, one mean per tier. Label-free; p = the frame's input prediction."""
    if "d" not in _DET:
        _DET["d"] = (pl.concat([pl.read_parquet(f) for f in scope.day_files(DET4)], how="diagonal")
                     .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).unique("MVT_ID_mvt"))
    x = (dep.select("MVT_ID_mvt", "apt", "T", "t_LOBT_flt", "t_AOBT_3_flt").with_columns(p=pl.Series(np.asarray(p, dtype=np.float64)))
         .with_row_index("_i").join(_DET["d"].select("MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts"), on="MVT_ID_mvt", how="left")
         .sort("apt", "T"))
    assert x.height == dep.height
    t = x["T"].cast(pl.Float64).to_numpy()
    q = x["p"].to_numpy()
    rl = t - x["t_LOBT_flt"].cast(pl.Float64).fill_null(np.nan).to_numpy() - q
    ra = t - x["t_AOBT_3_flt"].cast(pl.Float64).fill_null(np.nan).to_numpy() - q
    tier = x["adsb_tier"].fill_null("").to_numpy()
    ab = np.clip(t - x["adsb_pushback_ts"].cast(pl.Float64).fill_null(np.nan).to_numpy() - q, -600, 600)
    apt = x["apt"].to_numpy()
    cols = ["h2_rl15", "h2_rl60", "h2_ra15", "h2_ra60", "h2_rl_pre", "h2_rl_post", "h2_ra_pre", "h2_ra_post",
            "h2_ab_app", "h2_ab_dw", "h2_ab_n"]
    o = {c: np.full(len(t), np.nan) for c in cols}
    for a in np.unique(apt):
        k = np.flatnonzero(apt == a)
        tt = t[k]
        for nm, v in (("rl", rl[k]), ("ra", ra[k])):
            o[f"h2_{nm}15"][k], _ = _win(tt, v, 900, 900)
            o[f"h2_{nm}60"][k], _ = _win(tt, v, 3600, 3600)
            o[f"h2_{nm}_pre"][k], _ = _win(tt, v, 1800, 0)
            o[f"h2_{nm}_post"][k], _ = _win(tt, v, 0, 1800)
        o["h2_ab_app"][k], na = _win(tt, np.where(tier[k] == "appear", ab[k], np.nan), 1800, 1800, min_n=1)
        o["h2_ab_dw"][k], nd = _win(tt, np.where(tier[k] == "dwell", ab[k], np.nan), 1800, 1800, min_n=1)
        o["h2_ab_n"][k] = na + nd
    out = pl.DataFrame({"_i": x["_i"].to_numpy(), **o}).sort("_i")
    return out.select([pl.col(c).cast(pl.Float64).fill_nan(None) for c in cols])


def hour_block(dep: pl.DataFrame, arr: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Block 11: hour-window state from the other departures within +-30 min (their takeoff minus LOBT / AOBT_3 /
    SCHED minus their input prediction p), same-runway counts, LOBT / SCHED backlog, arrivals' taxi-in excess
    (src/post/hour_feats.py). Label-free; the frame's own p is the neighbours' prediction."""
    from post import hour_feats as HF
    if "t" not in _TINMED:   # 2025 airport x runway median taxi-in, the same table for every frame
        _, a25 = movements(str(RAW / "training_*.parquet"))
        _TINMED["t"] = HF.tin_medians(a25.select(apt="apt", RUNWAY_mvt="rwy", tin=pl.col("B") - pl.col("L")))
    d = pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], "apt": dep["apt"], "RUNWAY_mvt": dep["rwy"], "t": dep["T"],
                      "lo": dep["t_LOBT_flt"], "a3": dep["t_AOBT_3_flt"], "s": dep["t_SCHED_TIME_UTC_mvt"],
                      "p": np.asarray(p, dtype=np.float64)})
    a = arr.select(apt="apt", RUNWAY_mvt="rwy", t="L", tin=pl.col("B") - pl.col("L"))
    f = HF.build(d, a, _TINMED["t"])
    out = d.select("MVT_ID_mvt").with_row_index("_i").join(f, on="MVT_ID_mvt", how="left").sort("_i")
    assert out.height == dep.height
    return out.select([pl.col(c).cast(pl.Float64) for c in HF.HOUR_COLS])


def build(dep: pl.DataFrame, arr: pl.DataFrame, p: np.ndarray, cells) -> tuple[pl.DataFrame, list[str], list[str]]:
    T = dep["T"]
    g = {c: T - dep[f"t_{c}"] for c in REFS}
    base = pl.DataFrame({"p": p, "apt_c": code(dep["apt"], "apt"),
                         "hour": ((dep["T"] % 86400) / 3600.0), "g_aobt3": g["AOBT_3_flt"], "g_sched": g["SCHED_TIME_UTC_mvt"]})
    ctrl = list(base.columns)
    b1 = pl.DataFrame({"stand_pfx": code(_prefix(dep["apt"], dep["stand"]), "stand_pfx")})
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
    parts = ([base, b1, b2, b3, b4, b56] + ([runway_block(dep, arr)] if RWY else []) + ([airline_block(dep, p)] if AIR else [])
             + ([adsb_block(dep, p)] if ADSB else []) + ([reocc_block(dep)] if RO else [])
             + ([hour_block(dep, arr, p)] if HOUR else []) + ([hour2_block(dep, p)] if HOUR2 else []))
    X = pl.concat(parts, how="horizontal")
    return X, ctrl, ["apt_c", "stand_pfx"] + (["al_op"] if AIR else [])


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
