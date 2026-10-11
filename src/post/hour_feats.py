"""Within-day (hour-window) inputs for the day-shift allocation G3 (reports/day_alloc_g3_preregistration.md).

Label-free: movement data plus the input prediction p of the other departures, so the same code serves 2025, the
Jan/Jul 2026 board and the four-month blind board.

  dep: MVT_ID_mvt, apt, RUNWAY_mvt, t, lo, a3, s (epoch seconds: takeoff, LOBT, AOBT_3, SOBT), p
  arr: apt, RUNWAY_mvt, t (landing, epoch s), tin (taxi-in seconds)
  tin_med: apt, RUNWAY_mvt, med (2025 airport x runway median taxi-in)
"""
import numpy as np
import polars as pl

W = 1800.0          # hour window: +-30 min around the takeoff
BK = 3 * 3600.0     # backlog look-back
COLS = ["h_rl", "h_ra", "h_rs", "h_p", "h_n", "rwy30", "rwy15", "bk_lobt", "bk_sobt", "h_tinx", "rl", "ra"]
HOUR_COLS = [c for c in COLS if c not in ("rl", "ra")]


def _win_mean(t, v, w, min_n=3):
    """Mean of v over the other rows with t_j in [t_i - w, t_i + w] (t sorted); NaN v are skipped."""
    ok = np.isfinite(v)
    v0 = np.where(ok, v, 0.0)
    cs, cn = np.concatenate([[0.0], np.cumsum(v0)]), np.concatenate([[0], np.cumsum(ok)])
    lo, hi = np.searchsorted(t, t - w, "left"), np.searchsorted(t, t + w, "right")
    s, n = cs[hi] - cs[lo] - v0, cn[hi] - cn[lo] - ok
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), (hi - lo - 1).astype(float)


def tin_medians(arr: pl.DataFrame) -> pl.DataFrame:
    return arr.filter(pl.col("tin").is_between(0, 7200)).group_by("apt", "RUNWAY_mvt").agg(med=pl.col("tin").median())


def build(dep: pl.DataFrame, arr: pl.DataFrame, tin_med: pl.DataFrame) -> pl.DataFrame:
    dep = dep.with_columns(rl=pl.col("t") - pl.col("lo") - pl.col("p"), ra=pl.col("t") - pl.col("a3") - pl.col("p"),
                           rs=pl.col("t") - pl.col("s") - pl.col("p"))
    amed = tin_med.group_by("apt").agg(amed=pl.col("med").median())
    arr = (arr.filter(pl.col("tin").is_between(0, 7200)).join(tin_med, on=["apt", "RUNWAY_mvt"], how="left")
           .join(amed, on="apt", how="left").with_columns(tinx=pl.col("tin") - pl.col("med").fill_null(pl.col("amed"))))
    out = []
    for a in dep["apt"].unique().sort().to_list():
        d = dep.filter(pl.col("apt") == a).sort("t")
        t = d["t"].to_numpy().astype(float)
        n = len(t)
        o = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        for c, nm in [("rl", "h_rl"), ("ra", "h_ra"), ("rs", "h_rs"), ("p", "h_p")]:
            o[nm], cnt = _win_mean(t, d[c].to_numpy().astype(float), W)
        o["h_n"] = cnt
        rw = d["RUNWAY_mvt"].fill_null("?").to_numpy().astype(str)
        c30, c15 = np.zeros(n), np.zeros(n)
        for r_ in np.unique(rw):
            k = np.flatnonzero(rw == r_)
            tt = t[k]
            c30[k] = np.searchsorted(tt, tt, "left") - np.searchsorted(tt, tt - 1800.0, "left")
            c15[k] = np.searchsorted(tt, tt + 900.0, "right") - np.searchsorted(tt, tt, "right")
        o["rwy30"], o["rwy15"] = c30, c15
        took = np.searchsorted(t, t, "right") - np.searchsorted(t, t - BK, "left")
        for c, nm in [("lo", "bk_lobt"), ("s", "bk_sobt")]:
            b = d[c].to_numpy().astype(float)
            ob = np.sort(b[np.isfinite(b)])
            o[nm] = (np.searchsorted(ob, t, "right") - np.searchsorted(ob, t - BK, "left") - took).astype(float)
        ar = arr.filter(pl.col("apt") == a).sort("t")
        at, ax = ar["t"].to_numpy().astype(float), ar["tinx"].to_numpy().astype(float)
        cs = np.concatenate([[0.0], np.cumsum(ax)])
        lo_i, hi_i = np.searchsorted(at, t - W, "left"), np.searchsorted(at, t + W, "right")
        o["h_tinx"] = np.where(hi_i - lo_i >= 3, (cs[hi_i] - cs[lo_i]) / np.maximum(hi_i - lo_i, 1), np.nan)
        o["rl"], o["ra"] = d["rl"].to_numpy().astype(float), d["ra"].to_numpy().astype(float)
        out.append(pl.DataFrame(o))
    return pl.concat(out).with_columns(pl.col(COLS).cast(pl.Float64).fill_nan(None))


def movements(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(departures without p, arrivals) from a raw movement frame (training file(s) or a ranking file)."""
    ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)  # noqa: E731
    dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), apt=pl.col("ADEP_mvt"), RUNWAY_mvt=pl.col("RUNWAY_mvt"),
        t=ep("MVT_TIME_UTC_mvt"), lo=ep("LOBT_flt"), a3=ep("AOBT_3_flt"), s=ep("SCHED_TIME_UTC_mvt"))
    arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select(
        apt=pl.col("ADES_mvt"), RUNWAY_mvt=pl.col("RUNWAY_mvt"), t=ep("MVT_TIME_UTC_mvt"),
        tin=ep("BLOCK_TIME_UTC_mvt") - ep("MVT_TIME_UTC_mvt"))
    return dep, arr


RAW_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "LOBT_flt", "AOBT_3_flt"]
