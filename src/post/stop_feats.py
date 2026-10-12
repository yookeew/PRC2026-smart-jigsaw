"""Runway stoppage / discharge inputs, block 14 of the context corrector (reports/ctx_stop_preregistration.md).

Label-free: the frame's own takeoff and landing times only. For each departure at takeoff T, over [T - 60 min, T] (the
window start counts as a boundary): the longest gap between consecutive movements, the time from that gap's end to T,
and the movements since it ended -- for the same departure runway's takeoffs (st_rw_*), all the airport's takeoffs
(st_ap_*) and the airport's landings (st_ar_*).

  dep: MVT_ID_mvt, apt, rwy, t (takeoff, epoch s);  arr: apt, t (landing, epoch s)
"""
import numpy as np
import polars as pl

W = 3600.0
STOP_COLS = ["st_rw_max", "st_rw_since", "st_rw_cnt", "st_ap_max", "st_ap_since", "st_ap_cnt",
             "st_ar_max", "st_ar_since", "st_ar_cnt"]


def _stop(tq: np.ndarray, tm: np.ndarray):
    """tm sorted. Per query t: max gap among [t - W] + movements in [t - W, t], time since its end, movements since."""
    n = len(tq)
    mx, since, cnt = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    lo = np.searchsorted(tm, tq - W, side="left")
    hi = np.searchsorted(tm, tq, side="right")
    for i in range(n):
        seg = tm[lo[i]:hi[i]]
        if len(seg) == 0:
            continue
        pts = np.concatenate([[tq[i] - W], seg])
        g = np.diff(pts)
        j = int(np.argmax(g))
        mx[i], since[i], cnt[i] = g[j], tq[i] - pts[j + 1], len(seg) - 1 - j
    return mx, since, cnt


def build(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    T = dep["t"].to_numpy().astype(float)
    apt = dep["apt"].to_numpy().astype(str)
    rw = dep["rwy"].fill_null("?").to_numpy().astype(str)
    out = {c: np.full(dep.height, np.nan) for c in STOP_COLS}
    ok = ~np.isnan(T)
    for a in np.unique(apt):
        ka = np.flatnonzero((apt == a) & ok)
        if len(ka) == 0:
            continue
        out["st_ap_max"][ka], out["st_ap_since"][ka], out["st_ap_cnt"][ka] = _stop(T[ka], np.sort(T[ka]))
        ta = np.sort(arr.filter(pl.col("apt") == a)["t"].drop_nulls().to_numpy().astype(float))
        out["st_ar_max"][ka], out["st_ar_since"][ka], out["st_ar_cnt"][ka] = _stop(T[ka], ta)
        for r in np.unique(rw[ka]):
            k = ka[rw[ka] == r]
            out["st_rw_max"][k], out["st_rw_since"][k], out["st_rw_cnt"][k] = _stop(T[k], np.sort(T[k]))
    return pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], **out}).with_columns(pl.col(STOP_COLS).fill_nan(None))
