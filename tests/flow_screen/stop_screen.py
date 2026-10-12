"""Runway stoppage / discharge inputs, screened on top of v66's corrector (ctxc1sel_rf). Label-free.
Per (airport, runway) and per airport (all departures; and arrivals' landings): the longest gap between consecutive
movements inside [T - 60 min, T], the time from that gap's end to T, and the number of movements since it ended."""
import glob, os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen import run
ROOT = "C:/Users/zhang/smart-jigsaw"; S = ROOT + "/cache/analysis/flow_screen"
W = 3600.0


def stop_feats(tq, tm):
    """tq: query times; tm: sorted movement times. For each query: max gap among consecutive movements in [t-W, t]
    (movements <= t; the window start counts as a boundary), time since that gap ended, movements since."""
    n = len(tq); mx = np.full(n, np.nan); since = np.full(n, np.nan); cnt = np.full(n, np.nan)
    lo = np.searchsorted(tm, tq - W, side="left"); hi = np.searchsorted(tm, tq, side="right")
    for i in range(n):
        seg = tm[lo[i]:hi[i]]
        if len(seg) == 0:
            continue
        pts = np.concatenate([[tq[i] - W], seg])
        g = np.diff(pts); j = int(np.argmax(g))
        mx[i] = g[j]; since[i] = tq[i] - pts[j + 1]; cnt[i] = len(seg) - 1 - j
    return mx, since, cnt


f = pl.read_parquet(S + "/dep25_air_rf.parquet")
raw = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "RUNWAY_mvt"]) for p in sorted(glob.glob(ROOT + "/data/raw/training_2025-*.parquet"))])
f = f.drop("rwy").join(raw.select(pl.col("MVT_ID_mvt").cast(pl.Int64), rwy=pl.col("RUNWAY_mvt").fill_null("?")), on="MVT_ID_mvt", how="left")
arr = pl.read_parquet(S + "/arr25.parquet")
out = {c: np.full(f.height, np.nan) for c in ["st_rw_max", "st_rw_since", "st_rw_cnt", "st_ap_max", "st_ap_since", "st_ap_cnt",
                                              "st_ar_max", "st_ar_since", "st_ar_cnt"]}
T = f["T"].to_numpy(); apt = f["apt"].to_numpy(); rw = f["rwy"].to_numpy().astype(str)
for a in np.unique(apt):
    ka = np.flatnonzero(apt == a)
    tm = np.sort(T[ka])
    out["st_ap_max"][ka], out["st_ap_since"][ka], out["st_ap_cnt"][ka] = stop_feats(T[ka], tm)
    ta = np.sort(arr.filter(pl.col("apt") == a)["T"].drop_nulls().to_numpy())
    out["st_ar_max"][ka], out["st_ar_since"][ka], out["st_ar_cnt"][ka] = stop_feats(T[ka], ta)
    for r in np.unique(rw[ka]):
        k = ka[rw[ka] == r]
        out["st_rw_max"][k], out["st_rw_since"][k], out["st_rw_cnt"][k] = stop_feats(T[k], np.sort(T[k]))
    print(a, flush=True)
f = f.with_columns([pl.Series(c, v) for c, v in out.items()])
f.write_parquet(S + "/dep25_stop_rf.parquet")
run(f, list(out), label="runway stoppage / discharge (on rf)")
