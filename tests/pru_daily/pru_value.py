"""Holdout value of EUROCONTROL's daily all-cause pre-departure delay sum (analysis only, 2025 data).
Published per airport x off-block day: D = sum max(0, AOBT - SOBT) [min].  For each airport-day (day assigned by the
PREDICTED off-block T - p, as it would be in 2026) find the constant shift delta of every prediction that matches the
published sum, apply it, and score.  Also the oracle ceiling (remove each airport-day's mean residual)."""
import os, glob, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from scipy.optimize import brentq
SCR = os.environ["SCR"]
chain = sys.argv[1] if len(sys.argv) > 1 else f"{SCR}/inc/chain_ctl.parquet"
files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
raw = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
p = pl.read_parquet(chain).select(pl.col("MVT_ID_mvt").cast(pl.Int64), p="pred")
d = (raw.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(p, on="MVT_ID_mvt", how="inner")
     .filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
     .with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), apt=pl.col("ADEP_mvt"),
                   ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                   day_true=pl.col("BLOCK_TIME_UTC_mvt").dt.date(),
                   nm=pl.col("FLIGHT_ID_mvt").is_not_null()))
d = d.with_columns(day_hat=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
pub = pl.read_csv(f"{SCR}/ext/all_pre_departure_delays_2025.csv", infer_schema_length=0).select(
    apt="APT_ICAO", day=pl.col("FLT_DATE").str.to_date(), D=pl.col("DLY_ALL_PRE_2").cast(pl.Float64))
y, pr, ts = d["y"].to_numpy(), d["p"].to_numpy(), d["ts"].to_numpy()
r0 = y - pr
print(f"rows {len(y):,}; RMSE {np.sqrt(np.mean(r0**2)):.3f}")
# oracle: remove each airport x true-day mean residual
key_t = (d["apt"] + "_" + d["day_true"].cast(pl.Utf8)).to_numpy()
_, inv = np.unique(key_t, return_inverse=True)
mean_r = np.bincount(inv, r0) / np.bincount(inv)
print(f"oracle airport-day mean removal: RMSE {np.sqrt(np.mean((r0 - mean_r[inv])**2)):.3f}")
# realistic: per airport x predicted-day, shift delta so that sum max(0, ts - (p+delta)) / 60 == D
dd = d.with_row_index("i").join(pub, left_on=["apt", "day_hat"], right_on=["apt", "day"], how="left")
idx = dd["i"].to_numpy(); Dv = dd["D"].to_numpy()
key_h = (dd["apt"] + "_" + dd["day_hat"].cast(pl.Utf8)).to_numpy()
newp = pr.copy(); deltas = []
apt_arr = d["apt"].to_numpy()
for k in np.unique(key_h):
    m = idx[key_h == k]
    Dk = Dv[key_h == k][0]
    if not np.isfinite(Dk):
        continue
    f = lambda dl: np.maximum(0.0, ts[m] - (pr[m] + dl)).sum() / 60.0 - Dk
    lo, hi = -20000.0, 20000.0
    if f(lo) * f(hi) > 0:
        continue
    dl = brentq(f, lo, hi, xtol=0.01)
    newp[m] = np.maximum(pr[m] + dl, 0.0)
    deltas.append((k, len(m), dl))
r1 = y - newp
print(f"published-sum shift (all airports): RMSE {np.sqrt(np.mean(r1**2)):.3f}")
dl_arr = np.array([x[2] for x in deltas]); n_arr = np.array([x[1] for x in deltas])
print(f"  delta: median |delta| {np.median(np.abs(dl_arr)):.1f}s, p90 {np.percentile(np.abs(dl_arr), 90):.1f}s, max {np.abs(dl_arr).max():.0f}s")
for a in sorted(set(apt_arr)):
    k = apt_arr == a
    print(f"  {a}: {np.sqrt(np.mean(r0[k]**2)):8.2f} -> {np.sqrt(np.mean(r1[k]**2)):8.2f}   oracle {np.sqrt(np.mean((r0[k]-mean_r[inv][k])**2)):8.2f}")
body = (d["nm"].to_numpy()) & (y <= 3600)
print(f"body: {np.sqrt(np.mean(r0[body]**2)):.3f} -> {np.sqrt(np.mean(r1[body]**2)):.3f}")
