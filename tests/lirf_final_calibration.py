"""Is v42's FINAL prediction miscalibrated on LIRF NM-matched rows? (PROGRESS.md §105; kind-mango's v64 isotonic gate
recalibration gained -1.14 s live over all LIRF rows. NM-unmatched rows are out of scope: team rule §87.)

Holdout only (v42 chain on Jan+Jul 2025). Per cell = echo-probability band (base stack mix 0.577 lgb + 0.428 cat) x
offset band, LIRF NM-matched rows: n, mean (y - P) per month, and a cross-month test -- the per-cell mean residual fitted
on one month (shrunk: n / (n + 50)) is added to the other month's predictions; SSE change on the applied month.

Run:  .venv/Scripts/python.exe -u tests/lirf_final_calibration.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "cache" / "eval"
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)
P_BANDS = [0, .02, .05, .1, .2, .4, .7, 1.01]
OFF_BANDS = [-1e9, 1800, 3600, 3 * 3600, 1e12]
SHRINK = 50


def main() -> None:
    raw = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet"))
                     .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_not_null())
                     .select(I64, y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                             off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
                     for a, b in (("01", "02"), ("07", "08"))]).collect()
    p = (pl.read_parquet(EVAL / "lgb_mixed_nb_holdout_ev.parquet").select(I64, pl_="echo_prob")
         .join(pl.read_parquet(EVAL / "cat_mixed_holdout_ev.parquet").select(I64, pc="echo_prob"), on="MVT_ID_mvt")
         .select("MVT_ID_mvt", p=0.577 * pl.col("pl_") + 0.428 * pl.col("pc")))
    d = (raw.join(p, on="MVT_ID_mvt").join(pl.read_parquet(EVAL / "v42_chain_holdout.parquet").select(I64, P="v42"), on="MVT_ID_mvt"))
    y, P, pp, off, mo = (d[c].to_numpy() for c in ("y", "P", "p", "off", "mo"))
    pb = np.digitize(pp, P_BANDS) - 1
    ob = np.digitize(off, OFF_BANDS) - 1
    cell = pb * 10 + ob
    r = y - P
    print(f"LIRF NM-matched holdout rows {len(y):,}; RMSE Jan {np.sqrt(np.mean(r[mo == 1] ** 2)):.1f}, Jul {np.sqrt(np.mean(r[mo == 7] ** 2)):.1f}")
    print(f"\n  {'p band':11s} {'offset':10s} {'n Jan':>6s} {'n Jul':>6s} {'bias Jan':>9s} {'bias Jul':>9s} {'%SSE':>6s}")
    offn = ["<30m", "30-60m", "1-3h", ">3h"]
    tot = np.sum(r ** 2)
    for c in sorted(set(cell)):
        k = cell == c
        if k.sum() < 30:
            continue
        kj, kl = k & (mo == 1), k & (mo == 7)
        bj = r[kj].mean() if kj.sum() else np.nan
        bl = r[kl].mean() if kl.sum() else np.nan
        print(f"  {P_BANDS[c // 10]:.2f}-{P_BANDS[c // 10 + 1]:.2f}   {offn[c % 10]:10s} {kj.sum():6,} {kl.sum():6,} {bj:+9.0f} {bl:+9.0f} {100 * np.sum(r[k] ** 2) / tot:6.1f}")
    print("\ncross-month: per-cell shrunk bias fitted on one month, applied to the other")
    for fit_m, app_m in ((1, 7), (7, 1)):
        adj = np.zeros(len(y))
        for c in set(cell):
            kf = (cell == c) & (mo == fit_m)
            if kf.sum():
                adj[(cell == c) & (mo == app_m)] = r[kf].mean() * kf.sum() / (kf.sum() + SHRINK)
        k = mo == app_m
        s0, s1 = np.sum(r[k] ** 2), np.sum((r[k] - adj[k]) ** 2)
        print(f"  fit {fit_m:2d} -> apply {app_m:2d}: LIRF NM-matched RMSE {np.sqrt(s0 / k.sum()):.2f} -> {np.sqrt(s1 / k.sum()):.2f} "
              f"(SSE {100 * (s1 / s0 - 1):+.2f}%)")


if __name__ == "__main__":
    main()
