"""Scoping: what sinks July? (PROGRESS §109; label-free signals only, no training)

On v42's holdout chain, per (airport, day), NM-matched departures with y <= 1 h (body) and all rows <= 5 h (trimmed):
  - concentration: share of the month's body SSE in its worst 5 / 10 airport-days, Jan vs Jul;
  - bias: mean residual (y - v42) on those days vs the rest;
  - day-level signals that exist in 2026: arrival taxi-in excess (mean taxi-in of the day's arrivals minus that
    airport-month's median day; arrivals are labelled in 2026) and METAR thunderstorm / snow share of the day's
    departures (base features). Correlation of the day mean residual with each signal, per month, and a cross-month
    test: a per-airport slope on the signal fitted in one month, applied to the other's day means.

Run:  .venv/Scripts/python.exe -u tests/july_days_scope.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)


def main() -> None:
    srcs = [str(ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet") for a, b in (("01", "02"), ("07", "08"))]
    m = pl.concat([pl.scan_parquet(s) for s in srcs]).with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
    dep = (m.filter(pl.col("PHASE_mvt") == "DEP").select(I64, "day", "mo", apt="ADEP_mvt", nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
                                                           y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).collect()
           .join(pl.read_parquet(ROOT / "cache" / "eval" / "v42_chain_holdout.parquet").select(I64, P="v42"), on="MVT_ID_mvt"))
    wx = pl.read_parquet(ROOT / "cache" / "features_nb" / "holdout_gap2025.parquet").select(I64, "wx_tstorm", "wx_snow")
    dep = dep.join(wx, on="MVT_ID_mvt", how="left").with_columns(r=pl.col("y") - pl.col("P"))
    arr = (m.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("TAXITIME_SEC_mvt").is_between(0, 3600))
           .group_by(pl.col("ADES_mvt").alias("apt"), "day", "mo").agg(tin=pl.col("TAXITIME_SEC_mvt").mean(), n_arr=pl.len()).collect())
    arr = arr.with_columns(tin_x=pl.col("tin") - pl.col("tin").median().over("apt", "mo"))
    body = dep.filter(pl.col("nm") & (pl.col("y") <= 3600))
    g = (body.group_by("apt", "day", "mo").agg(n=pl.len(), sse=(pl.col("r") ** 2).sum(), bias=pl.col("r").mean(),
                                                ts=pl.col("wx_tstorm").mean(), sn=pl.col("wx_snow").mean())
         .join(arr, on=["apt", "day", "mo"], how="left"))
    for mo in (1, 7):
        x = g.filter(pl.col("mo") == mo).sort("sse", descending=True)
        tot = x["sse"].sum()
        top = x.head(10)
        print(f"\nmonth {mo}: body RMSE {np.sqrt(tot / x['n'].sum()):.1f}; worst 5 airport-days {100 * x.head(5)['sse'].sum() / tot:.1f}% of SSE, "
              f"worst 10 {100 * top['sse'].sum() / tot:.1f}% (of {x.height} airport-days)")
        print(f"  {'apt':5s} {'day':10s} {'n':>5s} {'%SSE':>5s} {'rmse':>6s} {'bias':>6s} {'tstorm':>6s} {'snow':>5s} {'taxi-in excess':>14s}")
        for rr in top.iter_rows(named=True):
            print(f"  {rr['apt']:5s} {str(rr['day']):10s} {rr['n']:5d} {100 * rr['sse'] / tot:5.1f} {np.sqrt(rr['sse'] / rr['n']):6.0f} {rr['bias']:+6.0f} "
                  f"{rr['ts'] or 0:6.2f} {rr['sn'] or 0:5.2f} {rr['tin_x'] if rr['tin_x'] is not None else float('nan'):+14.0f}")
        rest = x.slice(10)
        print(f"  bias, worst-10 days n-weighted {np.average(top['bias'], weights=top['n']):+.1f} vs other days {np.average(rest['bias'], weights=rest['n']):+.1f}")
        for c in ("tin_x", "ts"):
            k = x.filter(pl.col(c).is_not_null())
            print(f"  corr(day bias, {c}) {np.corrcoef(k[c].to_numpy(), k['bias'].to_numpy())[0, 1]:+.3f}  (n {k.height})")
    print("\ncross-month: per-airport slope of day bias on taxi-in excess, fitted on one month, applied to the other (body RMSE)")
    for fm, am in ((1, 7), (7, 1)):
        f, a_ = g.filter((pl.col("mo") == fm) & pl.col("tin_x").is_not_null()), body.filter(pl.col("mo") == am)
        sl = {ap: float(np.sum(s["tin_x"] * s["bias"] * s["n"]) / max(np.sum(s["tin_x"] ** 2 * s["n"]), 1e-9))
              for ap, s in ((ap, f.filter(pl.col("apt") == ap)) for ap in f["apt"].unique())}
        adj = a_.join(arr.filter(pl.col("mo") == am).select("apt", "day", "tin_x"), on=["apt", "day"], how="left").with_columns(
            adj=pl.col("tin_x").fill_null(0) * pl.col("apt").replace_strict(sl, default=0.0))
        r = adj["r"].to_numpy(); d = adj["adj"].to_numpy()
        print(f"  fit {fm} -> apply {am}: {np.sqrt(np.mean(r ** 2)):.2f} -> {np.sqrt(np.mean((r - d) ** 2)):.2f}")


if __name__ == "__main__":
    main()
