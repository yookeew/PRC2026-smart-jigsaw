"""Long-taxi analysis (brainstorm item 2): on NM-matched holdout rows, does ADS-B (cache/adsb_pushback_v4) see the
long taxis, and does v42 use what it sees? No training.

Per bin of the ADS-B taxi (T - detected pushback) and per bin of v42's prediction: rows, truth mean, v42 mean,
ADS-B taxi mean, RMSE of v42 and of the ADS-B taxi (+ the bin's own median lag, an in-sample upper bound), and
the share of SSE. Also the truth-binned view (outcome-conditioned, information only): how many long-taxi rows have
a detection at all, by tier.

Run:  .venv/Scripts/python.exe -u tests/long_taxi_adsb.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "cache" / "adsb_pushback_v4"
BINS = [0, 600, 1200, 1800, 2400, 3600, 5400, 18000]


def main() -> None:
    raw = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet"))
                     .filter(pl.col("PHASE_mvt") == "DEP")
                     .select(pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                             nm=pl.col("FLIGHT_ID_mvt").is_not_null(), t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"),
                             a=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
                     for a, b in (("01", "02"), ("07", "08"))]).collect()
    det = pl.concat([pl.read_parquet(p) for p in sorted(DET.glob("day=2025-0[17]*.parquet"))], how="diagonal")
    det = det.select(pl.col("MVT_ID_mvt").cast(pl.Int64), tier="adsb_tier", pb=pl.col("adsb_pushback_ts").cast(pl.Int64),
                     mmin="adsb_min_own_m", npts="adsb_n_pts")
    v42 = pl.read_parquet(ROOT / "cache" / "eval" / "v42_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "v42")
    d = (v42.join(raw, on="MVT_ID_mvt").join(det, on="MVT_ID_mvt", how="left")
         .with_columns(at=(pl.col("t") - pl.col("pb")).cast(pl.Float64), tier=pl.col("tier").fill_null("none"))
         .filter(pl.col("nm") & (pl.col("y") <= 18000)))
    sse_tot = float(((d["y"] - d["v42"]) ** 2).sum())
    print(f"NM-matched holdout rows y<=5h: {d.height:,}; v42 RMSE {np.sqrt(sse_tot / d.height):.1f}")

    def table(col: str, sub: pl.DataFrame, title: str) -> None:
        print(f"\n{title}")
        print(f"  {'bin (s)':>12s} {'n':>7s} {'%SSE':>6s} {'y':>6s} {'v42':>6s} {'adsb':>6s} {'a3':>6s} "
              f"{'rmse v42':>8s} {'rmse ad+lag':>11s} {'rmse a3+lag':>11s} {'bias v42':>8s}")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = sub.filter((pl.col(col) > lo) & (pl.col(col) <= hi))
            if s.height < 30:
                continue
            y, p = s["y"].to_numpy(), s["v42"].to_numpy()
            at, a3 = s["at"].to_numpy(), s["a"].to_numpy()
            rm = lambda e: float(np.sqrt(np.nanmean(e ** 2)))  # noqa: E731
            lag_ad = np.nanmedian(y - at) if np.isfinite(at).any() else np.nan
            lag_a3 = np.nanmedian(y - a3) if np.isfinite(a3).any() else np.nan
            print(f"  {lo:5d}-{hi:<6d} {s.height:7,} {100 * ((y - p) ** 2).sum() / sse_tot:6.1f} {y.mean():6.0f} {p.mean():6.0f} "
                  f"{np.nanmean(at) if np.isfinite(at).any() else np.nan:6.0f} {np.nanmean(a3):6.0f} {rm(y - p):8.0f} "
                  f"{rm(y - at - lag_ad):11.0f} {rm(y - a3 - lag_a3):11.0f} {np.mean(p - y):+8.0f}")

    good = d.filter(pl.col("tier").is_in(["dwell", "pass"]))
    table("at", good, "rows with a dwell/pass ADS-B pushback, binned by the ADS-B taxi (label-free)")
    table("v42", good, "same rows, binned by v42's prediction (label-free)")
    table("a", d.filter(pl.col("a").is_not_null()), "all rows with AOBT_3, binned by the AOBT_3 taxi (label-free)")
    print("\ntruth-binned (outcome-conditioned, info only): share of rows by ADS-B tier")
    for lo, hi in ((0, 1800), (1800, 3600), (3600, 18000)):
        s = d.filter((pl.col("y") > lo) & (pl.col("y") <= hi))
        sh = s.group_by("tier").len().with_columns(sh=pl.col("len") / s.height).sort("tier")
        print(f"  y {lo}-{hi}: n {s.height:,}, %SSE {100 * float(((s['y'] - s['v42']) ** 2).sum()) / sse_tot:.1f}  "
              + "  ".join(f"{r['tier']} {r['sh']:.2f}" for r in sh.iter_rows(named=True)))


if __name__ == "__main__":
    main()
