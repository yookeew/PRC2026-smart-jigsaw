"""Follow-up to tests/anchor_drift_check.py: is a 2026 move of (prediction - AOBT_3 anchor) a real taxi change
or an AOBT_3 recording change? Two label-free witnesses, 2025 vs 2026, Jan/Jul, per airport:
  - arrival taxi-in (labelled in both years): does ground congestion move with the anchor?
  - ADS-B pushback detections (cache/adsb_pushback_v4, tiers dwell/pass): x - adsb_taxi and adsb_taxi - a.
    If adsb_taxi - a is stable but the anchor grew, taxis really grew; if adsb_taxi - a drops, AOBT_3 moved.
Means are of values clipped to +-CLIP s (the minute-quantised medians step).

Run:  .venv/Scripts/python.exe -u tests/anchor_drift_triangulate.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW, DET = ROOT / "data" / "raw", ROOT / "cache" / "adsb_pushback_v4"
CLIP = 300
GOOD = ["dwell", "pass"]


def frame(src: str, yr: int) -> pl.LazyFrame:
    return (pl.scan_parquet(src).filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
            .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "PHASE_mvt", apt="ADEP_mvt", ades="ADES_mvt",
                    mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                    t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
                    a=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
            .with_columns(yr=pl.lit(yr)))


def main() -> None:
    allm = pl.concat([frame(str(RAW / "training_*.parquet"), 2025), frame(str(ROOT / "data" / "ranking" / "ranking.parquet"), 2026)]).collect()
    arr = allm.filter(pl.col("PHASE_mvt") == "ARR").with_columns(apt=pl.col("ades"))
    det = pl.concat([pl.read_parquet(p) for p in sorted(DET.glob("day=202[56]-0[17]*.parquet"))], how="diagonal")
    det = det.select(pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_tier", pb=pl.col("adsb_pushback_ts"))
    if det["pb"].dtype != pl.Int64:
        det = det.with_columns(pb=pl.col("pb").dt.epoch("s") if det["pb"].dtype.is_temporal() else pl.col("pb").cast(pl.Int64))
    pred = pl.concat([
        pl.read_parquet(ROOT / "cache" / "eval" / "v42_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p="v42"),
        pl.read_parquet(ROOT / "data" / "submissions" / "smart-jigsaw_v42.parquet")
        .select(pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))])
    dep = (allm.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("nm") & pl.col("a").is_not_null())
           .join(pred, on="MVT_ID_mvt", how="left").join(det, on="MVT_ID_mvt", how="left")
           .with_columns(at=(pl.col("t") - pl.col("pb")).cast(pl.Float64), good=pl.col("adsb_tier").is_in(GOOD).fill_null(False)))
    cm = lambda e: e.clip(-CLIP, CLIP).mean()
    # x = truth in 2025, prediction in 2026; also the model's 2025 prediction
    x = pl.when(pl.col("yr") == 2025).then(pl.col("y")).otherwise(pl.col("p"))
    g = dep.group_by("apt", "mo", "yr").agg(
        a_mean=pl.col("a").filter(pl.col("a") <= 3600).mean(),
        x_a=cm(x - pl.col("a")), p_a=cm(pl.col("p") - pl.col("a")),
        good_n=pl.col("good").sum(),
        ad_a=cm((pl.col("at") - pl.col("a")).filter(pl.col("good"))),
        x_ad=cm((x - pl.col("at")).filter(pl.col("good"))),
        p_ad=cm((pl.col("p") - pl.col("at")).filter(pl.col("good"))))
    ti = arr.filter(pl.col("y").is_not_null() & (pl.col("y") <= 3600)).group_by("apt", "mo", "yr").agg(
        taxi_in=pl.col("y").mean(), n_arr=pl.len())
    g = g.join(ti, on=["apt", "mo", "yr"], how="left").sort("apt", "mo", "yr")
    print("x = truth (2025) / v42 prediction (2026); p = v42 prediction (2025 holdout, 2026); a = T - AOBT_3; ad = ADS-B taxi")
    print(f"{'apt':5s} {'mo':>2s} {'yr':>4s} | {'a mean':>6s} {'x-a':>6s} {'p-a':>6s} | {'ADS-B n':>7s} {'ad-a':>6s} "
          f"{'x-ad':>6s} {'p-ad':>6s} | {'taxi-in':>7s}")
    for r in g.iter_rows(named=True):
        f = lambda v: f"{v:6.0f}" if v is not None else "     -"
        print(f"{r['apt']:5s} {r['mo']:2d} {r['yr']:4d} | {f(r['a_mean'])} {f(r['x_a'])} {f(r['p_a'])} | {r['good_n']:7d} "
              f"{f(r['ad_a'])} {f(r['x_ad'])} {f(r['p_ad'])} | {f(r['taxi_in'])}")


if __name__ == "__main__":
    main()
