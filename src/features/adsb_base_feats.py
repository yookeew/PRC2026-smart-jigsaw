"""ADS-B detector fields as base-model features (reports/adsb_base_preregistration.md).

cache/features_nb/{train2025,holdout_gap2025,ranking}.parquet + the v4 detector output
(cache/adsb_pushback_v4/day=*.parquet), joined on MVT_ID_mvt -> cache/features_nb_adsb/.
Times are made relative to the takeoff T, so no absolute timestamp enters the model.
labels2025.parquet is copied unchanged.

Run:  .venv/Scripts/python.exe src/features/adsb_base_feats.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import shutil
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cache" / "features_nb"
DST = ROOT / "cache" / "features_nb_adsb"
DET = ROOT / "cache" / "adsb_pushback_v4"
TIERS = ["appear", "dwell", "pass", "fb_dwell", "fb_appear"]


def detections() -> pl.DataFrame:
    parts = []
    for p in sorted(DET.glob("day=*.parquet")):
        parts.append(pl.read_parquet(p).with_columns(adsb_day=pl.lit(1.0)))
    det = pl.concat(parts, how="diagonal")
    return det.select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_day",
        adsb_cov=pl.col("adsb_day_coverage").cast(pl.Float64),
        adsb_match=pl.col("adsb_matched").cast(pl.Float64),
        adsb_fb=pl.col("adsb_fallback").cast(pl.Float64),
        adsb_tier_code=pl.col("adsb_tier").replace_strict({t: i + 1 for i, t in enumerate(TIERS)},
                                                          default=0, return_dtype=pl.Float64),
        adsb_pb_ts=pl.col("adsb_pushback_ts"), adsb_first=pl.col("adsb_first_ts"),
        adsb_takeoff_gap_s=pl.col("adsb_takeoff_gap_s"), adsb_n_pts=pl.col("adsb_n_pts").cast(pl.Float64),
        adsb_first_gs=pl.col("adsb_first_gs"), adsb_first_own_m=pl.col("adsb_first_own_m"),
        adsb_min_own_m=pl.col("adsb_min_own_m"), adsb_pb_gap_s=pl.col("adsb_pb_gap_s"),
        adsb_pb_dist_m=pl.col("adsb_pb_dist_m"), adsb_pb_gs=pl.col("adsb_pb_gs"),
    ).unique("MVT_ID_mvt", keep="first")


def add(frame: pl.DataFrame, det: pl.DataFrame) -> pl.DataFrame:
    t = pl.col("T").dt.epoch("ms") / 1000.0
    out = (frame.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id"))
           .join(det.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
           .with_columns(adsb_taxi=t - pl.col("adsb_pb_ts"), adsb_L=t - pl.col("adsb_first"))
           .drop("_id", "adsb_pb_ts", "adsb_first"))
    assert out.height == frame.height
    return out


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    det = detections()
    for name in ("train2025", "holdout_gap2025", "ranking"):
        f = add(pl.read_parquet(SRC / f"{name}.parquet"), det)
        f.write_parquet(DST / f"{name}.parquet")
        day = f["adsb_day"].is_not_null().mean()
        m = (f["adsb_match"] == 1).mean()
        pb = f["adsb_taxi"].is_not_null().mean()
        print(f"{name}: {f.height:,} rows, {f.width} cols; ADS-B day {day:.3f}, matched {m:.3f}, pushback {pb:.3f}")
    shutil.copy2(SRC / "labels2025.parquet", DST / "labels2025.parquet")


if __name__ == "__main__":
    main()
