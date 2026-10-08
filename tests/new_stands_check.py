"""Label-free check of v42 on stands that are new or rare in 2026 (PROGRESS.md §105): EDDF prefixes J/H (0.2% -> 7% of
departures) and EDDM stands never seen in 2025. Per group, NM-matched rows: n, 2025 training rows, mean a = T - AOBT_3
(<= 1 h), clipped mean (x - a) with x = truth (2025) or v42 (2026, and 2025 holdout), and the same vs the ADS-B taxi
(dwell/pass) where present. The 2025 truth-vs-anchor offset of the airport is the yardstick (§98).

Run:  .venv/Scripts/python.exe -u tests/new_stands_check.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_drift_triangulate as T  # noqa: E402

CLIP = 300


def main() -> None:
    allm = pl.concat([T.frame(str(T.RAW / "training_*.parquet"), 2025), T.frame(str(ROOT / "data" / "ranking" / "ranking.parquet"), 2026)]).collect()
    st = pl.concat([pl.scan_parquet(s).filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64), stand=pl.col("STAND_mvt").fill_null("?"),
                                                                                       full_month=pl.col("MVT_TIME_UTC_mvt").dt.month())
                    for s in (str(T.RAW / "training_*.parquet"), str(ROOT / "data" / "ranking" / "ranking.parquet"))]).collect()
    det = pl.concat([pl.read_parquet(p) for p in sorted(T.DET.glob("day=202[56]-0[17]*.parquet"))], how="diagonal").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_tier", pb=pl.col("adsb_pushback_ts").cast(pl.Int64))
    pred = pl.concat([pl.read_parquet(ROOT / "cache" / "eval" / "v42_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p="v42"),
                      pl.read_parquet(ROOT / "data" / "submissions" / "smart-jigsaw_v42.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))])
    d = (allm.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("apt").is_in(["EDDF", "EDDM"]) & pl.col("nm"))
         .join(st, on="MVT_ID_mvt").join(pred, on="MVT_ID_mvt", how="left").join(det, on="MVT_ID_mvt", how="left")
         .with_columns(at=(pl.col("t") - pl.col("pb")).cast(pl.Float64), good=pl.col("adsb_tier").is_in(["dwell", "pass"]).fill_null(False),
                       pfx=pl.col("stand").str.extract(r"^([A-Za-z]+)", 1)))
    seen25 = set(st.join(allm.filter(pl.col("yr") == 2025).select("MVT_ID_mvt"), on="MVT_ID_mvt")["stand"])
    d = d.with_columns(grp=pl.when((pl.col("apt") == "EDDF") & pl.col("pfx").is_in(["J", "H"])).then(pl.lit("EDDF J/H"))
                       .when(pl.col("apt") == "EDDF").then(pl.lit("EDDF other"))
                       .when(~pl.col("stand").is_in(list(seen25))).then(pl.lit("EDDM new"))
                       .otherwise(pl.lit("EDDM other")))
    cm = lambda e: e.clip(-CLIP, CLIP).mean()  # noqa: E731
    x = pl.when(pl.col("yr") == 2025).then(pl.col("y")).otherwise(pl.col("p"))
    g = (d.filter(pl.col("a").is_not_null()).group_by("grp", "yr").agg(
        n=pl.len(), n_ho=pl.col("p").is_not_null().sum(), a=pl.col("a").filter(pl.col("a") <= 3600).mean(),
        x_a=cm(x - pl.col("a")), p_a=cm((pl.col("p") - pl.col("a")).filter(pl.col("p").is_not_null())),
        n_ad=pl.col("good").sum(), x_ad=cm((x - pl.col("at")).filter(pl.col("good"))),
        p_ad=cm((pl.col("p") - pl.col("at")).filter(pl.col("good") & pl.col("p").is_not_null())),
        mean_x=x.mean()).sort("grp", "yr"))
    print("x = truth (2025, all months) / v42 (2026); p = v42 (2025 holdout Jan+Jul, and 2026); a = T - AOBT_3; ad = ADS-B taxi")
    print(f"{'group':11s} {'yr':>4s} {'n':>7s} {'n ho':>6s} {'a':>6s} {'x-a':>6s} {'p-a':>6s} | {'n ad':>6s} {'x-ad':>6s} {'p-ad':>6s} | {'mean x':>7s}")
    f = lambda v: f"{v:6.0f}" if v is not None else "     -"  # noqa: E731
    for r in g.iter_rows(named=True):
        print(f"{r['grp']:11s} {r['yr']:4d} {r['n']:7,} {r['n_ho']:6,} {f(r['a'])} {f(r['x_a'])} {f(r['p_a'])} | {r['n_ad']:6,} {f(r['x_ad'])} {f(r['p_ad'])} | {r['mean_x']:7.0f}")
    print("\nEDDF J/H 2025 rows by month (training support):", dict(d.filter((pl.col("grp") == "EDDF J/H") & (pl.col("yr") == 2025)).group_by("full_month").len().sort("full_month").iter_rows()))
    print("EDDF J/H 2026 rows by month:", dict(d.filter((pl.col("grp") == "EDDF J/H") & (pl.col("yr") == 2026)).group_by("full_month").len().sort("full_month").iter_rows()))
    print("EDDM new stands 2026 (top):", d.filter(pl.col("grp") == "EDDM new").group_by("stand").len().sort("len", descending=True).head(10).rows())


if __name__ == "__main__":
    main()
