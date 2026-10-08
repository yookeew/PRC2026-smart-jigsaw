"""Follow-up (PROGRESS §110): July 2026 arrival taxi-in excess (arrx) at EDDF/LEMD dropped below 2025's monthly range.
Real operational change or recording change? Raw arrivals, Jul 2025 vs Jul 2026 (and Jan for reference): by runway
(share, median taxi-in), by stand prefix, and the taxi-in quantiles. Arrival times are never blanked.

Run:  .venv/Scripts/python.exe -u tests/arr_taxiin_shift.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = [ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet" for a, b in (("01", "02"), ("07", "08"))] + [ROOT / "data" / "ranking" / "ranking.parquet"]


def main() -> None:
    cols = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt"]
    a = (pl.concat([pl.read_parquet(p, columns=cols) for p in SRC])
         .filter((pl.col("PHASE_mvt") == "ARR") & pl.col("ADES_mvt").is_in(["EDDF", "LEMD", "EDDM", "LSZH"]))
         .with_columns(ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
                       tin=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds(),
                       sp=pl.col("STAND_mvt").str.slice(0, 1))
         .filter(pl.col("tin").is_between(0, 3600)))
    q = lambda: [pl.len().alias("n"), *[pl.col("tin").quantile(x).alias(f"q{int(x*100)}") for x in (.1, .5, .9)], pl.col("tin").mean().round(0).alias("mean")]  # noqa: E731
    with pl.Config(tbl_rows=60, tbl_width_chars=220, tbl_cols=12):
        print("arrival taxi-in by airport x month (s)")
        print(a.group_by("ADES_mvt", "ym").agg(q()).sort("ADES_mvt", "ym"))
        for apt in ("EDDF", "LEMD"):
            x = a.filter(pl.col("ADES_mvt") == apt)
            r = (x.group_by("RUNWAY_mvt", "ym").agg(pl.len().alias("n"), pl.col("tin").median().alias("med"))
                 .with_columns(share=(pl.col("n") / pl.col("n").sum().over("ym")).round(3)))
            print(f"\n{apt}: by landing runway, share | median taxi-in")
            print(r.pivot(on="ym", index="RUNWAY_mvt", values=["share", "med"]).sort("RUNWAY_mvt"))
            s = (x.group_by("sp", "ym").agg(pl.len().alias("n"), pl.col("tin").median().alias("med"))
                 .with_columns(share=(pl.col("n") / pl.col("n").sum().over("ym")).round(3)))
            print(f"\n{apt}: by stand prefix, share | median taxi-in (prefixes >= 2% in some month)")
            p = s.pivot(on="ym", index="sp", values=["share", "med"])
            sh = [c for c in p.columns if c.startswith("share")]
            print(p.filter(pl.max_horizontal(sh) >= 0.02).sort("sp"))
            print(f"\n{apt}: share of BLOCK seconds == 0 (time resolution) by month")
            print(x.group_by("ym").agg((pl.col("BLOCK_TIME_UTC_mvt").dt.second() == 0).mean().round(3).alias("block_sec0"),
                                       (pl.col("MVT_TIME_UTC_mvt").dt.second() == 0).mean().round(3).alias("mvt_sec0")).sort("ym"))


if __name__ == "__main__":
    main()
