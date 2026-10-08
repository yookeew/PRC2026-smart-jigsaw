"""Project a holdout gain onto the 2026 ranking composition.

Cell = (airport, month, served by combiner [has ADS-B info] vs neighbour stage [no ADS-B info]).
Expected 2026 dSSE = sum over cells of (holdout dSSE per row) x (2026 rows in the cell).
Usage: reweight26.py DIR ARM [exclude_LIRF=0/1]
"""
import os, sys, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
pl.Config.set_tbl_rows(60)
d0, arm = sys.argv[1], sys.argv[2]
ex_lirf = len(sys.argv) > 3 and sys.argv[3] == "1"


def has_info(pattern):
    det = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_matched", "adsb_tier"])
                     for p in sorted(glob.glob(f"cache/adsb_pos_v4/{pattern}"))])
    return det.select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                      has=(pl.col("adsb_matched").fill_null(False) | pl.col("adsb_tier").is_in(["fb_dwell", "fb_appear"]).fill_null(False)))


files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
raw = (pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
       .select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), apt="ADEP_mvt",
               mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), nmu=pl.col("FLIGHT_ID_mvt").is_null()))
ctl = pl.read_parquet(f"{d0}/chain_ctl.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), b="pred")
trt = pl.read_parquet(f"{d0}/chain_{arm}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p="pred")
h = raw.join(ctl, on="MVT_ID_mvt").join(trt, on="MVT_ID_mvt").join(has_info("day=2025-0[17]*.parquet"), on="MVT_ID_mvt", how="left")
h = h.with_columns(has=pl.col("has").fill_null(False))
if ex_lirf:
    h = h.with_columns(p=pl.when(pl.col("apt") == "LIRF").then(pl.col("b")).otherwise(pl.col("p")))
h = h.with_columns(d=(pl.col("y") - pl.col("p")) ** 2 - (pl.col("y") - pl.col("b")) ** 2,
                   body=(~pl.col("nmu")) & (pl.col("y") <= 3600))
cells = h.group_by("apt", "mo", "has").agg(n25=pl.len(), dpr=pl.col("d").mean(), dpr_body=(pl.col("d") * pl.col("body")).mean())
rk = (pl.read_parquet("data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
      .select(pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
      .join(has_info("day=2026-*.parquet"), on="MVT_ID_mvt", how="left").with_columns(has=pl.col("has").fill_null(False)))
n26 = rk.group_by("apt", "mo", "has").agg(n26=pl.len())
c = cells.join(n26, on=["apt", "mo", "has"], how="full", coalesce=True).fill_null(0)
c = c.with_columns(e25=pl.col("dpr") * pl.col("n25"), e26=pl.col("dpr") * pl.col("n26"),
                   e26_body=pl.col("dpr_body") * pl.col("n26"))
print(c.sort("apt", "mo", "has").with_columns(pl.col("dpr").round(1), (pl.col("e25") / 1e6).round(1), (pl.col("e26") / 1e6).round(1)))
S_board = 245.3839 ** 2 * 344841
tot25, tot26, tot26b = c["e25"].sum(), c["e26"].sum(), c["e26_body"].sum()
print(f"\nholdout dSSE {tot25:.4g}; projected 2026 dSSE {tot26:.4g} (body-only cells {tot26b:.4g})")
print(f"projected board: {np.sqrt((S_board + tot26) / 344841):.3f} (from 245.384, {np.sqrt((S_board + tot26) / 344841) - 245.3839:+.3f}); "
      f"body-only projection {np.sqrt((S_board + tot26b) / 344841) - 245.3839:+.3f}")
