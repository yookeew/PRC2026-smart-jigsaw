"""Is the first arrival at our stand after SOBT + 5 min our own (late) inbound, or a re-occupier?
Label (2025): re-occupier iff its in-block > our true off-block (BLOCK). Label-free identity cues:
  origin cue  : arrival ADEP == our ADES (out-and-back rotation)
  type cue    : arrival type == our type
  operator cue: arrival operator == our operator
  flight-number cue: same airline prefix, numbers differ by 1 (e.g. BA123 / BA124)
Population: non-LIRF departures with such an arrival >= 30 min before T."""
import os, glob, re
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="vertical_relaxed")
dep = (tr.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("STAND_mvt").is_not_null() & (pl.col("ADEP_mvt") != "LIRF"))
       .select(pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", stand="STAND_mvt", T="MVT_TIME_UTC_mvt", S="SCHED_TIME_UTC_mvt",
               B="BLOCK_TIME_UTC_mvt", ades="ADES_mvt", dtype=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
               dop="AIRCRAFT_OPERATOR_flt", dfl="FLIGHT_mvt", nm=pl.col("FLIGHT_ID_mvt").is_not_null())
       .with_columns(t_lo=pl.col("S") + pl.duration(minutes=5)))
arr = (tr.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
       .select(apt="ADES_mvt", stand="STAND_mvt", a_in="BLOCK_TIME_UTC_mvt", a_adep="ADEP_mvt",
               atype=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), aop="AIRCRAFT_OPERATOR_flt", afl="FLIGHT_mvt").sort("a_in"))
j = dep.filter(pl.col("t_lo").is_not_null()).sort("t_lo").join_asof(arr, left_on="t_lo", right_on="a_in", by=["apt", "stand"], strategy="forward")
j = j.filter(pl.col("a_in").is_not_null() & (pl.col("a_in") < pl.col("T")))
j = j.with_columns(gap=(pl.col("T") - pl.col("a_in")).dt.total_seconds()).filter(pl.col("gap") >= 1800)


def fl_num(c):
    return pl.col(c).str.extract(r"(\d+)\D*$", 1).cast(pl.Int64, strict=False)


j = j.with_columns(
    reocc=pl.col("a_in") > pl.col("B"),
    origin=(pl.col("a_adep") == pl.col("ades")).fill_null(False),
    tmatch=(pl.col("atype") == pl.col("dtype")).fill_null(False),
    omatch=(pl.col("aop") == pl.col("dop")).fill_null(False),
    flnear=((pl.col("afl").str.slice(0, 2) == pl.col("dfl").str.slice(0, 2)) & ((fl_num("afl") - fl_num("dfl")).abs() <= 1)).fill_null(False),
)
print("population:", j.height, " re-occupier share:", j["reocc"].mean())
print(j.group_by("origin", "tmatch").agg(n=pl.len(), reocc=pl.col("reocc").mean()).sort("origin", "tmatch"))
print(j.group_by("flnear").agg(n=pl.len(), reocc=pl.col("reocc").mean()))
print(j.group_by("origin", "tmatch", "omatch", "flnear").agg(n=pl.len(), reocc=pl.col("reocc").mean()).sort("reocc", descending=True).head(16))
