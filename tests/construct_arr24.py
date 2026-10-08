import polars as pl, glob
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(14)
def load(src):
    return pl.concat([pl.read_parquet(f) for f in src], how="diagonal_relaxed").with_columns(
        off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(), y=pl.col("TAXITIME_SEC_mvt"),
        day=pl.col("MVT_TIME_UTC_mvt").dt.date(), nmu=pl.col("FLIGHT_ID_mvt").is_null())
d25 = load(sorted(glob.glob("data/raw/training_*.parquet"))); d26 = load(["data/ranking/ranking.parquet"])
for nm, d in (("2025", d25), ("2026", d26)):
    a = d.filter(pl.col("PHASE_mvt")=="ARR")
    big = a.filter(pl.col("y").abs()>=40000)
    print(f"\n{nm} arrivals with |taxi-in| >= 40,000 s by airport:", big.group_by("ADES_mvt").len().sort("len",descending=True).rows())
    print(big.filter(pl.col("ADES_mvt")=="LIRF").select("day","FLIGHT_mvt","nmu","SCHED_TIME_UTC_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","y").sort("day").head(40))
dep24 = d25.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="LIRF")&pl.col("nmu")&(pl.col("off")>=43200)&((pl.col("y")-86400).abs()<5400)).select("day","FLIGHT_mvt","off","y").sort("day")
print("\n2025 LIRF +24h departures:"); print(dep24)
