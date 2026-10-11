"""Airline-network delay (PROGRESS §137 follow-up): the same operator's departures at the OTHER airports within
+-60 min (count, median T - SOBT, median T - LOBT, share > 30 min late) and its arrivals at all ten airports within
+-60 min (count, median landing - SIBT). Label-free. Writes cache/analysis/flow_screen/dep25_air.parquet."""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
sys.path.insert(0, "C:/Users/zhang/smart-jigsaw/src")
from post import flow_feats as FF
S = "C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"
f = pl.read_parquet(S + "/dep25_flow2.parquet")
arr = pl.read_parquet(S + "/arr25.parquet")
import glob
raw = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"]) for p in sorted(glob.glob("C:/Users/zhang/smart-jigsaw/data/raw/training_2025-*.parquet"))])
arr = arr.join(raw.select(pl.col("MVT_ID_mvt").cast(pl.Int64), op="AIRCRAFT_OPERATOR_flt"), on="MVT_ID_mvt", how="left")
a = f.select("MVT_ID_mvt", "apt", "op", t=pl.col("T"))
b = f.select(id_j="MVT_ID_mvt", apt_j="apt", op="op", tj="T", off_j="off", dl_j="dl")
x = FF._win_join(a, b, ["op"], "t", "tj", 3600.0).filter(pl.col("apt_j") != pl.col("apt"))
g1 = FF._stats(x, "anx")
ba = arr.select(id_j=-pl.col("MVT_ID_mvt"), op="op", tj="T", off_j=pl.col("T") - pl.col("S"), dl_j=pl.lit(None, pl.Float64))
x = FF._win_join(a, ba, ["op"], "t", "tj", 3600.0)
g2 = x.group_by("MVT_ID_mvt").agg(ana_n=pl.len(), ana_moff=pl.col("off_j").median())
f = f.join(g1, on="MVT_ID_mvt", how="left").join(g2, on="MVT_ID_mvt", how="left")
f.write_parquet(S + "/dep25_air.parquet")
print(f.select([c for c in f.columns if c.startswith(("anx_", "ana_"))]).describe())
