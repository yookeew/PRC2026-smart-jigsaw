"""Neighbour rotation pressure, screened on top of v66's corrector (ctxc1sel_rf). Label-free.
slack = SOBT - inbound in-block (Stage 1 link). Over the other departures at the airport within +-30 min: share with
slack < 30 min, median slack, median T - SOBT of tight (< 30) and of slack (>= 60) neighbours, and their difference."""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "C:/Users/zhang/smart-jigsaw/src")
from screen import run
from post import flow_feats as FF
ROOT = "C:/Users/zhang/smart-jigsaw"; S = ROOT + "/cache/analysis/flow_screen"
fe = pl.read_parquet(ROOT + "/cache/features_nb/train2025.parquet", columns=["MVT_ID_mvt", "inbound_mvt_id"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
arr = pl.read_parquet(S + "/arr25.parquet").select(inbound_mvt_id="MVT_ID_mvt", A_own="B")
fe = fe.with_columns(pl.col("inbound_mvt_id").cast(pl.Int64)).join(arr, on="inbound_mvt_id", how="left")
f = pl.read_parquet(S + "/dep25_stop_rf.parquet").join(fe.select("MVT_ID_mvt", "A_own"), on="MVT_ID_mvt", how="left")
f = f.with_columns(slack=pl.col("S") - pl.col("A_own"))
print("slack coverage", f["slack"].is_not_null().mean(), "median", f["slack"].median())
a = f.select("MVT_ID_mvt", "apt", t="T")
b = f.select("apt", id_j="MVT_ID_mvt", tj="T", sl_j="slack", off_j="off")
parts = []
for ap in f["apt"].unique().to_list():
    x = FF._win_join(a.filter(pl.col("apt") == ap), b.filter(pl.col("apt") == ap), ["apt"], "t", "tj", 1800.0)
    parts.append(x.group_by("MVT_ID_mvt").agg(
        rt_n=pl.col("sl_j").is_not_null().sum(), rt_tight=(pl.col("sl_j") < 1800).mean(), rt_msl=pl.col("sl_j").median(),
        rt_off_tight=pl.col("off_j").filter(pl.col("sl_j") < 1800).median(),
        rt_off_slack=pl.col("off_j").filter(pl.col("sl_j") >= 3600).median()))
g = pl.concat(parts).with_columns(rt_dif=pl.col("rt_off_tight") - pl.col("rt_off_slack"))
f = f.join(g, on="MVT_ID_mvt", how="left")
f.write_parquet(S + "/dep25_rot_rf.parquet")
rt = ["rt_n", "rt_tight", "rt_msl", "rt_off_tight", "rt_off_slack", "rt_dif"]
st = [c for c in f.columns if c.startswith("st_")]
run(f, rt, label="neighbour rotation pressure (on rf)")
run(f, rt + st, label="rotation pressure + stoppage (on rf)")
