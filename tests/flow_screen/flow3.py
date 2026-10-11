import os, sys, numpy as np, polars as pl
S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from screen import run
f=pl.read_parquet(S+"/dep25_flow2.parquet")
arr=pl.read_parquet(S+"/arr25.parquet")
W=1800
a=f.select("MVT_ID_mvt","apt","op","T").with_columns(bk=(pl.col("T")//W).cast(pl.Int64))
b=f.select("apt","op",pl.col("T").alias("Tj"),pl.col("off").alias("off_j"),pl.col("dl").alias("dl_j"),pl.col("MVT_ID_mvt").alias("id_j")).with_columns(bk0=(pl.col("Tj")//W).cast(pl.Int64))
x=pl.concat([a.join(b.with_columns(bk=pl.col("bk0")-d),on=["apt","op","bk"],how="inner") for d in (-1,0,1)]).filter((pl.col("id_j")!=pl.col("MVT_ID_mvt"))&((pl.col("Tj")-pl.col("T")).abs()<=W))
op=x.group_by("MVT_ID_mvt").agg(op_n=pl.len(),op_moff=pl.col("off_j").median(),op_mdl=pl.col("dl_j").median())
f=f.join(op,on="MVT_ID_mvt",how="left")
dest=[c for c in f.columns if c.startswith(("xo60_","so90_"))]; reg=[c for c in f.columns if c.startswith(("rso_","rxo_"))]
seq=["sq_n","sq_over","sq_under","sq_mdT","sq_net"]; opc=["op_n","op_moff","op_mdl"]
run(f,opc,label="operator window alone")
run(f,dest+reg+seq,label="dest+region+seq (block-13 candidate)")
