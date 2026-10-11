import os, sys, numpy as np, polars as pl
S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from screen import run
f=pl.read_parquet(S+"/dep25_flow.parquet").with_columns(reg=pl.col("ades").str.slice(0,2))
def win_join(a,b,key,tcol_a,tcol_b,W):
    a=a.with_columns(bk=(pl.col(tcol_a)//W).cast(pl.Int64)); b=b.with_columns(bk0=(pl.col(tcol_b)//W).cast(pl.Int64))
    x=pl.concat([a.join(b.with_columns(bk=pl.col("bk0")-d),on=key+["bk"],how="inner") for d in (-1,0,1)])
    return x.filter((pl.col("id_j")!=pl.col("MVT_ID_mvt"))&((pl.col(tcol_b)-pl.col(tcol_a)).abs()<=W))
W=3600
a=f.select("MVT_ID_mvt","apt","reg","ades","T")
b=f.select(pl.col("apt").alias("apt_j"),"reg",pl.col("ades").alias("ades_j"),pl.col("T").alias("Tj"),pl.col("off").alias("off_j"),pl.col("dl").alias("dl_j"),pl.col("MVT_ID_mvt").alias("id_j"))
x=win_join(a,b,["reg"],"T","Tj",W).filter(pl.col("ades_j")!=pl.col("ades"))
agg=lambda g,tag: g.group_by("MVT_ID_mvt").agg(**{f"{tag}_n":pl.len(),f"{tag}_moff":pl.col("off_j").median(),f"{tag}_mdl":pl.col("dl_j").median(),f"{tag}_f30":(pl.col("off_j")>1800).mean()})
f=f.join(agg(x.filter(pl.col("apt_j")==pl.col("apt")),"rso"),on="MVT_ID_mvt",how="left").join(agg(x.filter(pl.col("apt_j")!=pl.col("apt")),"rxo"),on="MVT_ID_mvt",how="left")
del x
# sequence displacement at own airport: neighbours by schedule +-20 min
W2=1200
a=f.select("MVT_ID_mvt","apt","S","T"); b=f.select("apt",pl.col("S").alias("Sj"),pl.col("T").alias("Tj"),pl.col("MVT_ID_mvt").alias("id_j"))
x=win_join(a,b,["apt"],"S","Sj",W2)
sq=x.group_by("MVT_ID_mvt").agg(sq_n=pl.len(),
   sq_over=((pl.col("Sj")>pl.col("S"))&(pl.col("Tj")<pl.col("T"))).sum(),
   sq_under=((pl.col("Sj")<pl.col("S"))&(pl.col("Tj")>pl.col("T"))).sum(),
   sq_mdT=(pl.col("Tj")-pl.col("T")).median())
del x
f=f.join(sq,on="MVT_ID_mvt",how="left").with_columns(sq_net=(pl.col("sq_over")-pl.col("sq_under"))/pl.col("sq_n"))
f.write_parquet(S+"/dep25_flow2.parquet")
dest=[c for c in f.columns if c.startswith(("xo60_","so90_"))]
reg=[c for c in f.columns if c.startswith(("rso_","rxo_"))]
seq=["sq_n","sq_over","sq_under","sq_mdT","sq_net"]
run(f,reg,label="region flow alone")
run(f,seq,label="sequence displacement alone")
run(f,dest+reg,label="dest+region flow")
