import os, numpy as np, polars as pl
S="C:/Users/zhang/smart-jigsaw/cache/analysis/flow_screen"
f=pl.read_parquet(S+"/dep25.parquet")
# destination-flow features: other departures (any challenge origin != own) to the same ADES with |T_j - T_i| <= W
f=f.with_columns(off=pl.col("T")-pl.col("S"), dl=pl.col("T")-pl.col("LOBT"))
def flow(W, same_origin=False):
    a=f.select("MVT_ID_mvt","apt","ades","T").sort("T")
    b=f.select(pl.col("apt").alias("apt_j"),"ades",pl.col("T").alias("Tj"),pl.col("off").alias("off_j"),pl.col("dl").alias("dl_j"),pl.col("MVT_ID_mvt").alias("id_j"))
    # bucket join: T rounded to W bins, neighbouring bins
    a=a.with_columns(bk=(pl.col("T")//W).cast(pl.Int64))
    b=b.with_columns(bk0=(pl.col("Tj")//W).cast(pl.Int64))
    parts=[]
    for d in (-1,0,1):
        parts.append(a.join(b.with_columns(bk=pl.col("bk0")-d),on=["ades","bk"],how="inner"))
    x=pl.concat(parts).filter((pl.col("id_j")!=pl.col("MVT_ID_mvt"))&((pl.col("Tj")-pl.col("T")).abs()<=W))
    x=x.filter(pl.col("apt_j")==pl.col("apt")) if same_origin else x.filter(pl.col("apt_j")!=pl.col("apt"))
    return x.group_by("MVT_ID_mvt").agg(n=pl.len(),moff=pl.col("off_j").median(),mdl=pl.col("dl_j").median(),
            f30=(pl.col("off_j")>1800).mean())
for W,so in [(3600,False),(5400,True)]:
    g=flow(W,so); tag=("so" if so else "xo")+str(W//60)
    g=g.rename({c:f"{tag}_{c}" for c in g.columns if c!="MVT_ID_mvt"})
    f=f.join(g,on="MVT_ID_mvt",how="left")
f.write_parquet(S+"/dep25_flow.parquet")
nm=f.filter(~pl.col("nm"))
print("coverage xo60 (>=1 other-origin flight to same ADES within 60 min):", nm["xo60_n"].is_not_null().mean())
print(nm.group_by("apt").agg(cov=pl.col("xo60_n").is_not_null().mean(),covso=pl.col("so90_n").is_not_null().mean()).sort("apt"))
