"""Scoping (PROGRESS §117): bounding the hedge non-echo branch by the late-branch limit off - 3606."""
import sys; sys.path.insert(0,"src")
import polars as pl, numpy as np
import post.lirf_signals as L
d = L.frame(str(L.RAW/"training_*.parquet")).join(L.p_2025(), on="MVT_ID_mvt", how="left")
d = d.join(pl.read_parquet(L.EVAL/"v42_chain_holdout.parquet").select(L.I64,"v42"), on="MVT_ID_mvt", how="left")
def run(z, Pcol, clf_by_month):
    out=[]
    for m in sorted(z["mt"].unique().to_list()):
        te = z.filter(pl.col("mt")==m)
        if te.height==0: continue
        P = te[Pcol].to_numpy(); P1,k,q = L.apply(te, P, clf_by_month(m))       # v45 treatment
        p = te["p"].fill_null(1.0).to_numpy(); off = te["off"].to_numpy(); offc=np.clip(off,0,L.WIDE); y = te["y"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"): pne = (P1-q*offc)/(1-q)
        for fl in (0, 120, 300):
            ub = np.maximum(off-3606, fl)
            P2 = np.where(k, q*offc + (1-q)*np.minimum(pne, ub), P1)
            out.append((m, fl, np.sum(((y-P1)**2)[k]), np.sum(((y-P2)**2)[k]), k.sum(), np.mean((pne>ub)[k]) if k.any() else 0))
    return pl.DataFrame(out, schema=["m","floor","sse_v45","sse_clip","n","viol"], orient="row")
clfs = {m: L.fit(d.filter(pl.col("mt")!=m)) for m in d["mt"].unique().to_list()}
r = run(d.filter(pl.col("Pb").is_not_null()), "Pb", lambda m: clfs[m])
print("12 months, base level (v45-treated):")
print(r.group_by("floor").agg(sse_v45=pl.col("sse_v45").sum(), sse_clip=pl.col("sse_clip").sum(), months_better=(pl.col("sse_clip")<pl.col("sse_v45")).sum(), viol=(pl.col("viol")*pl.col("n")).sum()/pl.col("n").sum()).sort("floor"))
clf10 = L.fit(d.filter(~pl.col("mt").is_in(["2025-01","2025-07"])))
r2 = run(d.filter(pl.col("v42").is_not_null()), "v42", lambda m: clf10)
print("holdout chain (v42 + v45 treatment):"); print(r2.sort("floor","m"))
