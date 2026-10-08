"""Scoping (PROGRESS §118): same-month / same-day airline matched rate on top of v48."""
import sys; sys.path.insert(0,"src")
import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
import post.lirf_signals as L
import post.lirf_hedge2 as H
from post.lirf_twin_bounds import twin_frame
src = str(L.RAW/"training_*.parquet")
def period_feats(src):
    a = (pl.scan_parquet(src).filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="LIRF")).select(L.I64, al=pl.col("FLIGHT_mvt").str.slice(0,3),
         nm=pl.col("FLIGHT_ID_mvt").is_not_null(), off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
         mt=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("SCHED_TIME_UTC_mvt").dt.date()).collect())
    late = a.filter(pl.col("off")>=3600)
    out = a.select("MVT_ID_mvt","al","mt","day","nm")
    for key, nm in ((["al","mt"],"m"), (["al","day"],"d")):
        g = late.group_by(key).agg(n=pl.len(), k=pl.col("nm").sum())
        out = out.join(g, on=key, how="left").with_columns(pl.col("n","k").fill_null(0))
        # exclude own row if it is itself a late departure (scope rows are late & unmatched -> subtract 1 from n)
        out = out.join(late.select("MVT_ID_mvt", own=pl.lit(1)), on="MVT_ID_mvt", how="left").with_columns(own=pl.col("own").fill_null(0))
        out = out.with_columns(n_=(pl.col("n")-pl.col("own")).clip(0, None), k_=(pl.col("k")-pl.col("own")*pl.col("nm").cast(pl.Int64)).clip(0, None))
        out = out.with_columns(**{f"r_{nm}": pl.when(pl.col("n_")>=3).then(pl.col("k_")/pl.col("n_")).otherwise(None), f"c_{nm}": pl.col("n_").cast(pl.Float64).log1p()}).drop("n","k","own","n_","k_")
    return out.select("MVT_ID_mvt","r_m","c_m","r_d","c_d")
pf = period_feats(src)
hist = H.airline_late(src)
d = H.with_loo_rate(H.frame25(), hist).join(pf, on="MVT_ID_mvt", how="left")
tw = set(twin_frame(src)["MVT_ID_mvt"].to_list()); d = d.with_columns(twin=pl.col("MVT_ID_mvt").is_in(list(tw)))
def design3(z, which):
    cols=[H.design2(z)]
    for nm in which:
        r = z[f"r_{nm}"].to_numpy().astype(float); has = ~np.isnan(r)
        cols += [np.nan_to_num(r, nan=0.0), has.astype(float), z[f"c_{nm}"].to_numpy().astype(float)]
    return np.column_stack(cols)
for which in ([], ["m"], ["d"], ["m","d"]):
    tot_b=tot_n=0; better=0; per=[]
    for m in sorted(d["mt"].unique().to_list()):
        tr, te = d.filter(pl.col("mt")!=m), d.filter(pl.col("mt")==m)
        c45, c2 = L.fit(tr), H.fit2(tr)
        t = L.train_rows(tr); c3 = LogisticRegression(C=1.0, max_iter=4000).fit(design3(t, which), t["echo"].to_numpy().astype(int))
        b45,_,_ = L.apply(te, te["Pb"].to_numpy(), c45)
        b48, k = H.treat(te, b45, c45, c2, te["twin"].to_numpy())
        # round 3: replace q2 by q3 with the same late-branch bound
        p = te["p"].fill_null(1.0).to_numpy(); off = te["off"].to_numpy(); offc=np.clip(off,0,L.WIDE)
        q45 = p.copy(); q3 = p.copy()
        if k.any():
            zk = te.filter(pl.Series(k)); q45[k]=c45.predict_proba(L.design(zk))[:,1]; q3[k]=c3.predict_proba(design3(zk, which))[:,1]
        with np.errstate(divide="ignore", invalid="ignore"): pne=(b45-q45*offc)/(1-q45)
        n3 = np.where(k, np.maximum(q3*offc+(1-q3)*np.minimum(pne, np.maximum(off-3606,0)),0), b45)
        y = te["y"].to_numpy(); sb=np.sum(((y-b48)**2)[k]); sn=np.sum(((y-n3)**2)[k])
        tot_b+=sb; tot_n+=sn; better+= sn<sb; per.append(sn/sb if sb>0 else 1)
    print(f"features {which or ['(v48 refit)']}: in-scope SSE v48 {tot_b:.4g} -> {tot_n:.4g} ({tot_n/tot_b:.3f}); months better {better}/12; per month {' '.join(f'{x:.2f}' for x in per)}")
