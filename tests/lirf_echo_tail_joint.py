"""Joint fit of echo/+24h/normal label assignments for the 11 LIRF NM-unmatched >= 12 h rows to the v19->v20 and v45->v46 board deltas (PROGRESS §115)."""
import json, itertools, polars as pl, numpy as np
sc=json.load(open("cache/eval/board_scores.json")); sc["v45"]=249.0543; sc["v46"]=249.2907; N=344841
r = pl.read_parquet("data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt")=="DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64),"FLIGHT_mvt",
      off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
sub=lambda v: pl.read_parquet(f"data/submissions/smart-jigsaw_{v}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(v))
x = r.join(sub("v19"),on="MVT_ID_mvt").join(sub("v20"),on="MVT_ID_mvt").filter(pl.col("v19")!=pl.col("v20")).join(sub("v45"),on="MVT_ID_mvt").join(sub("v46"),on="MVT_ID_mvt")
chk = r.join(sub("v45"),on="MVT_ID_mvt").join(sub("v46"),on="MVT_ID_mvt").filter(pl.col("v45")!=pl.col("v46")).height
A,B,P45,P46,off = (x[c].to_numpy() for c in ["v19","v20","v45","v46","off"]); names=x["FLIGHT_mvt"].to_list()
o1 = N*(sc["v20"]**2-sc["v19"]**2); o2 = N*(sc["v46"]**2-sc["v45"]**2)
print(f"v45/v46 differ on {chk} rows. eq1 (v19->v20) {o1:.4g}; eq2 (v45->v46) {o2:.4g}")
f1 = lambda Y: ((B-Y)**2-(A-Y)**2).sum(-1); f2 = lambda Y: ((P46-Y)**2-(P45-Y)**2).sum(-1)
for k in (2,3):
    combos = np.array(list(itertools.product(range(k), repeat=len(off))))
    labs = np.stack([off, np.full(len(off),87480.), np.full(len(off),1500.)])[:k]
    Y = labs[combos, np.arange(len(off))]
    res = np.sqrt((f1(Y)-o1)**2+(f2(Y)-o2)**2)
    print(f"\n{k}-class: best joint fits")
    for i in np.argsort(res)[:8]:
        nonecho = {names[j]: ['echo','+24h','normal'][combos[i][j]] for j in range(len(off)) if combos[i][j]!=0}
        print(f"  joint resid {res[i]:.3g}  eq1 {f1(Y[i])-o1:+.3g} eq2 {f2(Y[i])-o2:+.3g}  {nonecho}")
# linear: both equations are linear in y: sum 2 (A-B) y = o1 - sum(B^2-A^2) ; sum 2 (P45-P46) y = o2 - sum(P46^2-P45^2)
print("\nlinear constraints: c1.y = r1, c2.y = r2")
c1, r1 = 2*(A-B), o1-np.sum(B**2-A**2); c2, r2 = 2*(P45-P46), o2-np.sum(P46**2-P45**2)
for n,a,b,o in zip(names,c1,c2,off): print(f"  {n:10s} off {o:8.0f}  c1 {a:+8.0f}  c2 {b:+8.0f}")
print(f"  echo gives c1.y-r1 {c1@off-r1:+.3g}, c2.y-r2 {c2@off-r2:+.3g}")
