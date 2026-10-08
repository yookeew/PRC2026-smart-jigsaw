"""Robustness of v46 (reports/lirf_echo_tail_preregistration.md): enumerate echo/+24h label assignments of the 11 rows,
weight by the 2025 prior and the v19->v20 board equation, and solve single-row exact explanations of the residual."""
import json, itertools, polars as pl, numpy as np
sc=json.load(open("cache/eval/board_scores.json")); N=344841
r = pl.read_parquet("data/ranking/ranking.parquet").filter(pl.col("PHASE_mvt")=="DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64),"FLIGHT_mvt",
      off=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
sub=lambda v: pl.read_parquet(f"data/submissions/smart-jigsaw_{v}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(v))
x = r.join(sub("v19"),on="MVT_ID_mvt").join(sub("v20"),on="MVT_ID_mvt").filter(pl.col("v19")!=pl.col("v20")).join(sub("v45"),on="MVT_ID_mvt").join(sub("v46"),on="MVT_ID_mvt")
A,B,P45,P46,off = (x[c].to_numpy() for c in ["v19","v20","v45","v46","off"]); names=x["FLIGHT_mvt"].to_list()
obs = N*(sc["v20"]**2-sc["v19"]**2)
combos = np.array(list(itertools.product(range(2), repeat=len(off))))   # 0 echo, 1 +24h
Y = np.where(combos==0, off, 87480.)
pred = ((B-Y)**2-(A-Y)**2).sum(1); gain = ((P46-Y)**2-(P45-Y)**2).sum(1)
prior = np.prod(np.where(combos==0, 0.62, 0.38), axis=1)
print(f"observed {obs:.4g}")
order = np.argsort(np.abs(pred-obs))
for i in order[:8]:
    print(f"  residual {pred[i]-obs:+.3g}  +24h rows {[names[j] for j in np.where(combos[i]==1)[0]]}  v46 dSSE {gain[i]:+.3g} (board {np.sqrt(249.0543**2+gain[i]/N):.2f})")
for sd in (1e8, 2e8, 4e8):
    w = prior*np.exp(-0.5*((pred-obs)/sd)**2); w/=w.sum()
    print(f"likelihood sd {sd:.0e}: P(v46 better) {w[gain<0].sum():.3f}; expected board {np.sum(w*np.sqrt(249.0543**2+gain/N)):.2f}; P(all echo) {w[0]:.3f}; "
          f"P(row is +24h): " + ", ".join(f"{n} {p:.2f}" for n,p in zip(names,(w[:,None]*combos).sum(0)) if p>0.05))
w = prior; print(f"prior only (ignore board): P(v46 better) {w[gain<0].sum():.3f}, expected board {np.sum(w*np.sqrt(249.0543**2+gain/N)):.2f}")
print("\nexact single-row explanations of the residual (other 10 rows echo):")
d = B-A
base = ((B-off)**2-(A-off)**2).sum()
for i,n in enumerate(names):
    # pred(y_i) = base - 2 d_i (y_i - off_i)  = obs
    yi = off[i] - (obs-base)/(2*d[i])
    Y = off.copy(); Y[i] = yi
    g = ((P46-Y)**2-(P45-Y)**2).sum()
    print(f"  {n:10s} off {off[i]:8.0f}  needs y {yi:9.0f} ({yi-off[i]:+7.0f} vs echo)  -> v46 dSSE {g:+.3g} (board {np.sqrt(249.0543**2+g/N):.2f})")
