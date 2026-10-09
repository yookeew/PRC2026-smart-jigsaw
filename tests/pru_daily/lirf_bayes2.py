"""Calibrated version: prior pi_cal from a cross-month isotonic map of pi_raw -> echo; non-echo value N = (p - pi_cal*E)/(1-pi_cal);
posterior from the day sum; q = post*E + (1-post)*N (q == p when post == pi_cal)."""
import os, glob, itertools
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
SCR = os.environ["SCR"]
TAU = float(os.environ.get("TAU", "1100")); ONLY_NMU = os.environ.get("ONLY_NMU") == "1"; TEMP = float(os.environ.get("NOISE_X", "1"))
files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
raw = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
l = pl.read_parquet(f"{SCR}/v48m_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p="v48")
d = (raw.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(l, on="MVT_ID_mvt").filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
     .with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                   ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                   nm=pl.col("FLIGHT_ID_mvt").is_not_null()))
d = d.with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
pub = pl.read_csv(f"{SCR}/ext/all_pre_departure_delays_2025.csv", infer_schema_length=0).filter(pl.col("APT_ICAO") == "LIRF").select(
    day=pl.col("FLT_DATE").str.to_date(), D=pl.col("DLY_ALL_PRE_2").cast(pl.Float64))
d = d.join(pub, on="day", how="left")
y, p, ts, mo = (d[c].to_numpy() for c in ("y", "p", "ts", "mo")); nmu = ~d["nm"].to_numpy()
dh = ts - p; dt = ts - y
H = (ts >= 2700) & (p > TAU + 60) & (p < ts - 60)
if ONLY_NMU: H &= nmu
echo_true = np.abs(dt) <= 120
pr = np.clip((p - TAU) / (ts - TAU), 0, 1)
pi = np.zeros(len(y))
for m in (1, 7):
    tr, te = H & (mo != m), H & (mo == m)
    feats = lambda k: pr[k] + 10 * nmu[k]          # separate maps for NM-matched / unmatched
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998).fit(feats(tr), echo_true[tr])
    pi[te] = iso.predict(feats(te))
pi = np.clip(pi, 0.002, 0.998)
E = ts; N = (p - pi * E) / (1 - pi)
print(f"hedged {H.sum():,}; mean pi_cal {pi[H].mean():.3f} vs true {echo_true[H].mean():.3f}; N median {np.median(N[H]):.0f}s, "
      f"N<0 share {(N[H] < 0).mean():.3f}")
N = np.maximum(N, 0)
key = d["day"].cast(pl.Utf8).to_numpy(); Dv = d["D"].to_numpy(); days = sorted(set(key))
eN = {k: (np.maximum(0, dt) - np.maximum(0, dh))[(key == k) & ~H].sum() / 60 for k in days}
mo_day = {k: mo[key == k][0] for k in days}
sig = {m: TEMP * np.std([eN[k] for k in days if mo_day[k] != m]) for m in (1, 7)}
bias = {m: np.mean([eN[k] for k in days if mo_day[k] != m]) for m in (1, 7)}
post = pi.copy(); q = p.copy()
for k in days:
    m = key == k
    if not np.isfinite(Dv[m][0]): continue
    hm = np.where(m & H)[0]
    if len(hm) == 0: continue
    base = np.maximum(0, dh[m & ~H]).sum() / 60
    c = np.maximum(ts[hm] - N[hm], 0) / 60
    R = Dv[m][0] - base - bias[mo_day[k]]; s = sig[mo_day[k]]
    if len(hm) <= 16:
        cfg = np.array(list(itertools.product([0, 1], repeat=len(hm))))
        lp = (cfg * np.log(pi[hm]) + (1 - cfg) * np.log(1 - pi[hm])).sum(1) - 0.5 * ((R - ((1 - cfg) * c).sum(1)) / s) ** 2
        w = np.exp(lp - lp.max()); w /= w.sum(); post[hm] = (cfg * w[:, None]).sum(0)
    else:
        mu = ((1 - pi[hm]) * c).sum(); var = s ** 2 + (pi[hm] * (1 - pi[hm]) * c ** 2).sum()
        for j, i in enumerate(hm):
            mu_o = mu - (1 - pi[i]) * c[j]; var_o = var - pi[i] * (1 - pi[i]) * c[j] ** 2
            l1 = np.log(pi[i]) - 0.5 * (R - mu_o) ** 2 / var_o; l0 = np.log(1 - pi[i]) - 0.5 * (R - mu_o - c[j]) ** 2 / var_o
            post[i] = 1 / (1 + np.exp(l0 - l1))
    q[hm] = post[hm] * E[hm] + (1 - post[hm]) * N[hm]
r0, r1 = y - p, y - q
for name, mm in (("LIRF all", np.ones(len(y), bool)), ("Jan", mo == 1), ("Jul", mo == 7), ("NM-matched", ~nmu), ("NM-unmatched", nmu)):
    print(f"{name:13s} RMSE {np.sqrt(np.mean(r0[mm] ** 2)):8.2f} -> {np.sqrt(np.mean(r1[mm] ** 2)):8.2f}   dSSE {((r1 ** 2) - (r0 ** 2))[mm].sum():+.3e}")
tot = 296.427 ** 2 * 344339
print(f"holdout full RMSE effect: {np.sqrt((tot + ((r1 ** 2) - (r0 ** 2)).sum()) / 344339) - 296.427:+.3f}")
print("posterior calibration:", [(a, round(float(echo_true[H][(post[H] >= a) & (post[H] < a + 0.2)].mean()), 2), int(((post[H] >= a) & (post[H] < a + 0.2)).sum())) for a in (0, 0.2, 0.4, 0.6, 0.8)])
np.save(f"{SCR}/lirf_bayes_q.npy", np.column_stack([d["MVT_ID_mvt"].to_numpy(), q]))
