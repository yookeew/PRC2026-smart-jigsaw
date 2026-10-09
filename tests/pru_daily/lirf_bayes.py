"""LIRF: Bayesian update of hedged rows' echo probability from the published daily sum (holdout, 2025).
Hedged row: T - S >= 45 min and p strictly between tau and T - S.  Two-point model per row:
  echo -> taxi = T - S (contributes 0 to D);  real -> taxi = tau (contributes (T-S-tau)/60 min).
pi = (p - tau) / (T - S - tau).  Residual R = D - sum_nonH max(0, d_hat) - sum_H (1 - e_i)(ts_i - tau)/60,
noise sigma_N per day from the non-hedged rows' true error (cross-month).  Posterior by enumeration (<= 16 rows) else
independent Gaussian approximation."""
import os, glob, itertools
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
SCR = os.environ["SCR"]
TAU = float(os.environ.get("TAU", "1100"))
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
y, p, ts, mo = (d[c].to_numpy() for c in ("y", "p", "ts", "mo"))
dh = ts - p; dt = ts - y
H = (ts >= 2700) & (p > TAU + 60) & (p < ts - 60)
pi = np.clip((p - TAU) / (ts - TAU), 0.001, 0.999)
echo_true = np.abs(dt) <= 120
print(f"LIRF rows {len(y):,}; hedged {H.sum():,} ({H.mean():.3%}); hedged SSE share {((y - p) ** 2)[H].sum() / ((y - p) ** 2).sum():.3f}; "
      f"true echo rate among hedged {echo_true[H].mean():.3f} vs mean pi {pi[H].mean():.3f}")
key = d["day"].cast(pl.Utf8).to_numpy(); Dv = d["D"].to_numpy()
days = sorted(set(key))
# noise: non-hedged rows' true contribution error per day
eN = {k: (np.maximum(0, dt) - np.maximum(0, dh))[(key == k) & ~H].sum() / 60 for k in days}
mo_day = {k: mo[key == k][0] for k in days}
for m in (1, 7):
    v = np.array([eN[k] for k in days if mo_day[k] == m]); print(f"  month {m}: non-hedged day error mean {v.mean():+.0f} min, sd {v.std():.0f} min")
sig = {m: np.std([eN[k] for k in days if mo_day[k] != m]) for m in (1, 7)}   # cross-month
bias = {m: np.mean([eN[k] for k in days if mo_day[k] != m]) for m in (1, 7)}
q = p.copy(); post = pi.copy()
for k in days:
    m = key == k
    if not np.isfinite(Dv[m][0]): continue
    hm = np.where(m & H)[0]
    if len(hm) == 0: continue
    base = np.maximum(0, dh[m & ~H]).sum() / 60
    c = (ts[hm] - TAU) / 60                       # contribution if real
    R = Dv[m][0] - base - bias[mo_day[k]]        # what the hedged rows (+ noise) must add up to
    s = sig[mo_day[k]]
    if len(hm) <= 16:
        cfg = np.array(list(itertools.product([0, 1], repeat=len(hm))))   # 1 = echo
        lp = (cfg * np.log(pi[hm]) + (1 - cfg) * np.log(1 - pi[hm])).sum(1) - 0.5 * ((R - ((1 - cfg) * c).sum(1)) / s) ** 2
        w = np.exp(lp - lp.max()); w /= w.sum()
        post[hm] = (cfg * w[:, None]).sum(0)
    else:   # one-at-a-time Gaussian update holding others at their prior mean
        mu = ((1 - pi[hm]) * c).sum(); var = s ** 2 + (pi[hm] * (1 - pi[hm]) * c ** 2).sum()
        for j, i in enumerate(hm):
            mu_o = mu - (1 - pi[i]) * c[j]; var_o = var - pi[i] * (1 - pi[i]) * c[j] ** 2
            l1 = np.log(pi[i]) - 0.5 * (R - mu_o) ** 2 / var_o
            l0 = np.log(1 - pi[i]) - 0.5 * (R - mu_o - c[j]) ** 2 / var_o
            post[i] = 1 / (1 + np.exp(l0 - l1))
    q[hm] = post[hm] * ts[hm] + (1 - post[hm]) * TAU
r0, r1 = y - p, y - q
for name, mm in (("LIRF all", np.ones(len(y), bool)), ("Jan", mo == 1), ("Jul", mo == 7), ("hedged", H), ("NM-matched", d["nm"].to_numpy()), ("NM-unmatched", ~d["nm"].to_numpy())):
    print(f"{name:13s} RMSE {np.sqrt(np.mean(r0[mm] ** 2)):8.2f} -> {np.sqrt(np.mean(r1[mm] ** 2)):8.2f}   dSSE {((r1 ** 2) - (r0 ** 2))[mm].sum():+.3e}")
tot = 296.427 ** 2 * 344339
print(f"holdout full RMSE effect: {np.sqrt((tot + ((r1 ** 2) - (r0 ** 2)).sum()) / 344339) - 296.427:+.3f}")
print("echo calibration of posterior on hedged:", [(round(a, 1), round(float(echo_true[H][(post[H] >= a) & (post[H] < a + 0.2)].mean()), 2) if ((post[H] >= a) & (post[H] < a + 0.2)).any() else None) for a in (0, 0.2, 0.4, 0.6, 0.8)])
