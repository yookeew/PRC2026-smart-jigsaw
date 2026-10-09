"""Day-level reconciliation with EUROCONTROL's daily all-cause pre-departure delay (holdout test, 2025 data).
Published per airport x off-block date: D = sum max(0, AOBT - SOBT) [min]. Our d_hat = (T - S) - p.
Rows assigned to a day by predicted off-block (T - p), as in 2026.  For each airport-day solve lambda with
sum max(0, d_hat + lambda * w) = D, then p' = (T - S) - (d_hat + lambda * w).
Weights (cross-month: estimated on one holdout month, applied to the other):
  const : w = 1                         (constant shift)
  var   : w = sigma^2 of the cell        (cell = airport x NM x pred band x d_hat band)
  varp  : w = sigma^2 * P(d > 0 | cell)"""
import os, glob, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from scipy.optimize import brentq
SCR = os.environ["SCR"]
files = [f for f in sorted(glob.glob("data/raw/training_*.parquet")) if "2025-01-01" in f or "2025-07-01" in f]
raw = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
h = pl.read_parquet(f"{SCR}/h2/chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), ph="pred")
l = pl.read_parquet(f"{SCR}/v48m_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl_="v48")
d = (raw.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(h, on="MVT_ID_mvt").join(l, on="MVT_ID_mvt")
     .filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
     .with_columns(p=pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.col("pl_")).otherwise(pl.col("ph")),
                   y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), apt=pl.col("ADEP_mvt"), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                   ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                   nm=pl.col("FLIGHT_ID_mvt").is_not_null()))
d = d.with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date(),
                   dh=pl.col("ts") - pl.col("p"))
d = d.with_columns(pband=pl.col("p").cut([600, 900, 1200, 1800, 3600, 7200]).cast(pl.Utf8),
                   dband=pl.col("dh").cut([-600, 0, 600, 1800, 3600, 10800]).cast(pl.Utf8),
                   dtrue=pl.col("ts") - pl.col("y"))
d = d.with_columns(cell=pl.concat_str(["apt", pl.col("nm").cast(pl.Utf8), "pband", "dband"], separator="|"))
pub = pl.read_csv(f"{SCR}/ext/all_pre_departure_delays_2025.csv", infer_schema_length=0).select(
    apt="APT_ICAO", day=pl.col("FLT_DATE").str.to_date(), D=pl.col("DLY_ALL_PRE_2").cast(pl.Float64))
d = d.join(pub, on=["apt", "day"], how="left").with_row_index("i")
# cross-month cell stats
st = d.group_by("cell", "mo").agg(var=((pl.col("y") - pl.col("p")) ** 2).mean(), ppos=(pl.col("dtrue") > 0).mean(), n=pl.len())
other = st.with_columns(mo=pl.when(pl.col("mo") == 1).then(7).otherwise(1))
d = d.join(other.select("cell", "mo", var_o="var", ppos_o="ppos", n_o="n"), on=["cell", "mo"], how="left")
apt_var = d.group_by("apt", "mo").agg(av=((pl.col("y") - pl.col("p")) ** 2).mean()).with_columns(mo=pl.when(pl.col("mo") == 1).then(7).otherwise(1))
d = d.join(apt_var, on=["apt", "mo"], how="left").with_columns(
    var_o=pl.when(pl.col("n_o").fill_null(0) >= 30).then(pl.col("var_o")).otherwise(pl.col("av")),
    ppos_o=pl.when(pl.col("n_o").fill_null(0) >= 30).then(pl.col("ppos_o")).otherwise(0.5))
y, p, ts, dh = (d[c].to_numpy() for c in ("y", "p", "ts", "dh"))
key = (d["apt"] + "_" + d["day"].cast(pl.Utf8)).to_numpy(); Dv = d["D"].to_numpy()
apt, mo, nm = d["apt"].to_numpy(), d["mo"].to_numpy(), d["nm"].to_numpy()
W = {"const": np.ones(len(y)), "var": d["var_o"].to_numpy(), "varp": d["var_o"].to_numpy() * d["ppos_o"].to_numpy()}
uk, inv = np.unique(key, return_inverse=True)
groups = [np.where(inv == g)[0] for g in range(len(uk))]


def recon(w, skip=()):
    dn = dh.copy()
    for m in groups:
        a = apt[m[0]]
        Dk = Dv[m[0]]
        if not np.isfinite(Dk) or a in skip:
            continue
        ww = w[m] / w[m].mean()
        f = lambda lam: np.maximum(0.0, dh[m] + lam * ww).sum() / 60.0 - Dk
        lo, hi = -5e4, 5e4
        if f(lo) * f(hi) > 0:
            continue
        lam = brentq(f, lo, hi, xtol=0.01)
        dn[m] = dh[m] + lam * ww
    return np.maximum(ts - dn, 0.0)


def rep(name, q):
    r0, r1 = y - p, y - q
    body = nm & (y <= 3600)
    f = lambda m: (np.sqrt(np.mean(r0[m] ** 2)), np.sqrt(np.mean(r1[m] ** 2)))
    out = [f"{name:12s}"]
    for lab, m in (("full", np.ones(len(y), bool)), ("body", body), ("bodyJan", body & (mo == 1)), ("bodyJul", body & (mo == 7)),
                   ("fullJan", mo == 1), ("fullJul", mo == 7)):
        a, b = f(m); out.append(f"{lab} {b - a:+.3f}")
    print("  ".join(out))
    print("     per apt:", {a: round(f(apt == a)[1] - f(apt == a)[0], 2) for a in sorted(set(apt))})


print(f"rows {len(y):,}  base full {np.sqrt(np.mean((y - p) ** 2)):.3f}")
for k, w in W.items():
    rep(k, recon(w))
rep("varp_noLIRF", recon(W["varp"], skip=("LIRF",)))
