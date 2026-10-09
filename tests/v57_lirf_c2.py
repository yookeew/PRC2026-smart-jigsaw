"""v59 candidate C2 (reports/v57_daysum_model_preregistration.md): LIRF ordinary-row day shift, no per-row bets.

Ordinary = NM-matched, not hedged (src/post/pru_daily_recon.hedged), T - S < 12 h. Per LIRF day:
  sum_ordinary G(d_hat + lam) = 60 D - sum_hedged (1 - pi) max(0, T - S - N) - sum_other(<12 h) max(0, d_hat)   (>= 12 h rows: 0)
pi, N: section 125's prior (isotonic map, 650 s floor). Shift on ordinary rows: mu_month + a * u_g (F0, EB a).
2025 validation (decides): periods as tests/v57_dsm.py; LIRF predictions = v48-method holdout chain for (Jan, Jul),
corrector OOF elsewhere. Writes the 2026 shift to cache/v57/c2_delta_2026.parquet if PASS.

Run:  python tests/v57_lirf_c2.py
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
from post import pru_recon2 as R  # noqa: E402
from post import pru_daily_recon as P  # noqa: E402
import v57_dsm as T  # noqa: E402


def lirf25() -> pl.DataFrame:
    x = T.frame25(keep_lirf=True).filter(pl.col("apt") == "LIRF")
    ch = pl.read_parquet(ROOT / "cache/v54/v48m_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p48=pl.col("v48"))
    x = x.join(ch, on="MVT_ID_mvt", how="left").with_columns(
        p=pl.when(pl.col("period") == "P01").then(pl.col("p48")).otherwise(pl.col("p")).clip(0, None))
    # day assignment by the (possibly replaced) prediction
    x = x.drop("D", "nD").with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
    return x.join(T.DV.pub(2025), on=["apt", "day"], how="left")


def masks(x: pl.DataFrame):
    p, ts = x["p"].to_numpy().astype(float), x["ts"].to_numpy().astype(float)
    nmu = x["nmu"].to_numpy().astype(bool)
    H = P.hedged(p, ts)
    long = ts >= 12 * 3600
    ordy = ~nmu & ~H & ~long
    other = ~H & ~long & ~ordy
    return H, long, ordy, other


def prior(iso, p, ts, nmu):
    pi = iso.predict(P.ratio(p, ts) + 10 * nmu).clip(0.002, 0.998)
    N = (p - pi * ts) / (1 - pi)
    fl = N < P.N_MIN
    N = np.where(fl, P.N_MIN, N)
    pi = np.where(fl, (p - P.N_MIN) / np.maximum(ts - P.N_MIN, 1.0), pi).clip(0.002, 0.998)
    return pi, N


def calib_frame(x):
    return x.select("y", "p", "ts", "nmu", "day")


def solve_days(x: pl.DataFrame, iso, dist) -> list[dict]:
    p, ts = x["p"].to_numpy().astype(float), x["ts"].to_numpy().astype(float)
    nmu = x["nmu"].to_numpy().astype(bool)
    H, long, ordy, other = masks(x)
    Dv = x["D"].to_numpy().astype(float)
    mo = x["mo"].to_numpy()
    out = []
    for g in R.groups_of(x):
        go = g[ordy[g]]
        if len(go) < 20:
            continue
        gh, gx = g[H[g]], g[other[g]]
        fixed = 0.0
        if len(gh):
            pi, N = prior(iso, p[gh], ts[gh], nmu[gh])
            fixed += float(np.sum((1 - pi) * np.maximum(0, ts[gh] - N)))
        fixed += float(np.sum(np.maximum(0, ts[gx] - p[gx])))
        dh = ts[go] - p[go]
        lam = R.solve(dh, np.ones(len(go)), np.zeros(len(go), int), [dist], Dv[g[0]] * 60.0 - fixed)
        if np.isfinite(lam):
            out.append({"per": mo[g[0]], "go": go, "lam": lam})
    return out


def f0_shift(n, days, s2):
    pers = sorted(set(d["per"] for d in days))
    lam = np.array([d["lam"] for d in days])
    mu = {m: lam[[d["per"] == m for d in days]].mean() for m in pers}
    u = lam - np.array([mu[d["per"]] for d in days])
    vu = float(np.sum(u ** 2) / max(len(u) - len(pers), 1))
    a = max(0.0, 1 - s2 / vu) if vu > 0 else 0.0
    delta = np.zeros(n)
    for d, ug in zip(days, u):
        delta[d["go"]] = mu[d["per"]] + a * ug
    return delta, a, mu


def main():
    x = lirf25()
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    mo, per = x["mo"].to_numpy(), x["period"].to_numpy()
    H, long, ordy, other = masks(x)
    print(f"LIRF 2025 rows {x.height:,}: ordinary {ordy.mean():.3f}, hedged {H.mean():.3f}, >=12 h {long.mean():.4f}, other {other.mean():.3f}")
    ho = x.filter(pl.col("period") == "P01")
    days, cals = [], {}
    for pk, (m1, m2) in T.PERIODS.items():
        for m in (m1, m2):
            mk = mo == m
            if pk == "P01":
                c = ho.filter(pl.col("mo") != m)
                cal_eta = c
            else:
                c = ho
                cal_eta = x.filter(~pl.col("period").is_in(["P01", pk]))
            iso = P.calibrate(calib_frame(c))[0]
            ce = cal_eta.filter(pl.Series(masks(cal_eta)[2]))
            eta = R.eta_sample(ce["y"].to_numpy(), ce["p"].to_numpy(), ce["day"].cast(pl.Utf8).to_numpy())
            xm = x.filter(pl.Series(mk)); idx = np.flatnonzero(mk)
            for d in solve_days(xm, iso, R.EtaDist(eta)):
                d["go"] = idx[d["go"]]; d["period"] = pk
                d["e"] = d["lam"] - float(np.mean(p[d["go"]] - y[d["go"]]))
                days.append(d)
    nz = pl.DataFrame({"period": [d["period"] for d in days], "per": [d["per"] for d in days], "e": [d["e"] for d in days]})
    nz = nz.with_columns(ec=pl.col("e") - pl.col("e").mean().over("period"))
    print(f"solved LIRF days {len(days)}; 2025 noise sd {float(np.sqrt((nz['ec'] ** 2).mean())):.1f} s; mean(lam - delta*) by period "
          f"{dict(nz.group_by('period').agg(pl.col('e').mean()).sort('period').iter_rows())}")
    q = p.copy()
    for pk in T.PERIODS:
        s2 = float((nz.filter(pl.col("period") != pk)["ec"] ** 2).mean())
        dp = [d for d in days if d["period"] == pk]
        delta, a, mu = f0_shift(x.height, dp, s2)
        k = delta != 0
        q[k] = np.maximum(p[k] - delta[k], 0)
        print(f"  {pk}: a {a:.2f}, month levels {({m: round(v, 1) for m, v in mu.items()})}")
    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    body = ordy & (y <= 3600)
    months = sorted(set(mo))
    res = {"full": rm(q, ordy) - rm(p, ordy), "body": rm(q, body) - rm(p, body),
           "months_full_better": int(sum(rm(q, ordy & (mo == m)) < rm(p, ordy & (mo == m)) for m in months))}
    print("C2 (LIRF ordinary rows, 2025, vs no change):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    print("  full by month:", {m[5:]: round(rm(q, ordy & (mo == m)) - rm(p, ordy & (mo == m)), 2) for m in months})
    print("  LIRF all rows full:", round(rm(q, np.ones(len(y), bool)) - rm(p, np.ones(len(y), bool)), 3))
    gate = [res["full"] <= -0.10, res["body"] <= 0, res["months_full_better"] >= 8]
    print(f"  C2 GATE (full<=-0.10, body<=0, >=8/12): {gate} -> {'PASS' if all(gate) else 'FAIL'}")
    if not all(gate):
        return
    # ---- 2026 shift (production): calibration and eta from the whole holdout chain, s2 from all 2025 periods
    import v57_build as B
    x26 = B.ranking_frame().filter(pl.col("apt") == "LIRF").with_columns(nmu=pl.col("FLIGHT_ID_mvt").is_null(), p=pl.col("v55"))
    x26 = x26.drop("D", "nD").with_columns(day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date())
    x26 = x26.join(T.DV.pub(2026), on=["apt", "day"], how="left")
    iso = P.calibrate(calib_frame(ho))[0]
    ce = ho.filter(pl.Series(masks(ho)[2]))
    dist = R.EtaDist(R.eta_sample(ce["y"].to_numpy(), ce["p"].to_numpy(), ce["day"].cast(pl.Utf8).to_numpy()))
    d26 = solve_days(x26, iso, dist)
    delta, a, mu = f0_shift(x26.height, d26, float((nz["ec"] ** 2).mean()))
    print(f"2026 LIRF: {len(d26)} days solved, a {a:.2f}, month levels {({m: round(v, 1) for m, v in mu.items()})}, "
          f"ordinary rows {masks(x26)[2].sum():,}, moved {(delta != 0).sum():,}")
    pl.DataFrame({"MVT_ID_mvt": x26["MVT_ID_mvt"], "delta": delta}).write_parquet(ROOT / "cache/v57/c2_delta_2026.parquet")


if __name__ == "__main__":
    main()
