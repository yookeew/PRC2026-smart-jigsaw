"""H: intra-day level from ADS-B-tracked departures (reports/intraday_trk_preregistration.md).

Gate on 2025 after G2: primary the Jan + Jul holdout chain (beta cross-month), secondary the five OOF periods
(beta leave-period-out among them).

Run:  .venv/Scripts/python.exe tests/intraday_trk.py          (H1 / H2)
      .venv/Scripts/python.exe tests/intraday_trk.py --h3     (amendment A1: H3a / H3b)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
from post import pru_recon2 as R  # noqa: E402
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402
import day_alloc_gbm as G  # noqa: E402

WIN, CLIP, K = 90 * 60, 1200.0, 10.0
H3 = "--h3" in sys.argv                     # amendment A1: signal = tiers appear/dwell, bias-corrected, clip 300 s
SIG_TIERS = ("appear", "dwell")


def adsb_pb(files) -> pl.DataFrame:
    d = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "adsb_tier", "adsb_pushback_ts"]) for f in files], how="diagonal")
    return d.filter(pl.col("adsb_tier").is_in(list(SIG_TIERS if H3 else R.TRACKED)) & pl.col("adsb_pushback_ts").is_not_null()).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_tier", pb=pl.col("adsb_pushback_ts").cast(pl.Float64)).unique("MVT_ID_mvt")


def signal(apt, t, tday, trk, w):
    """z for every row (H1, H2) and the local count n; tracked rows' own values are not used for themselves."""
    n_all = len(apt)
    z1, z2, nn = np.zeros(n_all), np.zeros(n_all), np.zeros(n_all)
    for a in np.unique(apt):
        ia = np.flatnonzero(apt == a)
        it = ia[trk[ia] & np.isfinite(w[ia])]
        if len(it) == 0:
            continue
        o = np.argsort(t[it]); tt, ww = t[it][o], w[it][o]
        cs = np.concatenate([[0.0], np.cumsum(ww)])
        iu = ia[~trk[ia]]
        lo, hi = np.searchsorted(tt, t[iu] - WIN, "left"), np.searchsorted(tt, t[iu] + WIN, "right")
        n = (hi - lo).astype(float)
        m = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), 0.0)
        dm = pl.DataFrame({"d": tday[it], "w": w[it]}).group_by("d").agg(pl.col("w").mean())
        dmap = dict(zip(dm["d"].to_list(), dm["w"].to_list()))
        d = np.array([dmap.get(v, 0.0) for v in tday[iu]])
        sh = n / (n + K)
        z1[iu], z2[iu], nn[iu] = (m - d) * sh, m * sh, n
    return z1, z2, nn


def betas(z, e, apt, m):
    return {a: float(np.sum(z[m & (apt == a)] * e[m & (apt == a)]) / max(np.sum(z[m & (apt == a)] ** 2), 1e-9)) for a in T.A8}


def main() -> None:
    x = T.frame25()
    lam, _ = L.lam_rows(x)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    in8 = np.isin(apt, T.A8)
    ok = in8 & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    trim, body = y <= 18000, ~nmu & (y <= 3600)
    X = G.inputs(x, lam0, True)
    q = p.copy()
    for pk in T.PERIODS:
        bst = G.slope_fit(X, p - y, lam0, ok & trim & (per != pk) & (np.abs(lam0) >= 1))
        k = ok & (per == pk)
        q[k] = np.maximum(p[k] - np.clip(bst.predict(X[k], raw_score=True) + G.INIT, 0, 1) * lam0[k], 0)
    print("G2 leave-period-out done", flush=True)

    pb = adsb_pb(sorted((ROOT / "cache/adsb_pushback_v4").glob("day=2025-*.parquet")))
    x = x.join(pb, on="MVT_ID_mvt", how="left", maintain_order="left")
    t = x["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(float)
    tday = x["MVT_TIME_UTC_mvt"].dt.date().cast(pl.Utf8).to_numpy()
    trk = x["pb"].is_not_null().to_numpy() & in8 & (True if H3 else x["trk"].to_numpy())
    taxi_adsb = t - x["pb"].to_numpy().astype(float)
    cl = CLIP
    if H3:                                   # per (airport, tier) offset from the other periods
        tier = x["adsb_tier"].fill_null("").to_numpy()
        bias = np.zeros(len(t))
        for pk in T.PERIODS:
            k = trk & (per == pk)
            o = trk & (per != pk)
            ref = pl.DataFrame({"a": apt[o], "t": tier[o], "e": (taxi_adsb - y)[o]}).group_by("a", "t").agg(pl.col("e").median())
            bm = {(a, tt): v for a, tt, v in ref.iter_rows()}
            bias[k] = [bm.get((a, tt), 0.0) for a, tt in zip(apt[k], tier[k])]
        taxi_adsb, cl = taxi_adsb - bias, 300.0
    w = np.where(trk, np.clip(taxi_adsb - q, -cl, cl), np.nan)
    print(f"tracked rows at the 8 airports: {trk.sum():,} ({trk[in8].mean():.1%}); ADS-B taxi vs label on them: "
          f"RMS {np.sqrt(np.nanmean((taxi_adsb[trk] - y[trk]) ** 2)):.1f} s, median |d| {np.nanmedian(np.abs(taxi_adsb[trk] - y[trk])):.0f}", flush=True)
    z1, z2, nn = signal(apt, t, tday, trk, w)
    e = y - q
    unt = in8 & ~trk & (nn >= 1)
    fitm = unt & trim
    print(f"untracked rows with >= 1 tracked neighbour in +-90 min: {unt.sum():,} of {(in8 & ~trk).sum():,}", flush=True)
    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    ch = per == "P01"
    oofp = [pk for pk in T.PERIODS if pk != "P01"]
    rows, out = [], {}
    for name, z in ((("H3a" if H3 else "H1"), z1), (("H3b" if H3 else "H2"), z2)):
        qq = q.copy()
        bj, bl = betas(z, e, apt, fitm & (mo == "2025-07")), betas(z, e, apt, fitm & (mo == "2025-01"))
        for mm, b in (("2025-01", bj), ("2025-07", bl)):
            k = unt & (mo == mm)
            qq[k] = np.maximum(q[k] + np.array([b[a] for a in apt[k]]) * z[k], 0)
        for pk in oofp:
            b = betas(z, e, apt, fitm & ~ch & (per != pk))
            k = unt & (per == pk)
            qq[k] = np.maximum(q[k] + np.array([b[a] for a in apt[k]]) * z[k], 0)
        out[name] = qq
        c8, o8 = in8 & ch, in8 & ~ch
        res = {"var": name, "chain_full": rm(qq, c8) - rm(q, c8), "chain_body": rm(qq, c8 & body) - rm(q, c8 & body),
               "Jan": rm(qq, in8 & (mo == "2025-01")) - rm(q, in8 & (mo == "2025-01")),
               "Jul": rm(qq, in8 & (mo == "2025-07")) - rm(q, in8 & (mo == "2025-07")),
               "oof_full": rm(qq, o8) - rm(q, o8), "oof_body": rm(qq, o8 & body) - rm(q, o8 & body),
               "Feb": rm(qq, in8 & (mo == "2025-02")) - rm(q, in8 & (mo == "2025-02")),
               "Jun": rm(qq, in8 & (mo == "2025-06")) - rm(q, in8 & (mo == "2025-06"))}
        res["pass"] = (res["chain_full"] <= -0.05 and res["chain_body"] <= 0 and res["Jan"] <= 0 and res["Jul"] <= 0
                       and res["oof_full"] <= 0 and res["Feb"] <= 0 and res["Jun"] <= 0)
        rows.append(res)
        print(f"{name} chain betas fit on Jul (for Jan):", {a: round(v, 3) for a, v in bj.items()})
        print(f"{name} chain betas fit on Jan (for Jul):", {a: round(v, 3) for a, v in bl.items()})
        print(f"{name} chain months per airport full:", {a: round(rm(qq, ch & (apt == a)) - rm(q, ch & (apt == a)), 2) for a in T.A8})
        print(f"{name} OOF per airport full:", {a: round(rm(qq, ~ch & (apt == a)) - rm(q, ~ch & (apt == a)), 2) for a in T.A8})
    with pl.Config(tbl_cols=20, float_precision=3, tbl_width_chars=250):
        print("vs G2 (v63's recipe), 8 airports' rows")
        print(pl.DataFrame(rows))
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    for name, qq in out.items():
        dmse = sum(n26[a] * np.mean((y[ch & (apt == a)] - qq[ch & (apt == a)]) ** 2 - (y[ch & (apt == a)] - q[ch & (apt == a)]) ** 2)
                   for a in T.A8) / sum(n26.values())
        print(f"{name} 2026 projection from the chain months: {np.sqrt(238.3173 ** 2 + dmse) - 238.3173:+.3f} s")
    print("production betas (both chain months):", {n: {a: round(v, 3) for a, v in betas(z, e, apt, fitm & ch).items()}
                                                  for n, z in ((("H3a" if H3 else "H1"), z1), (("H3b" if H3 else "H2"), z2))})


if __name__ == "__main__":
    main()
