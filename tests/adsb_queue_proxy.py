"""Step 2b of reports/adsb_queue_scoping.md: label-free ADS-B proxy of the O1 quantities on the OOF ADS-B days.

Same rows, folds and residual model as tests/adsb_queue_oracle.py. Proxies use the v4 detector's first surface
sample of OTHER matched departures (adsb_first_ts) -- no label of any flight.
Arms: C' = C + own (T - first_ts) - s;  P = C' + proxies;  O1' = C' + oracle O1;  PO = C' + proxies + oracle O1.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_queue_oracle as Q  # noqa: E402

ROOT = Q.ROOT
XS = Q.XS


def proxy_feats(g: pl.DataFrame) -> dict[str, np.ndarray]:
    n = g.height
    T, s = g["T"].to_numpy(), g["s"].to_numpy()
    F = g["adsb_first_ts"].to_numpy().astype(float)
    mt = ~np.isnan(F) & (F <= T)
    Fm = np.where(mt, F, np.inf)
    rw = g["RUNWAY_mvt"].fill_null("?").to_numpy()
    Ls = np.where(mt, np.clip(T - np.where(mt, F, T), 0, 3600) - s, np.nan)
    f = {}
    lo, hi = np.searchsorted(T, T - 3600), np.searchsorted(T, T + 3600, side="right")
    cm = np.concatenate([[0], np.cumsum(mt)])
    m = (cm[hi] - cm[lo]) / np.maximum(hi - lo, 1)
    f["pm_rate"] = m
    Fs, Tms = np.sort(Fm), np.sort(T[mt])
    for x in XS:
        t = T - x
        q = np.searchsorted(Fs, t, side="right") - np.searchsorted(Tms, t, side="right")
        q = q - (mt & (Fm <= t) & (T > t))
        f[f"pv_apt_{x}"] = q.astype(float)
        f[f"pvn_apt_{x}"] = q / np.maximum(m, 0.05)
    for k in ("pv_rwy", "pva_rwy", "pvn_rwy", "pvna_rwy"):
        for x in XS:
            f[f"{k}_{x}"] = np.full(n, np.nan)
    for k in ("pL_prev3", "pL_next3", "pL_prev1", "pL_next1"):
        f[k] = np.full(n, np.nan)
    for rr in np.unique(rw):
        ix = np.flatnonzero(rw == rr)
        Tr, Fr, mr = T[ix], Fm[ix], mt[ix]
        Frs, Tmr = np.sort(Fr), np.sort(Tr[mr])
        for x in XS:
            t = Tr - x
            q = np.searchsorted(Frs, t, side="right") - np.searchsorted(Tmr, t, side="right")
            q = q - (mr & (Fr <= t) & (Tr > t))
            f[f"pv_rwy_{x}"][ix] = q
            f[f"pvn_rwy_{x}"][ix] = q / np.maximum(m[ix], 0.05)
            a, b = np.searchsorted(Tr, t, side="right"), np.searchsorted(Tr, Tr, side="left")
            qa = np.zeros(len(ix))
            for j in np.flatnonzero(b > a):
                qa[j] = np.count_nonzero(Fr[a[j]:b[j]] <= t[j])
            f[f"pva_rwy_{x}"][ix] = qa
            f[f"pvna_rwy_{x}"][ix] = qa / np.maximum(m[ix], 0.05)
        im = ix[mr]
        if len(im) == 0:
            continue
        Tm_, Lm = T[im], Ls[im]
        p0 = np.searchsorted(Tm_, Tr, side="left")
        p1 = np.searchsorted(Tm_, Tr, side="right")
        P = np.full((len(ix), 3), np.nan); N = np.full((len(ix), 3), np.nan)
        for k in (1, 2, 3):
            pp = p0 - k; ok = pp >= 0
            P[ok, k - 1] = Lm[pp[ok]]
            nn = p1 + k - 1; ok = nn < len(im)
            N[ok, k - 1] = Lm[nn[ok]]
        with np.errstate(all="ignore"):
            f["pL_prev3"][ix] = np.nanmean(P, axis=1); f["pL_next3"][ix] = np.nanmean(N, axis=1)
        f["pL_prev1"][ix] = P[:, 0]; f["pL_next1"][ix] = N[:, 0]
    return f


P_FEATS = (["pm_rate"] + [f"{k}_{x}" for k in ("pv_apt", "pvn_apt", "pv_rwy", "pvn_rwy", "pva_rwy", "pvna_rwy") for x in XS]
           + ["pL_prev3", "pL_next3", "pL_prev1", "pL_next1"])


def main():
    mv = Q.load()
    gd = sorted(Q.good_days())
    det = pl.concat([pl.read_parquet(Q.V4 / f"day={d}.parquet") for d in gd]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_first_ts")
    mv = mv.join(det, on="MVT_ID_mvt", how="left").sort("ADEP_mvt", "T")
    parts = []
    for (ap,), g in mv.group_by("ADEP_mvt", maintain_order=True):
        if ap == "LIRF":
            continue
        f = Q.group_feats(g)
        f.update(proxy_feats(g))
        parts.append(g.with_columns(**{k: pl.Series(v) for k, v in f.items()}))
    df = pl.concat(parts).filter(pl.col("adsb_day"))
    apts = sorted(df["ADEP_mvt"].unique().to_list())
    df = df.with_columns(apt=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(apts)}, return_dtype=pl.Int32),
                         a3s=pl.col("a3") - pl.col("s"),
                         adsbs=pl.when(pl.col("adsb_tier").is_in(Q.OBS)).then(pl.col("T") - pl.col("adsb_pushback_ts") - pl.col("s")),
                         ownL=pl.when(pl.col("adsb_first_ts") <= pl.col("T")).then(pl.col("T") - pl.col("adsb_first_ts") - pl.col("s")))
    print(f"eval rows {df.height:,}; proxy null share pL_prev3 {df['pL_prev3'].is_nan().mean():.3f}; "
          f"matched rate {df['pm_rate'].mean():.3f}", flush=True)
    Cp = Q.C_FEATS + ["ownL"]
    noN = [c for c in P_FEATS if not c.startswith("pvn")]
    arms = {"C'": Cp, "P": Cp + P_FEATS, "Praw": Cp + noN, "O1'": Cp + Q.O1_FEATS, "PO": Cp + P_FEATS + Q.O1_FEATS}
    y, s, ym = df["y"].to_numpy(), df["s"].to_numpy(), df["ym"].to_numpy()
    pred = {a: np.full(df.height, np.nan) for a in arms}
    tr_ok = ~np.isnan(y) & (y <= Q.TRIM)
    for mo in Q.MONTHS:
        te = ym == mo
        for a, cols in arms.items():
            X = df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
            ds = lgb.Dataset(X[tr_ok & ~te], (y - s)[tr_ok & ~te], categorical_feature=[cols.index("apt")])
            pred[a][te] = s[te] + lgb.train(Q.PARAMS, ds, Q.ROUNDS).predict(X[te])
        print(f"fold {mo} done", flush=True)
    nmu, apt = df["nmu"].to_numpy(), df["ADEP_mvt"].to_numpy()
    obs = df["adsb_tier"].is_in(Q.OBS).fill_null(False).to_numpy()
    ok = ~np.isnan(y)
    body = ok & ~nmu & (y <= 3600)
    R = lambda p, k: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))  # noqa: E731
    names = list(arms)
    print("\nRMSE on held-out months; deltas vs C'")
    print(f"{'slice':26s} " + " ".join(f"{a:>8s}" for a in names))
    for nme, k in [("full", ok), ("trimmed", ok & (y <= Q.TRIM)), ("body", body), ("body, no own pushback", body & ~obs),
                   ("NM-matched 30-60 min", ok & ~nmu & (y > 1800) & (y <= 3600))]:
        c = R(pred["C'"], k)
        print(f"{nme:26s} {c:8.2f} " + " ".join(f"{R(pred[a], k) - c:+8.2f}" for a in names[1:]))
    print("\nbody by month (P-C', Praw-C', O1'-C', PO-C'):")
    for mo in Q.MONTHS:
        k = body & (ym == mo); c = R(pred["C'"], k)
        print(f"  {mo}: C' {c:7.2f} " + " ".join(f"{R(pred[a], k) - c:+6.2f}" for a in names[1:]))
    print("\nbody by airport (C', P-C', Praw-C', O1'-C', PO-C'):")
    for a in apts:
        k = body & (apt == a); c = R(pred["C'"], k)
        print(f"  {a}: {c:7.2f} " + " ".join(f"{R(pred[b], k) - c:+6.2f}" for b in names[1:]))
    out = pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], **{f"p_{a}": pred[a] for a in arms}})
    out.write_parquet(ROOT / "cache" / "analysis" / "adsb_queue_proxy.parquet")


if __name__ == "__main__":
    main()
