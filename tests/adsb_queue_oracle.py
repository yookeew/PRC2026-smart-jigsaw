"""Step 2 of reports/adsb_queue_scoping.md: oracle bound for features built from OTHER aircraft's surface state.

10 OOF months, the 115 OOF ADS-B days (v4 detections), non-LIRF departures, no holdout row read. Residual
r = y - s with s = NNLS(ctxc1sel OOF, cat_mixed OOF). LightGBM Huber residual models, leave-one-month-out.
  C  : s, airport, hour, T - SCHED, (T - AOBT_3) - s, own ADS-B taxi - s (observed tiers), same-runway takeoffs
       in +-15 min, mean NM pseudo-residual of same-runway departures in +-15 min, and the TRUE residual mean/count
       of same-airport departures in +-30 min whose pushback v4 observed (oracle nbq).
  O1 : C + surface occupancy from other flights' TRUE off-blocks (taxiing at T - x, airport / own runway / ahead).
  O2 : O1 + TRUE residuals of the 3 previous and 3 next departures on the own runway.
Own label never enters a feature (own occupancy contribution removed explicitly; asserted on a sample).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
RAW, OOF, V4 = ROOT / "data" / "raw", ROOT / "cache" / "oof", ROOT / "cache" / "adsb_pushback_v4"
MONTHS = ("2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12")
OBS = ["dwell", "pass", "appear"]
XS = (0, 300, 600, 900, 1200, 1800)
TRIM = 18000


def good_days() -> set[str]:
    """adsb_combiner_12m_test.good_days() with MONTHS = the 10 OOF months (thin-day rule, label-free)."""
    v4 = {p.name[4:14] for p in V4.glob("day=2025-*.parquet") if p.name[4:11] in MONTHS}
    man = pl.read_csv(ROOT / "external-data" / "adsb-restofyear" / "manifest.csv").with_row_index("_i").sort("_i")
    m = man.group_by("day").last().filter(pl.col("day").is_in(list(v4)))
    m = m.with_columns(med=pl.col("points").filter(pl.col("status").is_in(["ok", "tolerant"]))
                       .median().over(pl.col("day").str.slice(0, 7)))
    keep = m.filter(pl.col("status").is_in(["ok", "tolerant"]) & (pl.col("points") >= 0.5 * pl.col("med")))
    return set(keep["day"].to_list())


def load() -> pl.DataFrame:
    paths = [str(p) for p in sorted(RAW.glob("training_2025-*.parquet")) if p.name[9:16] in MONTHS]
    mv = (pl.scan_parquet(paths).filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "RUNWAY_mvt",
        y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
        T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0,
        off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        a3=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
        nmu=pl.col("FLIGHT_ID_mvt").is_null(), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
        day=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d"), hour=pl.col("MVT_TIME_UTC_mvt").dt.hour())
          .collect())
    pr = None
    for e, d in (("lgb", "ctxc1sel"), ("cat", "cat_mixed")):
        s = pl.read_parquet(OOF / d / "*.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias(e))
        pr = s if pr is None else pr.join(s, on="MVT_ID_mvt")
    mv = mv.join(pr, on="MVT_ID_mvt", how="inner")
    gd = sorted(good_days())
    det = pl.concat([pl.read_parquet(V4 / f"day={d}.parquet") for d in gd]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_tier", "adsb_pushback_ts")
    mv = mv.join(det, on="MVT_ID_mvt", how="left").with_columns(adsb_day=pl.col("day").is_in(gd))
    k = mv.filter(pl.col("adsb_day") & pl.col("y").is_not_null())
    w, _ = nnls(k.select("lgb", "cat").to_numpy(), k["y"].to_numpy())
    print(f"rows {mv.height:,}; ADS-B days {len(gd)}; OOF stack w {np.round(w, 3)}", flush=True)
    return mv.with_columns(s=pl.Series(mv.select("lgb", "cat").to_numpy() @ w)).sort("ADEP_mvt", "T")


def group_feats(g: pl.DataFrame) -> dict[str, np.ndarray]:
    """Features for one airport (all its departures in the 10 months, sorted by T)."""
    n = g.height
    T, y, s = g["T"].to_numpy(), g["y"].to_numpy(), g["s"].to_numpy()
    ok = ~np.isnan(y)
    B = np.where(ok, T - np.nan_to_num(y), np.inf)             # unknown label: never counted as taxiing
    r = np.clip(np.where(ok, y - s, np.nan), -3600, 3600)
    p_nm = np.clip(g["a3"].to_numpy() - s, -3600, 3600)
    obs = g["adsb_tier"].is_in(OBS).fill_null(False).to_numpy()
    rw = g["RUNWAY_mvt"].fill_null("?").to_numpy()
    f = {}
    Bs, Ts = np.sort(B), T                                       # T already sorted
    for x in XS:
        t = T - x
        q = np.searchsorted(Bs, t, side="right") - np.searchsorted(Ts, t, side="right")
        own = ok & (B <= t) & (T > t)
        f[f"q_apt_{x}"] = (q - own).astype(float)
    # oracle nbq (airport, +-30 min, observed pushback, true residual), own excluded
    lo, hi = np.searchsorted(T, T - 1800), np.searchsorted(T, T + 1800, side="right")
    cr = np.concatenate([[0], np.cumsum(np.where(obs & ok, r, 0))])
    cn = np.concatenate([[0], np.cumsum(obs & ok)])
    sr = cr[hi] - cr[lo] - np.where(obs & ok, r, 0)
    sn = cn[hi] - cn[lo] - (obs & ok)
    f["nbq_or_n"] = sn.astype(float)
    f["nbq_or"] = np.where(sn > 0, sr / np.maximum(sn, 1), np.nan)
    for k in ("q_rwy", "qa_rwy"):
        for x in XS:
            f[f"{k}_{x}"] = np.full(n, np.nan)
    for k in ("n_rwy15", "nbp15", "prev3_r", "next3_r", "prev3_y", "prev1_r", "next1_r"):
        f[k] = np.full(n, np.nan)
    for rr in np.unique(rw):
        ix = np.flatnonzero(rw == rr)
        Tr, Br, okr, yr = T[ix], B[ix], ok[ix], y[ix]
        Brs = np.sort(Br)
        for x in XS:
            t = Tr - x
            q = np.searchsorted(Brs, t, side="right") - np.searchsorted(Tr, t, side="right")
            own = okr & (Br <= t) & (Tr > t)
            f[f"q_rwy_{x}"][ix] = q - own
            # ahead: taxiing at t and departing before T_i: j with T_j in (t, T_i) and B_j <= t (own excluded: T_j<T_i)
            a, b = np.searchsorted(Tr, t, side="right"), np.searchsorted(Tr, Tr, side="left")
            qa = np.zeros(len(ix))
            for m in np.flatnonzero(b > a):
                qa[m] = np.count_nonzero(Br[a[m]:b[m]] <= t[m])
            f[f"qa_rwy_{x}"][ix] = qa
        lo, hi = np.searchsorted(Tr, Tr - 900), np.searchsorted(Tr, Tr + 900, side="right")
        f["n_rwy15"][ix] = hi - lo - 1
        pn = p_nm[ix]
        pv = np.where(np.isnan(pn), 0, pn)
        cp = np.concatenate([[0], np.cumsum(pv)]); cq = np.concatenate([[0], np.cumsum(~np.isnan(pn))])
        num = cp[hi] - cp[lo] - pv; den = cq[hi] - cq[lo] - (~np.isnan(pn))
        f["nbp15"][ix] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        rr_ = r[ix]
        L = len(ix)
        P = np.full((L, 3), np.nan); N = np.full((L, 3), np.nan); PY = np.full((L, 3), np.nan)
        for k in (1, 2, 3):
            P[k:, k - 1] = rr_[:-k]; N[:-k, k - 1] = rr_[k:]
            PY[k:, k - 1] = np.clip(yr[:-k], 0, 7200)
        with np.errstate(all="ignore"):
            f["prev3_r"][ix] = np.nanmean(P, axis=1); f["next3_r"][ix] = np.nanmean(N, axis=1)
            f["prev3_y"][ix] = np.nanmean(PY, axis=1)
        f["prev1_r"][ix] = P[:, 0]; f["next1_r"][ix] = N[:, 0]
    return f


def build() -> pl.DataFrame:
    mv = load()
    parts = []
    for (ap,), g in mv.group_by("ADEP_mvt", maintain_order=True):
        if ap == "LIRF":
            continue
        f = group_feats(g)
        parts.append(g.with_columns(**{k: pl.Series(v) for k, v in f.items()}))
    df = pl.concat(parts).filter(pl.col("adsb_day"))
    # own-label leak check: shifting the own label must not change own features
    chk = mv.filter((pl.col("ADEP_mvt") == "EDDF")).head(60000)
    f0 = group_feats(chk)
    chk2 = chk.with_columns(y=pl.when(pl.int_range(pl.len()) == 30000).then(pl.col("y") + 2400).otherwise("y"))
    f1 = group_feats(chk2)
    own = {k: (np.nan_to_num(f0[k][30000], nan=-9) == np.nan_to_num(f1[k][30000], nan=-9)) for k in f0}
    assert all(own.values()), [k for k, v in own.items() if not v]
    print(f"own-label invariance OK on {len(own)} features; eval rows {df.height:,}", flush=True)
    return df


C_FEATS = ["s", "apt", "hour", "off", "a3s", "adsbs", "n_rwy15", "nbp15", "nbq_or", "nbq_or_n"]
O1_FEATS = [f"q_apt_{x}" for x in XS] + [f"q_rwy_{x}" for x in XS] + [f"qa_rwy_{x}" for x in XS]
O2_FEATS = ["prev3_r", "next3_r", "prev3_y", "prev1_r", "next1_r"]
PARAMS = dict(objective="huber", alpha=300.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1,
              num_threads=14, seed=42, deterministic=True, force_row_wise=True)
ROUNDS = 600


def main():
    df = build()
    apts = sorted(df["ADEP_mvt"].unique().to_list())
    df = df.with_columns(apt=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(apts)}, return_dtype=pl.Int32),
                         a3s=pl.col("a3") - pl.col("s"),
                         adsbs=pl.when(pl.col("adsb_tier").is_in(OBS)).then(pl.col("T") - pl.col("adsb_pushback_ts") - pl.col("s")))
    y, s = df["y"].to_numpy(), df["s"].to_numpy()
    ym = df["ym"].to_numpy()
    arms = {"C": C_FEATS, "O1": C_FEATS + O1_FEATS, "O2": C_FEATS + O1_FEATS + O2_FEATS}
    pred = {a: np.full(df.height, np.nan) for a in arms}
    train_ok = ~np.isnan(y) & (y <= TRIM)
    for m in MONTHS:
        te = ym == m
        tr = train_ok & ~te
        for a, cols in arms.items():
            X = df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
            ds = lgb.Dataset(X[tr], (y - s)[tr], categorical_feature=[cols.index("apt")], free_raw_data=True)
            mdl = lgb.train(PARAMS, ds, ROUNDS)
            pred[a][te] = s[te] + mdl.predict(X[te])
        print(f"fold {m} done", flush=True)
    nmu, apt = df["nmu"].to_numpy(), df["ADEP_mvt"].to_numpy()
    obs = df["adsb_tier"].is_in(OBS).fill_null(False).to_numpy()
    ok = ~np.isnan(y)
    body, trim = ok & ~nmu & (y <= 3600), ok & (y <= TRIM)
    R = lambda p, k: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))  # noqa: E731
    print("\nRMSE on held-out months (non-LIRF, 115 OOF ADS-B days)")
    print(f"{'slice':28s} {'s':>9s} {'C':>9s} {'O1':>9s} {'O2':>9s} {'O1-C':>7s} {'O2-C':>7s}")
    for nme, k in [("full", ok), ("trimmed y<=5h", trim), ("body", body), ("body, no own pushback", body & ~obs),
                   ("body, own pushback", body & obs), ("y 30-60 min NM-matched", ok & ~nmu & (y > 1800) & (y <= 3600))]:
        v = [R(s, k)] + [R(pred[a], k) for a in arms]
        print(f"{nme:28s} {v[0]:9.2f} {v[1]:9.2f} {v[2]:9.2f} {v[3]:9.2f} {v[2] - v[1]:+7.2f} {v[3] - v[1]:+7.2f}")
    print("\nbody by month (O1-C, O2-C):")
    for m in MONTHS:
        k = body & (ym == m)
        print(f"  {m}: C {R(pred['C'], k):7.2f}  O1-C {R(pred['O1'], k) - R(pred['C'], k):+6.2f}  O2-C {R(pred['O2'], k) - R(pred['C'], k):+6.2f}")
    print("\nbody by airport (C, O1-C, O2-C):")
    for a in apts:
        k = body & (apt == a)
        print(f"  {a}: {R(pred['C'], k):7.2f}  {R(pred['O1'], k) - R(pred['C'], k):+6.2f}  {R(pred['O2'], k) - R(pred['C'], k):+6.2f}")
    pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], **{f"p_{a}": pred[a] for a in arms}}).write_parquet(
        ROOT / "cache" / "analysis" / "adsb_queue_oracle.parquet")


if __name__ == "__main__":
    main()
