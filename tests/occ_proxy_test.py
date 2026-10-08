"""Label-free occupancy proxies vs the O1 oracle (reports/occ_proxy_scoping.md).

Same rows (115 OOF ADS-B days, non-LIRF), folds (leave-one-month-out) and residual model as
tests/adsb_queue_oracle.py. Occupancy blocks are rebuilt from estimated off-blocks of OTHER flights:
  Pa3   : AOBT_3 where present, else T - s;   Ppred : T - s (out-of-sample stack);   Pboth : both.
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

XS = Q.XS


def occ_block(T: np.ndarray, B: np.ndarray, rw: np.ndarray, pre: str) -> dict[str, np.ndarray]:
    """Occupancy at T - x (airport, own runway), queue-ahead on the own runway, and traffic-normalised copies.
    B = estimated off-block of every flight (np.inf = unknown); each flight's own entry is excluded."""
    n = len(T)
    f = {}
    Bs = np.sort(B)
    for x in XS:
        t = T - x
        q = np.searchsorted(Bs, t, side="right") - np.searchsorted(T, t, side="right") - ((B <= t) & (T > t))
        thr = np.searchsorted(T, t + 1800, side="right") - np.searchsorted(T, t, side="right")
        f[f"{pre}_apt_{x}"] = q.astype(float)
        f[f"{pre}n_apt_{x}"] = q / np.maximum(thr, 1)
    for k in ("rwy", "rwya", "nrwy", "nrwya"):
        for x in XS:
            f[f"{pre}_{k}_{x}"] = np.full(n, np.nan)
    for rr in np.unique(rw):
        ix = np.flatnonzero(rw == rr)
        Tr, Br = T[ix], B[ix]
        Brs = np.sort(Br)
        for x in XS:
            t = Tr - x
            q = np.searchsorted(Brs, t, side="right") - np.searchsorted(Tr, t, side="right") - ((Br <= t) & (Tr > t))
            thr = np.searchsorted(Tr, t + 1800, side="right") - np.searchsorted(Tr, t, side="right")
            a, b = np.searchsorted(Tr, t, side="right"), np.searchsorted(Tr, Tr, side="left")
            qa = np.zeros(len(ix))
            for m in np.flatnonzero(b > a):
                qa[m] = np.count_nonzero(Br[a[m]:b[m]] <= t[m])
            f[f"{pre}_rwy_{x}"][ix] = q
            f[f"{pre}_rwya_{x}"][ix] = qa
            f[f"{pre}_nrwy_{x}"][ix] = q / np.maximum(thr, 1)
            f[f"{pre}_nrwya_{x}"][ix] = qa / np.maximum(thr, 1)
    return f


def names(pre: str) -> list[str]:
    return ([f"{pre}_apt_{x}" for x in XS] + [f"{pre}n_apt_{x}" for x in XS]
            + [f"{pre}_{k}_{x}" for k in ("rwy", "rwya", "nrwy", "nrwya") for x in XS])


def build() -> pl.DataFrame:
    mv = Q.load()
    parts = []
    for (ap,), g in mv.group_by("ADEP_mvt", maintain_order=True):
        if ap == "LIRF":
            continue
        T, s = g["T"].to_numpy(), g["s"].to_numpy()
        y = g["y"].to_numpy()
        rw = g["RUNWAY_mvt"].fill_null("?").to_numpy()
        a3 = g["a3"].to_numpy()
        Bt = np.where(~np.isnan(y), T - np.nan_to_num(y), np.inf)
        Bp = T - np.clip(s, 0, 7200)
        Ba = np.where(~np.isnan(a3), T - a3, Bp)
        f = Q.group_feats(g)                       # control + oracle features
        f.update(occ_block(T, Ba, rw, "pa"))
        f.update(occ_block(T, Bp, rw, "pp"))
        assert np.allclose(np.nan_to_num(occ_block(T, Bt, rw, "o")["o_rwya_600"]), np.nan_to_num(f["qa_rwy_600"]))
        parts.append(g.with_columns(**{k: pl.Series(v) for k, v in f.items()}))
    return pl.concat(parts).filter(pl.col("adsb_day"))


def main():
    df = build()
    apts = sorted(df["ADEP_mvt"].unique().to_list())
    df = df.with_columns(apt=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(apts)}, return_dtype=pl.Int32),
                         a3s=pl.col("a3") - pl.col("s"),
                         adsbs=pl.when(pl.col("adsb_tier").is_in(Q.OBS)).then(pl.col("T") - pl.col("adsb_pushback_ts") - pl.col("s")))
    print(f"eval rows {df.height:,}", flush=True)
    C = Q.C_FEATS
    arms = {"C": C, "O1": C + Q.O1_FEATS, "Pa3": C + names("pa"), "Ppred": C + names("pp"),
            "Pboth": C + names("pa") + names("pp")}
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
    ok = ~np.isnan(y)
    body = ok & ~nmu & (y <= 3600)
    R = lambda p, k: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))  # noqa: E731
    Rc = R(pred["C"], body)
    contrib = lambda p, k: float((((y - p) ** 2)[k].sum() - ((y - pred["C"]) ** 2)[k].sum()) / body.sum() / (2 * Rc))  # noqa: E731
    P = ["O1", "Pa3", "Ppred", "Pboth"]
    print("\nRMSE deltas vs C on held-out months (non-LIRF, 115 OOF ADS-B days)")
    for nme, k in [("full", ok), ("trimmed", ok & (y <= Q.TRIM)), ("body", body),
                   ("NM-matched 30-60 min", ok & ~nmu & (y > 1800) & (y <= 3600))]:
        c = R(pred["C"], k)
        print(f"  {nme:22s} C {c:8.2f} " + " ".join(f"{a} {R(pred[a], k) - c:+7.2f}" for a in P))
    print("\nbody by month:")
    for mo in Q.MONTHS:
        k = body & (ym == mo); c = R(pred["C"], k)
        print(f"  {mo}: C {c:7.2f} " + " ".join(f"{a} {R(pred[a], k) - c:+6.2f}" for a in P))
    print("\nbody by airport (own-row delta | contribution to pooled body delta | fraction of O1 recovered):")
    for a in apts:
        k = body & (apt == a); c = R(pred["C"], k); o = R(pred["O1"], k) - c
        print(f"  {a}: C {c:7.2f} O1 {o:+6.2f} | " + " ".join(
            f"{p} {R(pred[p], k) - c:+6.2f} ({contrib(pred[p], k):+.2f}, {(R(pred[p], k) - c) / o if o < 0 else float('nan'):.2f})"
            for p in P[1:]))
    k2 = body & np.isin(apt, ["LFPG", "LTFM"])
    print("\nLFPG+LTFM contribution to pooled body delta: " + " ".join(f"{p} {contrib(pred[p], k2):+.3f}" for p in P))
    print("\nLFPG / LTFM body by month (O1, Pa3, Ppred, Pboth vs C):")
    for a in ("LFPG", "LTFM"):
        for mo in Q.MONTHS:
            k = body & (apt == a) & (ym == mo); c = R(pred["C"], k)
            print(f"  {a} {mo}: C {c:7.2f} " + " ".join(f"{p} {R(pred[p], k) - c:+6.2f}" for p in P))
    pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], **{f"p_{a}": pred[a] for a in arms}}).write_parquet(
        Q.ROOT / "cache" / "analysis" / "occ_proxy.parquet")


if __name__ == "__main__":
    main()
