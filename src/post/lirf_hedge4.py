"""LIRF NM-unmatched hedge, round 4: flight-number echo history (reports/lirf_hedge4_preregistration.md).

On v48's scope (v45's scope minus arrival twins): q4 = logistic on v48's design + the flight number's history among
late (>= 1 h) LIRF departures (NM-unmatched / NM-matched counts and smoothed echo rates), past months only; the
non-echo branch and its late-branch bound exactly as v48:
  P' = q4 offc + (1 - q4) min(P_ne, max(off - 3606, 0)),  P_ne recovered from v47's value with v45's q.
v48 is recomputed from v47 with q2 as a check (must equal the v48 file on every treated row). The new values replace
the base file's rows only where the base still carries v48's value.

Run:  .venv/Scripts/python.exe -u src/post/lirf_hedge4.py V47.parquet V48.parquet BASE.parquet OUT.parquet  (v60: ... smart-jigsaw_v64 smart-jigsaw_v60; rejected, PROGRESS §141)  (data/submissions)
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import post.lirf_signals as L  # noqa: E402
import post.lirf_hedge2 as H  # noqa: E402
from post.lirf_twin_bounds import twin_frame  # noqa: E402
import scope  # noqa: E402

SUB = scope.out_dir(ROOT / "data" / "submissions")
PRIOR, W, GAP = 0.49, 2.0, 1          # §111 smoothing; training history from months <= own month - GAP


def late_deps(src) -> pl.DataFrame:
    """Late (>= 1 h) LIRF departures: MVT_ID, flight number, NM-unmatched, month, has block, echo."""
    return (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")).select(
        L.I64, fn="FLIGHT_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(), mi=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64),
        off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        has_blk=pl.col("BLOCK_TIME_UTC_mvt").is_not_null(),
        echo=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() < 30)
        .filter(pl.col("off") >= 3600).collect())


def hist_feats(z: pl.DataFrame, hist: pl.DataFrame, past_only: bool) -> np.ndarray:
    """z: fn, mi. past_only: history months <= mi - GAP (2025 training rows); else every hist row (2026 rows)."""
    h = hist.filter(pl.col("fn").is_not_null() & pl.col("has_blk")).select("fn", "nmu", hm="mi", he=pl.col("echo").cast(pl.Float64))
    t = z.select("fn", pl.col("mi").cast(pl.Int64)).with_row_index("_r")
    j = t.join(h, on="fn", how="inner")
    if past_only:
        j = j.filter(pl.col("hm") <= pl.col("mi") - GAP)
    g = j.group_by("_r", "nmu").agg(n=pl.len(), e=pl.col("he").sum())
    out = np.zeros((t.height, 4)); out[:, 1] = out[:, 3] = PRIOR
    for nmu, c in ((True, 0), (False, 2)):
        s = g.filter(pl.col("nmu") == nmu); r = s["_r"].to_numpy()
        out[r, c] = s["n"].to_numpy(); out[r, c + 1] = (s["e"].to_numpy() + W * PRIOR) / (s["n"].to_numpy() + W)
    lo = lambda q: np.log(np.clip(q, .02, .98) / (1 - np.clip(q, .02, .98)))  # noqa: E731
    return np.column_stack([lo(out[:, 1]) * (out[:, 0] > 0), np.log1p(out[:, 0]), lo(out[:, 3]) * (out[:, 2] > 0), np.log1p(out[:, 2])])


def main() -> None:
    v47, v48, base, dst = (SUB / a for a in sys.argv[1:5])
    src25 = str(L.RAW / "training_*.parquet")
    hist = H.airline_late(src25)
    late = late_deps(src25)
    d25 = H.with_loo_rate(H.frame25(), hist).join(late.select("MVT_ID_mvt", "fn"), on="MVT_ID_mvt", how="left") \
        .with_columns(mi=pl.col("mt").str.slice(5, 2).cast(pl.Int64))
    clf45, clf2 = L.fit(d25), H.fit2(d25)
    t = L.train_rows(d25)
    clf4 = LogisticRegression(C=1.0, max_iter=4000).fit(np.column_stack([H.design2(t), hist_feats(t, late, True)]),
                                                        t["echo"].to_numpy().astype(int))
    rk = str(L.RANKING)
    fn26 = (pl.scan_parquet(rk).filter(pl.col("PHASE_mvt") == "DEP")
            .select(L.I64, fn="FLIGHT_mvt", mi=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64)).collect())
    rd = lambda f, c: pl.read_parquet(f).select(L.I64, pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    d26 = (L.frame(rk).join(L.p_2026(), on="MVT_ID_mvt", how="left").join(H.airline_late(rk).select("MVT_ID_mvt", "al"), on="MVT_ID_mvt", how="left")
           .join(rd(v47, "P"), on="MVT_ID_mvt").join(rd(v48, "P48"), on="MVT_ID_mvt").join(rd(base, "PB"), on="MVT_ID_mvt")
           .join(fn26, on="MVT_ID_mvt", how="left"))
    d26 = H.add_rate(d26, H.rate_table(hist))
    twins = set(twin_frame(rk)["MVT_ID_mvt"].to_list())
    skip = d26["MVT_ID_mvt"].is_in(list(twins)).to_numpy()
    P, P48, PB = d26["P"].to_numpy(), d26["P48"].to_numpy(), d26["PB"].to_numpy()
    Q48, k = H.treat(d26, P, clf45, clf2, skip)
    chk = np.abs(np.round(Q48[k]) - P48[k])
    print(f"v48 recomputed from v47 on {k.sum()} treated rows: max |diff| {chk.max():.1f} s (rows > 1 s: {(chk > 1).sum()})")
    assert (chk > 1).sum() == 0, "v48 not reproduced"
    # round 4
    p = d26["p"].fill_null(1.0).to_numpy(); off = d26["off"].to_numpy(); offc = np.clip(off, 0, L.WIDE)
    zk = d26.filter(pl.Series(k))
    q45, q4 = p.copy(), p.copy()
    q45[k] = clf45.predict_proba(L.design(zk))[:, 1]
    fk = hist_feats(zk, late, False)
    q4[k] = clf4.predict_proba(np.column_stack([H.design2(zk), fk]))[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pne = (P - q45 * offc) / (1 - q45)
    Q4 = np.where(k, np.maximum(q4 * offc + (1 - q4) * np.minimum(pne, np.maximum(off - 3606, 0)), 0), P)
    live = k & (np.abs(PB - P48) <= 1)
    has = (fk[:, 1] + fk[:, 3]) > 0
    mo = d26["mi"].to_numpy()
    print(f"treated rows {k.sum()}; base still at v48's value on {live.sum()} (others kept as base: {(k & ~live).sum()}); "
          f"with a flight-number history {has.sum()} of {k.sum()}")
    for m in np.unique(mo[k]):
        km = live & (mo == m)
        dq = (Q4 - PB)[km]
        print(f"  month {m}: rows {km.sum()}, moved > 1 s {(np.abs(dq) > 1).sum()}, mean change {dq.mean():+.0f} s, RMS {np.sqrt(np.mean(dq ** 2)):.0f} s, "
              f"mean q2 -> q4 {np.mean(H_q2(d26, clf2, k)[km]):.3f} -> {q4[km].mean():.3f}")
    sub = pl.read_parquet(base)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    upd = d26.select("MVT_ID_mvt", new=pl.Series(Q4), k=pl.Series(live)).filter(pl.col("k"))
    out = (sub.with_columns(_id=L.I64).join(upd.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
           .with_columns(TAXITIME_SEC_mvt=pl.when(pl.col("k")).then(pl.col("new").round()).otherwise(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).cast(dtype))
           .select(sub.columns))
    tpl = pl.read_parquet(L.TEMPLATE)
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list()) and float(out["TAXITIME_SEC_mvt"].min()) >= 0
    diff = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
    print(f"rows differing from {base.name}: {int((diff != 0).sum())}; sum of squared change {float((diff ** 2).sum()):.3g}; "
          f"RMS over all rows {float((diff ** 2).mean() ** 0.5):.2f} s")
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


def H_q2(d26, clf2, k):
    q = np.full(d26.height, np.nan)
    q[k] = clf2.predict_proba(H.design2(d26.filter(pl.Series(k))))[:, 1]
    return q


if __name__ == "__main__":
    main()
