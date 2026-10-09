"""LIRF NM-unmatched hedge, round 2 (reports/lirf_hedge2_preregistration.md).

On v45's scope minus arrival-twin rows: q2 = logistic on v45's design + the airline's NM-matched rate among its
LIRF departures >= 1 h late (history only), and the non-echo branch bounded by the late-branch limit off - 3,606:
  P' = q2 offc + (1 - q2) min(P_ne, max(off - 3606, 0)),  P_ne recovered from the input with v45's q.

Run:  .venv/Scripts/python.exe -u src/post/lirf_hedge2.py IN.parquet OUT.parquet   (files in data/submissions)
PRC_SCOPE=final4m: final_ranking rows, files in data/submissions/final4m (src/scope.py, PROGRESS.md §126).
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
from post.lirf_twin_bounds import twin_frame  # noqa: E402
import scope  # noqa: E402

SUB = scope.out_dir(ROOT / "data" / "submissions")
WIN = 3606
MIN_HIST = 5
RATE_FEW = 0.75


def airline_late(src) -> pl.DataFrame:
    """LIRF departures >= 1 h late: MVT_ID, airline prefix, NM-matched flag, month."""
    return (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
            .select(L.I64, al=pl.col("FLIGHT_mvt").str.slice(0, 3), nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
                    off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                    mt=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"))
            .collect())


def rate_table(hist: pl.DataFrame) -> pl.DataFrame:
    h = hist.filter(pl.col("off") >= 3600)
    return h.group_by("al").agg(N=pl.len(), NM=pl.col("nm").sum())


def add_rate(z: pl.DataFrame, tab: pl.DataFrame) -> pl.DataFrame:
    z = z.drop([c for c in ("N", "NM", "rate", "few", "lN") if c in z.columns])
    return (z.join(tab, on="al", how="left").with_columns(pl.col("N", "NM").fill_null(0))
            .with_columns(rate=pl.when(pl.col("N") >= MIN_HIST).then(pl.col("NM") / pl.col("N")).otherwise(RATE_FEW),
                          few=(pl.col("N") < MIN_HIST).cast(pl.Float64), lN=pl.col("N").cast(pl.Float64).log1p()))


def design2(z: pl.DataFrame) -> np.ndarray:
    return np.column_stack([L.design(z)] + [z[c].to_numpy().astype(float) for c in ("rate", "few", "lN")])


def frame25() -> pl.DataFrame:
    src = str(L.RAW / "training_*.parquet")
    d = L.frame(src).join(L.p_2025(), on="MVT_ID_mvt", how="left")
    return d.join(airline_late(src).select("MVT_ID_mvt", "al"), on="MVT_ID_mvt", how="left")


def with_loo_rate(d: pl.DataFrame, hist: pl.DataFrame) -> pl.DataFrame:
    """2025 rows: rate from the other 11 months of history."""
    parts = []
    for m in sorted(d["mt"].unique().to_list()):
        parts.append(add_rate(d.filter(pl.col("mt") == m), rate_table(hist.filter(pl.col("mt") != m))))
    return pl.concat(parts, how="vertical_relaxed")


def fit2(d: pl.DataFrame) -> LogisticRegression:
    t = L.train_rows(d)
    return LogisticRegression(C=1.0, max_iter=3000).fit(design2(t), t["echo"].to_numpy().astype(int))


def treat(z: pl.DataFrame, P_in: np.ndarray, clf45, clf2, skip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P' for rows of z. P_in is the input prediction (already carrying v45's treatment on scope)."""
    k = L.in_scope(z) & ~skip
    p = z["p"].fill_null(1.0).to_numpy()
    off = z["off"].to_numpy()
    offc = np.clip(off, 0, L.WIDE)
    q45, q2 = p.copy(), p.copy()
    if k.any():
        zk = z.filter(pl.Series(k))
        q45[k] = clf45.predict_proba(L.design(zk))[:, 1]
        q2[k] = clf2.predict_proba(design2(zk))[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pne = (P_in - q45 * offc) / (1 - q45)
    ub = np.maximum(off - WIN, 0.0)
    out = np.where(k, np.maximum(q2 * offc + (1 - q2) * np.minimum(pne, ub), 0.0), P_in)
    return out, k


def main() -> None:
    src, dst = SUB / sys.argv[1], SUB / sys.argv[2]
    hist = airline_late(str(L.RAW / "training_*.parquet"))
    d25 = with_loo_rate(frame25(), hist)
    clf45, clf2 = L.fit(d25), fit2(d25)
    sub = pl.read_parquet(src)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    rk = str(L.RANKING)
    d26 = (L.frame(rk).join(L.p_2026(), on="MVT_ID_mvt", how="left").join(airline_late(rk).select("MVT_ID_mvt", "al"), on="MVT_ID_mvt", how="left")
           .join(sub.select(L.I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt"))
    d26 = add_rate(d26, rate_table(hist))
    twins = set(twin_frame(rk)["MVT_ID_mvt"].to_list())
    skip = np.array([i in twins for i in d26["MVT_ID_mvt"].to_list()])
    P = d26["P"].to_numpy()
    Q, k = treat(d26, P, clf45, clf2, skip)
    print(f"2026 scope rows (non-twin) {k.sum()}; moved {(np.abs(Q - P) > 0.5).sum()}; mean change {np.mean((Q - P)[k]):+.0f} s; "
          f"RMS on scope {np.sqrt(np.mean((Q - P)[k] ** 2)):.0f} s; sum (Q-P)^2 {np.sum((Q - P) ** 2):.3g}")
    # label-free drift: airline matched rate 2025 vs 2026
    t25 = rate_table(hist).filter(pl.col("N") >= 30)
    t26 = rate_table(airline_late(rk)).rename({"N": "N26", "NM": "NM26"})
    dr = t25.join(t26, on="al").filter(pl.col("N26") >= 10).with_columns(r25=pl.col("NM") / pl.col("N"), r26=pl.col("NM26") / pl.col("N26"))
    print(f"drift (airlines with >= 30 late 2025 / >= 10 late 2026 departures, n {dr.height}): corr(r25, r26) "
          f"{dr.select(pl.corr('r25', 'r26')).item():.3f}; mean r25 {dr['r25'].mean():.3f} vs r26 {dr['r26'].mean():.3f}")
    upd = d26.select("MVT_ID_mvt", new=pl.Series(Q), k=pl.Series(k)).filter(pl.col("k"))
    out = (sub.with_columns(_id=L.I64).join(upd.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
           .with_columns(TAXITIME_SEC_mvt=pl.when(pl.col("k")).then(pl.col("new").round()).otherwise(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).cast(dtype))
           .select(sub.columns))
    tpl = pl.read_parquet(L.TEMPLATE)
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list()) and float(out["TAXITIME_SEC_mvt"].min()) >= 0
    n_diff = int((out["TAXITIME_SEC_mvt"].cast(pl.Float64) != sub["TAXITIME_SEC_mvt"].cast(pl.Float64)).sum())
    print(f"rows differing from {src.name}: {n_diff}")
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
