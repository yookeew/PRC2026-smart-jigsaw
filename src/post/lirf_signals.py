"""LIRF NM-unmatched echo hedge with label-free movement signals (reports/lirf_signal_hedge_preregistration.md).

On LIRF NM-unmatched departures 1-6 h late, the stack echo probability p is replaced by q, a logistic on
[logit p, signals]. Signals count neighbours in the same MVT_TIME calendar month only, which is identical
in 2025 and the Jan/Jul-only 2026 ranking:
  nF / nF7  other LIRF departures with the same FLIGHT_mvt, |SCHED' - SCHED| < 36 h / 7 days
  nD        other LIRF departures from the same stand with MVT_TIME' in (SCHED + 10 min, T - 30 min)
  nA        LIRF arrivals at the same stand with in-block in that window
  fnull     FLIGHT_mvt is null
Only the echo weight moves: P' = P + (q - p) (offc - P_ne), P_ne = (P - p offc) / (1 - p), so every chain
correction inside the non-echo branch is kept. Rows with off >= 6 h (nm_lines) or p >= 0.99 are untouched.

Run:  .venv/Scripts/python.exe -u src/post/lirf_signals.py IN.parquet OUT.parquet
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
RAW = ROOT / "data" / "raw"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
EVAL, OOF = ROOT / "cache" / "eval", ROOT / "cache" / "oof"
W_LGB, W_CAT = 0.577, 0.428          # §101/§103 stack convention
WIDE = 140000.0
LO, HI = 3600, 6 * 3600              # scope: 1 h <= off < 6 h
ECHO_TOL = 30
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)


def frame(src) -> pl.DataFrame:
    """LIRF departures with off >= 1 h: ids, off, echo label (null without a block), month, signals."""
    m = (pl.scan_parquet(src)
         .filter(((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
                 | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")))
         .select(I64, "PHASE_mvt", "FLIGHT_mvt", "STAND_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(),
                 T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), S=pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s"),
                 B=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s"), mt=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
                 y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
         .collect())
    dep, arr = m.filter(pl.col("PHASE_mvt") == "DEP"), m.filter(pl.col("PHASE_mvt") == "ARR")
    t = dep.filter((pl.col("T") - pl.col("S")) >= LO).select(
        "MVT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "nmu", "T", "S", "mt", "y",
        off=(pl.col("T") - pl.col("S")).cast(pl.Float64), fnull=pl.col("FLIGHT_mvt").is_null(),
        echo=pl.when(pl.col("B").is_null()).then(None).otherwise((pl.col("B") - pl.col("S")).abs() < ECHO_TOL))
    f = t.select("MVT_ID_mvt", "FLIGHT_mvt", "mt", "S").join(
        dep.select(pl.col("MVT_ID_mvt").alias("i2"), "FLIGHT_mvt", "mt", pl.col("S").alias("S2")), on=["FLIGHT_mvt", "mt"]
    ).filter(pl.col("i2") != pl.col("MVT_ID_mvt")).with_columns(ds=(pl.col("S2") - pl.col("S")).abs())
    nF = f.group_by("MVT_ID_mvt").agg(nF=(pl.col("ds") < 36 * 3600).sum(), nF7=(pl.col("ds") < 7 * 86400).sum())
    win = lambda c: (pl.col(c) > pl.col("S") + 600) & (pl.col(c) < pl.col("T") - 1800)  # noqa: E731
    nD = (t.select("MVT_ID_mvt", "STAND_mvt", "mt", "S", "T")
          .join(dep.select(pl.col("MVT_ID_mvt").alias("i2"), "STAND_mvt", "mt", pl.col("T").alias("T2")), on=["STAND_mvt", "mt"])
          .filter((pl.col("i2") != pl.col("MVT_ID_mvt")) & win("T2")).group_by("MVT_ID_mvt").agg(nD=pl.len()))
    nA = (t.select("MVT_ID_mvt", "STAND_mvt", "mt", "S", "T")
          .join(arr.select("STAND_mvt", "mt", pl.col("B").alias("A2")), on=["STAND_mvt", "mt"])
          .filter(win("A2")).group_by("MVT_ID_mvt").agg(nA=pl.len()))
    for g in (nF, nD, nA):
        t = t.join(g, on="MVT_ID_mvt", how="left")
    return t.with_columns(pl.col("nF", "nF7", "nD", "nA").fill_null(0))


def p_frame(lgb: list[Path], cat: list[Path]) -> pl.DataFrame:
    """Stack echo probability p and stack base prediction Pb per MVT_ID."""
    rd = lambda ps, s: pl.concat([pl.read_parquet(p).select(I64, pl.col("echo_prob").alias("p" + s), pl.col("pred").alias("P" + s))  # noqa: E731
                                  for p in ps])
    return rd(lgb, "l").join(rd(cat, "c"), on="MVT_ID_mvt").select(
        "MVT_ID_mvt", p=W_LGB * pl.col("pl") + W_CAT * pl.col("pc"), Pb=W_LGB * pl.col("Pl") + W_CAT * pl.col("Pc"))


def p_2025() -> pl.DataFrame:
    return p_frame(sorted((OOF / "lgb_mixed_nb").glob("*.parquet")) + [EVAL / "lgb_mixed_nb_holdout_ev.parquet"],
                   sorted((OOF / "cat_mixed").glob("*.parquet")) + [EVAL / "cat_mixed_holdout_ev.parquet"])


def p_2026() -> pl.DataFrame:
    return p_frame([EVAL / "lgb_direct_nolirf_nb_rank.parquet"], [EVAL / "cat_mixed_rank.parquet"])


def design(d: pl.DataFrame) -> np.ndarray:
    p = np.clip(d["p"].to_numpy(), 1e-4, 1 - 1e-4)
    F = lambda c: d[c].to_numpy().astype(float)  # noqa: E731
    return np.column_stack([np.log(p / (1 - p)), np.log1p(F("nF")), F("nF") == 0, np.log1p(F("nF7")),
                            np.log1p(F("nD")), F("nD") > 0, np.log1p(F("nA")), F("fnull")])


def train_rows(d: pl.DataFrame) -> pl.DataFrame:
    return d.filter(pl.col("nmu") & pl.col("echo").is_not_null() & pl.col("p").is_not_null())


def fit(d: pl.DataFrame) -> LogisticRegression:
    t = train_rows(d)
    return LogisticRegression(C=1.0, max_iter=2000).fit(design(t), t["echo"].to_numpy().astype(int))


def in_scope(d: pl.DataFrame) -> np.ndarray:
    off, p = d["off"].to_numpy(), d["p"].fill_null(1.0).to_numpy()
    return d["nmu"].to_numpy() & (off >= LO) & (off < HI) & (p < 0.99)


def apply(d: pl.DataFrame, P: np.ndarray, clf: LogisticRegression) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P' for the rows of d (P aligned to d). Returns (P', scope mask, q)."""
    k = in_scope(d)
    p, offc = d["p"].fill_null(1.0).to_numpy(), np.clip(d["off"].to_numpy(), 0, WIDE)
    q = p.copy()
    if k.any():
        q[k] = clf.predict_proba(design(d.filter(pl.Series(k))))[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pne = (P - p * offc) / (1 - p)
    out = np.where(k, np.maximum(P + (q - p) * (offc - pne), 0.0), P)
    return out, k, q


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    d25 = frame(str(RAW / "training_*.parquet")).join(p_2025(), on="MVT_ID_mvt", how="left")
    clf = fit(d25)
    t = train_rows(d25)
    print(f"12-month fit on {t.height:,} rows (echo {t['echo'].mean():.3f}); coef {np.round(clf.coef_[0], 3)}, "
          f"intercept {clf.intercept_[0]:.3f}")
    sub = pl.read_parquet(src)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    d26 = (frame(str(RANKING)).join(p_2026(), on="MVT_ID_mvt", how="left")
           .join(sub.select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt", how="left"))
    assert d26["P"].null_count() == 0 and d26["p"].null_count() == 0
    P = d26["P"].to_numpy()
    new, k, q = apply(d26, P, clf)
    p, off = d26["p"].to_numpy(), d26["off"].to_numpy()
    print(f"2026 LIRF NM-unmatched >= 1 h: {int(d26['nmu'].sum())}; in scope {k.sum()} "
          f"(1-3 h {(k & (off < 3 * 3600)).sum()}, 3-6 h {(k & (off >= 3 * 3600)).sum()})")
    print(f"  q - p on scope: mean {np.mean((q - p)[k]):+.3f}, p10 {np.quantile((q - p)[k], .1):+.3f}, "
          f"p90 {np.quantile((q - p)[k], .9):+.3f}; mean p {p[k].mean():.3f} -> mean q {q[k].mean():.3f}")
    dP = (new - P)[k]
    print(f"  change on scope: mean {dP.mean():+.0f} s, RMS {np.sqrt(np.mean(dP ** 2)):.0f} s, max |{np.abs(dP).max():.0f}| s; "
          f"RMS over all 344,841 rows {np.sqrt(np.sum(dP ** 2) / sub.height):.1f} s")
    upd = d26.select("MVT_ID_mvt").with_columns(new=pl.Series(new), k=pl.Series(k)).filter(pl.col("k"))
    out = (sub.with_columns(_id=I64).join(upd.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
           .with_columns(TAXITIME_SEC_mvt=pl.when(pl.col("k")).then(pl.col("new").round()).otherwise(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).cast(dtype))
           .select(sub.columns))
    tpl = pl.read_parquet(TEMPLATE)
    assert out.height == tpl.height == 344_841 and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    assert float(out["TAXITIME_SEC_mvt"].min()) >= 0
    diff = out.with_columns(_id=I64).join(sub.select(_id=I64, old="TAXITIME_SEC_mvt"), on="_id").filter(
        pl.col("TAXITIME_SEC_mvt") != pl.col("old"))
    assert set(diff["_id"].to_list()) <= set(upd["MVT_ID_mvt"].to_list())
    print(f"  rows differing from {src.name}: {diff.height} (all in scope)")
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
