"""Fixed lines for NM-unmatched departures with a long delay (reports/nm_lines_preregistration.md).

Two rules applied to a finished submission, both least-squares lines in d = MVT - SCHED:
  - lines: per airport, y = a + b*d fitted on every NM-unmatched departure (global line for
    airports with < 50 rows); applied where the flight is NM-unmatched and d > 6 h;
  - Rome: LIRF, NM-unmatched, d in (15 h, 30 h]: one line fitted on those rows only (they
    are a mix of SCHED copies and ~87,000 s "+24 h" labels); overrides the airport line.
Idea from team outgoing-boat's public repository (README sections 1.1 and "Rome"); the
lines here are fitted on our own data. Holdout check: fitted on the 10 training months,
scored on Jan+Jul 2025 (cache/eval/v31_chain_holdout.parquet, + LOBT window = v32's method).
Ranking: fitted on all 12 months.

Run:  .venv/Scripts/python.exe src/post/nm_lines.py IN.parquet OUT.parquet
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from post.lobt_window import HALF  # noqa: E402

RAW = ROOT / "data" / "raw"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
HOLDOUT = ROOT / "cache" / "eval" / "v31_chain_holdout.parquet"
LINE_MIN_D = 6 * 3600
ROME_LO, ROME_HI = 15 * 3600, 30 * 3600
MIN_ROWS = 50


def frame(src) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt",
                    nm=pl.col("FLIGHT_ID_mvt").is_null(),
                    d=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                    lt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64),
                    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                    m=pl.col("MVT_TIME_UTC_mvt").dt.month()).collect())


def _line(d: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(d, y, 1)
    return float(a), float(b)


def fit(train: pl.DataFrame) -> dict:
    t = train.filter(pl.col("nm") & pl.col("d").is_not_null() & pl.col("y").is_not_null())
    lines = {a: _line(g["d"].to_numpy(), g["y"].to_numpy())
             for (a,), g in t.group_by("ADEP_mvt") if g.height >= MIN_ROWS}
    rome = t.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("d") > ROME_LO) & (pl.col("d") <= ROME_HI))
    return {"lines": lines, "global": _line(t["d"].to_numpy(), t["y"].to_numpy()),
            "rome": _line(rome["d"].to_numpy(), rome["y"].to_numpy()), "rome_n": rome.height}


def apply(df: pl.DataFrame, p: np.ndarray, prm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    apt, d, nm = df["ADEP_mvt"].to_numpy(), df["d"].to_numpy(), df["nm"].to_numpy()
    a = np.array([prm["lines"].get(x, prm["global"])[0] for x in apt])
    b = np.array([prm["lines"].get(x, prm["global"])[1] for x in apt])
    dd = np.nan_to_num(d, nan=-np.inf)
    use = nm & (dd > LINE_MIN_D)
    rome = (apt == "LIRF") & nm & (dd > ROME_LO) & (dd <= ROME_HI)
    out = np.asarray(p, float).copy()
    out[use] = np.clip(a + b * d, 0, None)[use]
    out[rome] = np.clip(prm["rome"][0] + prm["rome"][1] * d, 0, None)[rome]
    return out, use, rome


def holdout_check(raw: pl.DataFrame) -> None:
    prm = fit(raw.filter(~pl.col("m").is_in([1, 7])))
    h = pl.read_parquet(HOLDOUT).select("MVT_ID_mvt", "taxi", "pred_v31").join(raw, on="MVT_ID_mvt")
    lt, p = h["lt"].to_numpy(), h["pred_v31"].to_numpy()
    p = np.where(np.isnan(lt), p, np.clip(p, np.clip(lt - HALF, 0, None), lt + HALF))   # v32's method
    q, use, rome = apply(h, p, prm)
    y, m = h["taxi"].to_numpy(), h["m"].to_numpy()
    r = lambda z, k: float(np.sqrt(np.mean((y - z)[k] ** 2)))
    allr, tr = np.ones(len(y), bool), y <= 18000
    print(f"holdout (10-month fit; Rome a={prm['rome'][0]:.0f} b={prm['rome'][1]:.3f} on {prm['rome_n']} rows): "
          f"rows lines {use.sum()}, Rome {rome.sum()}")
    for nm_, k in (("full", allr), ("trimmed", tr), ("Jan", m == 1), ("Jul", m == 7), ("body y<=1h", y <= 3600)):
        print(f"  {nm_:10s} {r(p, k):8.2f} -> {r(q, k):8.2f}")


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    raw = frame(str(RAW / "training_*.parquet"))
    holdout_check(raw)
    prm = fit(raw)
    print("12-month fit:", {k: (round(v[0]), round(v[1], 3)) for k, v in sorted(prm["lines"].items())},
          f"| Rome a={prm['rome'][0]:.0f} b={prm['rome'][1]:.3f} ({prm['rome_n']} rows)")
    sub = pl.read_parquet(src)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    rk = sub.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id")).join(
        frame(RANKING).rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
    assert rk["ADEP_mvt"].null_count() == 0
    p = rk["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    q, use, rome = apply(rk, p, prm)
    print(f"ranking: line rows {use.sum()}, Rome rows {rome.sum()}; mean shift on changed rows "
          f"{np.mean((q - p)[use | rome]):+.0f} s; RMS change over all rows {np.sqrt(np.mean((q - p) ** 2)):.1f} s")
    out = rk.select("MVT_ID_mvt").with_columns(TAXITIME_SEC_mvt=pl.Series(np.round(q)).cast(dtype))
    tpl = pl.read_parquet(TEMPLATE)
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
