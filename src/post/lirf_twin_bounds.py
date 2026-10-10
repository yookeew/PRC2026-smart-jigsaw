"""LIRF NM-unmatched departures: hard bounds from the arrival twin's LOBT (reports/lirf_twin_bounds_preregistration.md).

The movement/NM join is a time window (|BLOCK - LOBT| <= 3,606 s on every matched 2025 departure). A LIRF departure
that failed it, whose flight matched as an arrival at another of the 10 airports, has its LOBT on that arrival row
(the twin). With tl = T - LOBT_twin the block is outside LOBT +- 3,606:
  forced (tl < 3,726)          block before the window: P' = max(P, tl + 3,606)
  not forced, off > tl - 3,606 the echo label would break the bound: P' = clip(P_ne, 0, tl - 3,606)
  not forced otherwise         P' = min(P, tl - 3,606)
P_ne is the non-echo branch implied by P (P = p offc + (1 - p) P_ne).

Run:  .venv/Scripts/python.exe -u src/post/lirf_twin_bounds.py IN.parquet OUT.parquet   (files in data/submissions)
PRC_SCOPE=final4m: final_ranking rows, files in data/submissions/final4m (src/scope.py, PROGRESS.md §132).
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
from post.lirf_signals import I64, WIDE, p_2026  # noqa: E402
import scope  # noqa: E402

RANKING = scope.RANKING
TEMPLATE = scope.TEMPLATE
SUB = scope.out_dir(ROOT / "data" / "submissions")
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
WIN = 3606
FORCED = WIN + 120
MAX_FLIGHT = 8 * 3600


def twin_frame(src) -> pl.DataFrame:
    """LIRF NM-unmatched departures with an NM-matched arrival twin: MVT_ID, off, tl, y (null in 2026)."""
    ts = lambda c: pl.col(c).dt.epoch("s")  # noqa: E731
    m = pl.scan_parquet(src).select(I64, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt",
                                    T=ts("MVT_TIME_UTC_mvt"), S=ts("SCHED_TIME_UTC_mvt"), L=ts("LOBT_flt"),
                                    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).collect()
    dep = m.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()
                   & pl.col("ADES_mvt").is_in(APT) & (pl.col("ADES_mvt") != "LIRF") & pl.col("FLIGHT_mvt").is_not_null())
    arr = m.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("L").is_not_null()).select(
        "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", Ta="T", La="L")
    j = (dep.join(arr, on=["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"])
         .filter((pl.col("Ta") > pl.col("T")) & (pl.col("Ta") < pl.col("T") + MAX_FLIGHT))
         .sort("Ta").group_by("MVT_ID_mvt", maintain_order=True).first())
    return j.select("MVT_ID_mvt", "y", off=(pl.col("T") - pl.col("S")).cast(pl.Float64), tl=(pl.col("T") - pl.col("La")).cast(pl.Float64),
                    m=pl.from_epoch("T").dt.month())


def bounds(P: np.ndarray, p: np.ndarray, off: np.ndarray, tl: np.ndarray) -> np.ndarray:
    offc = np.clip(off, 0, WIDE)
    forced = tl < FORCED
    up, lo = np.maximum(tl - WIN, 0.0), tl + WIN
    with np.errstate(divide="ignore", invalid="ignore"):
        pne = np.where(p < 0.99, (P - p * offc) / (1 - p), P)
    no_echo = ~forced & (off > up)
    return np.where(forced, np.maximum(P, lo), np.where(no_echo, np.clip(pne, 0, up), np.minimum(P, up)))


def main() -> None:
    src, dst = SUB / sys.argv[1], SUB / sys.argv[2]
    sub = pl.read_parquet(src)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    t = (twin_frame(str(RANKING)).join(p_2026(), on="MVT_ID_mvt", how="left")
         .join(sub.select(I64, P=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt"))
    assert t["p"].null_count() == 0
    P = t["P"].to_numpy()
    Q = bounds(P, t["p"].to_numpy(), t["off"].to_numpy(), t["tl"].to_numpy())
    print(f"2026 LIRF NM-unmatched rows with a twin: {t.height}; moved {(np.abs(Q - P) > 0.5).sum()}; "
          f"mean change {np.mean(Q - P):+.0f} s; sum (Q-P)^2 {np.sum((Q - P) ** 2):.3g}")
    upd = t.select("MVT_ID_mvt", new=pl.Series(Q))
    out = (sub.with_columns(_id=I64).join(upd.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
           .with_columns(TAXITIME_SEC_mvt=pl.coalesce(pl.col("new").round(), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).cast(dtype))
           .select(sub.columns))
    tpl = pl.read_parquet(TEMPLATE)
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list()) and float(out["TAXITIME_SEC_mvt"].min()) >= 0
    n_diff = int((out["TAXITIME_SEC_mvt"].cast(pl.Float64) != sub["TAXITIME_SEC_mvt"].cast(pl.Float64)).sum())
    print(f"rows differing from {src.name}: {n_diff}")
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
