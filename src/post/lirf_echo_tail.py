"""LIRF NM-unmatched departures >= 12 h late: predict the schedule echo (reports/lirf_echo_tail_preregistration.md).

P = T - SCHED + e_T, e_T = MVT_TIME's seconds offset from the nearest minute (LIRF block times carry -e_T, so the
echo label is exact to +-1 s in 2025). Every other row is copied from the input file.

Run:  .venv/Scripts/python.exe -u src/post/lirf_echo_tail.py IN.parquet OUT.parquet
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
SUB = ROOT / "data" / "submissions"
MIN_OFF = 12 * 3600
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)


def main() -> None:
    src, dst = SUB / sys.argv[1], SUB / sys.argv[2]
    sub = pl.read_parquet(src)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    T, S = pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s")
    rk = (pl.read_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP")
          .select(I64, "FLIGHT_mvt", sel=(pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null() & ((T - S) >= MIN_OFF),
                  echo=(T - S + (T + 30) % 60 - 30).cast(pl.Float64)))
    x = sub.with_columns(_id=I64).join(rk.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left")
    assert x["sel"].null_count() == 0
    old = x["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    k = x["sel"].to_numpy()
    new = np.where(k, x["echo"].to_numpy(), old)
    for f, o, n in zip(x.filter(pl.col("sel"))["FLIGHT_mvt"].to_list(), old[k], new[k]):
        print(f"  {str(f):10s} {o:9.0f} -> {n:9.0f}  ({n - o:+.0f})")
    print(f"rows changed {int((new != old).sum())} of {k.sum()} selected; if all are echoes: dSSE {-np.sum((old[k] - new[k]) ** 2):.4g}")
    out = x.with_columns(TAXITIME_SEC_mvt=pl.Series(np.round(new)).cast(dtype)).select(sub.columns)
    tpl = pl.read_parquet(TEMPLATE)
    assert out.height == tpl.height == 344_841 and out["TAXITIME_SEC_mvt"].null_count() == 0
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list()) and float(out["TAXITIME_SEC_mvt"].min()) >= 0
    out.write_parquet(dst)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
