"""Final step of reports/ctx_armc_preregistration.md: copy the BASE file's values onto LIRF NM-unmatched ranking rows
(team rule, PROGRESS §87), run the submission checks and report the change vs the BASE.

Run:  .venv/Scripts/python.exe src/post/armc_build.py BASE.parquet CHAIN.parquet OUT.parquet
      (paths relative to data/submissions/)
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "data" / "submissions"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
N_ROWS = 344_841


def main() -> None:
    base_f, chain_f, out_f = (SUB / a for a in sys.argv[1:4])
    tpl = pl.read_parquet(TEMPLATE)
    base, chain = pl.read_parquet(base_f), pl.read_parquet(chain_f)
    assert base["TAXITIME_SEC_mvt"].dtype == chain["TAXITIME_SEC_mvt"].dtype
    info = (pl.scan_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP")
            .select("MVT_ID_mvt", apt=pl.col("ADEP_mvt"), nm=pl.col("FLIGHT_ID_mvt").is_null()).collect())
    x = (tpl.select("MVT_ID_mvt")
         .join(chain.rename({"TAXITIME_SEC_mvt": "t"}), on="MVT_ID_mvt", how="left")
         .join(base.rename({"TAXITIME_SEC_mvt": "b"}), on="MVT_ID_mvt", how="left")
         .join(info.with_columns(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left"))
    assert x["t"].null_count() == 0 and x["b"].null_count() == 0 and x["apt"].null_count() == 0
    lirf_nmu = (pl.col("apt") == "LIRF") & pl.col("nm")
    x = x.with_columns(TAXITIME_SEC_mvt=pl.when(lirf_nmu).then(pl.col("b")).otherwise(pl.col("t")))
    out = x.select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
    # checks
    assert out.height == N_ROWS == tpl.height, out.height
    assert set(out["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    assert out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
    k = x.filter(lirf_nmu)
    assert (k["TAXITIME_SEC_mvt"] == k["b"]).all()
    print(f"checks: {out.height:,} rows, ID set = template, no nulls, no negatives; "
          f"LIRF NM-unmatched rows {k.height} identical to {base_f.name} "
          f"(the chain had moved {int((k['t'] != k['b']).sum())} of them)")
    out.write_parquet(out_f)
    print(f"wrote {out_f.name}")
    d = x.with_columns(dd=(pl.col("TAXITIME_SEC_mvt") - pl.col("b")).cast(pl.Float64))
    print(f"vs {base_f.name}: RMS {np.sqrt((d['dd'] ** 2).mean()):.1f} s, mean {d['dd'].mean():+.2f} s, "
          f"rows changed {(d['dd'] != 0).sum():,}")
    g = d.group_by("apt", "nm").agg(n=pl.len(), rms=(pl.col("dd") ** 2).mean().sqrt(), mean=pl.col("dd").mean()).sort("apt", "nm")
    print(f"{'apt':5s} {'NM':>10s} {'n':>7s} {'RMS':>7s} {'mean':>7s}")
    for r in g.iter_rows(named=True):
        print(f"{r['apt']:5s} {'unmatched' if r['nm'] else 'matched':>10s} {r['n']:7d} {r['rms']:7.1f} {r['mean']:+7.1f}")
    for lab, f in (("matched", ~d["nm"]), ("unmatched", d["nm"])):
        print(f"all airports, NM-{lab}: RMS {np.sqrt((d.filter(f)['dd'] ** 2).mean()):.1f} s over {f.sum():,} rows")
    v = out["TAXITIME_SEC_mvt"].to_numpy()
    print(f"rows under 120 s: {(v < 120).sum()} (min {v.min()})")


if __name__ == "__main__":
    main()
