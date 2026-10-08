"""Measurement of reports/celn_scale_preregistration.md: 2026 corrector output, cel_n-scaled treatment vs control
(both Arm C c1 members built on this machine from identical inputs), after the ctxsel selection (rb at
EHAM/LSZH/EDDF/LEBL, r elsewhere). Also the control vs v42's own corrector (teammate's machine) for scale.

Run:  .venv/Scripts/python.exe -u tests/celn_scale_compare.py [CONTROL_TAG TREATMENT_TAG]   (A1: _1fitctl _1fitceln)
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctx_adsb_stable_test as ST  # noqa: E402

SUB, EVAL = ROOT / "data" / "submissions", ROOT / "cache" / "eval"


def sel(tag: str) -> pl.DataFrame:
    s, _ = ST._select(pl.read_parquet(SUB / f"ctxc1r{tag}.parquet"), pl.read_parquet(SUB / f"ctxc1rb{tag}.parquet"), "TAXITIME_SEC_mvt", ST._airports())
    return s.select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(tag))


def main() -> None:
    t0, t1 = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("_ctl", "_celn")
    d = sel(t0).rename({t0: "_ctl"}).join(sel(t1).rename({t1: "_celn"}), on="MVT_ID_mvt").join(
        pl.read_parquet(SUB / "ctxc1sel.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), v42=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
    info = pl.read_parquet(ROOT / "data" / "ranking" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null())
    d = d.join(info, on="MVT_ID_mvt")
    dd = (d["_celn"] - d["_ctl"]).to_numpy()
    print(f"rows {d.height:,}")
    print(f"treatment vs control (corrector output, ctxsel): RMS {np.sqrt(np.mean(dd ** 2)):.2f} s, mean {dd.mean():+.2f} s, "
          f"rows moved > 1 s {np.mean(np.abs(dd) > 1):.3f}, > 30 s {np.mean(np.abs(dd) > 30):.4f}, max |d| {np.abs(dd).max():.0f}")
    dv = (d["_ctl"] - d["v42"]).to_numpy()
    print(f"for scale: control (this machine) vs v42's ctxc1sel (teammate's machine): RMS {np.sqrt(np.mean(dv ** 2)):.2f} s, mean {dv.mean():+.2f} s")
    print("\nby airport: n, RMS(treatment - control), mean")
    for a in sorted(d["apt"].unique()):
        k = (d["apt"] == a).to_numpy()
        print(f"  {a}  {k.sum():6,}  {np.sqrt(np.mean(dd[k] ** 2)):7.2f}  {dd[k].mean():+7.2f}")
    k = d["nmu"].to_numpy()
    print(f"NM-matched RMS {np.sqrt(np.mean(dd[~k] ** 2)):.2f}; NM-unmatched RMS {np.sqrt(np.mean(dd[k] ** 2)):.2f}")
    print(f"\nDECISION (pre-registered): RMS {np.sqrt(np.mean(dd ** 2)):.2f} s {'< 3 s -> CLOSE' if np.sqrt(np.mean(dd ** 2)) < 3 else '>= 3 s -> build v43 candidate'}")


if __name__ == "__main__":
    main()
