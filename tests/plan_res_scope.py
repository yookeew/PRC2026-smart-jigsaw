"""Scoping: is there taxi information in the filed NM flight duration that v42 does not use? (PROGRESS.md §106)

plan_res = (ARVT_1 - EOBT_1) - median over the 10 non-holdout 2025 months of the same (ADEP, ADES, AIRCRAFT_TYPE_flt)
route (>= 10 rows; else (ADEP, ADES)). The filed M1 duration = planned taxi-out + EET (+ planned taxi-in), so the
route-relative part carries the planned taxi time (kind-mango's plan_nm_taxi, v65: -1.22 s live). The corrector already
reads the raw duration (nm_dur_plan) but has no route identity. Also real_res = (ARVT_3 - T) - its route median
(flown airborne time), as a control that should carry little taxi-out information.

On v42's holdout (NM-matched, y <= 1 h): correlation of plan_res with the residual y - v42, per airport and month;
the binned mean residual; and a cross-month linear ceiling (residual ~ b * clip(plan_res) fitted per airport on one
month, applied to the other).

Run:  .venv/Scripts/python.exe -u tests/plan_res_scope.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
I64 = pl.col("MVT_ID_mvt").cast(pl.Int64)
CLIPV = 1800.0


def frame(src) -> pl.DataFrame:
    return (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_not_null())
            .select(I64, apt="ADEP_mvt", ades="ADES_flt", typ="AIRCRAFT_TYPE_flt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                    dur=(pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64),
                    air=(pl.col("ARVT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
            .collect())


def add_res(d: pl.DataFrame, ref: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    r = ref.filter(pl.col(col).is_not_null() & pl.col(col).is_between(0, 86400))
    m3 = r.group_by("apt", "ades", "typ").agg(pl.col(col).median().alias("_m3"), pl.len().alias("_n3")).filter(pl.col("_n3") >= 10)
    m2 = r.group_by("apt", "ades").agg(pl.col(col).median().alias("_m2"), pl.len().alias("_n2")).filter(pl.col("_n2") >= 10)
    x = d.join(m3, on=["apt", "ades", "typ"], how="left").join(m2, on=["apt", "ades"], how="left")
    return x.with_columns((pl.col(col) - pl.coalesce("_m3", "_m2")).alias(out)).drop("_m3", "_n3", "_m2", "_n2")


def main() -> None:
    d = frame(str(ROOT / "data" / "raw" / "training_*.parquet"))
    ref = d.filter(~pl.col("mo").is_in([1, 7]))
    ho = d.filter(pl.col("mo").is_in([1, 7])).join(
        pl.read_parquet(ROOT / "cache" / "eval" / "v42_chain_holdout.parquet").select(I64, P="v42"), on="MVT_ID_mvt")
    ho = add_res(add_res(ho, ref, "dur", "plan_res"), ref, "air", "real_res").filter(pl.col("y") <= 3600)
    ho = ho.with_columns(r=pl.col("y") - pl.col("P"))
    print(f"holdout NM-matched y <= 1 h: {ho.height:,}; plan_res present {ho['plan_res'].is_not_null().mean():.3f}, "
          f"real_res present {ho['real_res'].is_not_null().mean():.3f}")
    # 2026 coverage
    rk = (pl.scan_parquet(str(ROOT / "data" / "ranking" / "ranking.parquet")).filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_not_null())
          .select(apt="ADEP_mvt", ades="ADES_flt", typ="AIRCRAFT_TYPE_flt", dur=(pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64)).collect())
    rk = add_res(rk, d, "dur", "plan_res")
    print(f"2026 NM-matched: plan_res present {rk['plan_res'].is_not_null().mean():.3f}; sd 2025 holdout {ho['plan_res'].std():.0f} vs 2026 {rk['plan_res'].std():.0f}; "
          f"median {ho['plan_res'].median():.0f} vs {rk['plan_res'].median():.0f}")

    print("\ncorrelation with the v42 residual y - P (NM-matched, y <= 1 h), per airport: plan_res Jan / Jul | real_res Jan / Jul | n")
    for a in sorted(ho["apt"].unique()):
        row = []
        for c in ("plan_res", "real_res"):
            for m in (1, 7):
                s = ho.filter((pl.col("apt") == a) & (pl.col("mo") == m) & pl.col(c).is_not_null())
                x = np.clip(s[c].to_numpy(), -CLIPV, CLIPV)
                row.append(np.corrcoef(x, s["r"].to_numpy())[0, 1] if s.height > 100 else np.nan)
        print(f"  {a}  {row[0]:+.3f} / {row[1]:+.3f}  |  {row[2]:+.3f} / {row[3]:+.3f}  |  {ho.filter(pl.col('apt') == a).height:,}")

    print("\nmean v42 residual by plan_res band (pooled):")
    s = ho.filter(pl.col("plan_res").is_not_null())
    for lo, hi in ((-1e9, -600), (-600, -300), (-300, -120), (-120, -30), (-30, 30), (30, 120), (120, 300), (300, 600), (600, 1e9)):
        k = s.filter((pl.col("plan_res") > lo) & (pl.col("plan_res") <= hi))
        if k.height:
            print(f"  ({lo:>6.0f}, {hi:>6.0f}]  n {k.height:7,}  mean residual {k['r'].mean():+7.1f}  (Jan {k.filter(pl.col('mo') == 1)['r'].mean():+6.1f}, "
                  f"Jul {k.filter(pl.col('mo') == 7)['r'].mean():+6.1f})")

    print("\ncross-month linear ceiling, per airport slope on clip(plan_res, +-1800), body RMSE (NM-matched, y <= 1 h):")
    for fm, am in ((1, 7), (7, 1)):
        k = ho.filter(pl.col("mo") == am)
        adj = np.zeros(k.height)
        for a in sorted(ho["apt"].unique()):
            f = ho.filter((pl.col("mo") == fm) & (pl.col("apt") == a) & pl.col("plan_res").is_not_null())
            x = np.clip(f["plan_res"].to_numpy(), -CLIPV, CLIPV)
            b = float(np.sum(x * f["r"].to_numpy()) / max(np.sum(x * x), 1))
            ka = (k["apt"] == a).to_numpy() & k["plan_res"].is_not_null().to_numpy()
            adj[ka] = b * np.clip(k["plan_res"].to_numpy()[ka], -CLIPV, CLIPV)
        r = k["r"].to_numpy()
        print(f"  fit {fm} -> apply {am}: {np.sqrt(np.mean(r ** 2)):.2f} -> {np.sqrt(np.mean((r - adj) ** 2)):.2f}")


if __name__ == "__main__":
    main()
