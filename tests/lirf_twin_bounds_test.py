"""Gate for the LIRF twin-LOBT bounds (reports/lirf_twin_bounds_preregistration.md).

A  holdout chain (v42): full RMSE over all holdout rows, pooled / Jan / Jul.
B  12 months, base stack: SSE per month on the twin rows; pooled gain without the 3 largest single-row gains.

Run:  .venv/Scripts/python.exe -u tests/lirf_twin_bounds_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from post.lirf_signals import EVAL, I64, RAW, p_2025  # noqa: E402
from post.lirf_twin_bounds import bounds, twin_frame  # noqa: E402


def main() -> None:
    t = twin_frame(str(RAW / "training_*.parquet")).join(p_2025(), on="MVT_ID_mvt", how="left")
    print(f"2025 LIRF NM-unmatched rows with a twin: {t.height}")

    # A: holdout chain
    v42 = pl.read_parquet(EVAL / "v42_chain_holdout.parquet").select(I64, P="v42")
    lab = (pl.scan_parquet([str(RAW / f"training_2025-{a}-01_2025-{b}-01.parquet") for a, b in (("01", "02"), ("07", "08"))])
           .filter(pl.col("PHASE_mvt") == "DEP").select(I64, yy=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mm=pl.col("MVT_TIME_UTC_mvt").dt.month()).collect())
    h = v42.join(lab, on="MVT_ID_mvt")
    ht = h.join(t.drop("y", "m"), on="MVT_ID_mvt")
    Q = bounds(ht["P"].to_numpy(), ht["p"].to_numpy(), ht["off"].to_numpy(), ht["tl"].to_numpy())
    h = h.join(ht.select("MVT_ID_mvt", Q=pl.Series(Q)), on="MVT_ID_mvt", how="left").with_columns(Q=pl.col("Q").fill_null(pl.col("P")))
    y, P, Qa, mm = h["yy"].to_numpy(), h["P"].to_numpy(), h["Q"].to_numpy(), h["mm"].to_numpy()
    r = lambda z, k: float(np.sqrt(np.mean((y - z)[k] ** 2)))  # noqa: E731
    print(f"\nA. holdout chain: twin rows {ht.height}, moved {(np.abs(Q - ht['P'].to_numpy()) > 0.5).sum()}")
    dA = {}
    for nm, k in (("pooled", np.ones(len(y), bool)), ("Jan", mm == 1), ("Jul", mm == 7)):
        dA[nm] = r(Qa, k) - r(P, k)
        print(f"   full RMSE {nm:6s} {r(P, k):8.3f} -> {r(Qa, k):8.3f}  ({dA[nm]:+.3f})")

    # B: 12 months, base stack
    z = t.filter(pl.col("Pb").is_not_null())
    yb, Pb = z["y"].to_numpy(), z["Pb"].to_numpy()
    Qb = bounds(Pb, z["p"].to_numpy(), z["off"].to_numpy(), z["tl"].to_numpy())
    d, mb = (yb - Qb) ** 2 - (yb - Pb) ** 2, z["m"].to_numpy()
    better = sum(d[mb == m].sum() < 0 for m in range(1, 13))
    worse = sum(d[mb == m].sum() > 0 for m in range(1, 13))
    g = np.sort(d)
    print(f"\nB. 12 months base stack: rows {len(yb)}, SSE {np.sum((yb - Pb) ** 2):.4g} -> {np.sum((yb - Qb) ** 2):.4g}; "
          f"months better {better} / worse {worse}; gain without top-3 rows {g[3:].sum():.4g}")

    rules = {"A pooled < 0": dA["pooled"] < 0, "A Jan and Jul <= 0": dA["Jan"] <= 0 and dA["Jul"] <= 0,
             "B >= 9/12 months better": better >= 9, "B gain < 0 without top-3": g[3:].sum() < 0}
    for n, ok in rules.items():
        print(f"   rule {n}: {'PASS' if ok else 'FAIL'}")
    print(f"GATE: {'PASS' if all(rules.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
