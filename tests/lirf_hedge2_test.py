"""Gate for LIRF hedge round 2 (reports/lirf_hedge2_preregistration.md).

A  holdout chain: v42 holdout chain + v45 treatment (10-month fits) vs the same + this treatment; full RMSE pooled/Jan/Jul.
B  leave-one-month-out at base level (base = v45 treatment on the base stack): in-scope SSE per month.

Run:  .venv/Scripts/python.exe -u tests/lirf_hedge2_test.py
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
import post.lirf_signals as L  # noqa: E402
from post.lirf_hedge2 import airline_late, fit2, frame25, treat, with_loo_rate  # noqa: E402
from post.lirf_twin_bounds import twin_frame  # noqa: E402

HOLD = ["2025-01", "2025-07"]


def main() -> None:
    src = str(L.RAW / "training_*.parquet")
    hist = airline_late(src)
    d = with_loo_rate(frame25(), hist)
    twins = set(twin_frame(src)["MVT_ID_mvt"].to_list())
    d = d.with_columns(twin=pl.col("MVT_ID_mvt").is_in(list(twins)))
    print(f"LIRF NM-unmatched >= 1 h rows {int(d['nmu'].sum())}; twin rows excluded from scope {int((d['twin'] & d['nmu']).sum())}")

    # A: holdout chain. Both arms start from v42's chain with v45's treatment (10-month fit).
    tr10 = d.filter(~pl.col("mt").is_in(HOLD))
    clf45, clf2 = L.fit(tr10), fit2(tr10)
    v42 = pl.read_parquet(L.EVAL / "v42_chain_holdout.parquet").select(L.I64, P0="v42")
    lab = (pl.scan_parquet([str(L.RAW / f"training_2025-{a}-01_2025-{b}-01.parquet") for a, b in (("01", "02"), ("07", "08"))])
           .filter(pl.col("PHASE_mvt") == "DEP").select(L.I64, yy=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mm=pl.col("MVT_TIME_UTC_mvt").dt.month()).collect())
    h = v42.join(lab, on="MVT_ID_mvt")
    hd = h.join(d.filter(pl.col("mt").is_in(HOLD)).drop("y"), on="MVT_ID_mvt")
    base, _, _ = L.apply(hd, hd["P0"].to_numpy(), clf45)
    new, k = treat(hd, base, clf45, clf2, hd["twin"].to_numpy())
    h = h.join(hd.select("MVT_ID_mvt", B=pl.Series(base), Nw=pl.Series(new)), on="MVT_ID_mvt", how="left").with_columns(
        B=pl.col("B").fill_null(pl.col("P0")), Nw=pl.col("Nw").fill_null(pl.col("P0")))
    y, B, Nw, mm = h["yy"].to_numpy(), h["B"].to_numpy(), h["Nw"].to_numpy(), h["mm"].to_numpy()
    r = lambda z, s: float(np.sqrt(np.mean((y - z)[s] ** 2)))  # noqa: E731
    print(f"\nA. holdout chain: scope rows {k.sum()}")
    dA = {}
    for nm, s in (("pooled", np.ones(len(y), bool)), ("Jan", mm == 1), ("Jul", mm == 7)):
        dA[nm] = r(Nw, s) - r(B, s)
        print(f"   full RMSE {nm:6s} {r(B, s):8.3f} -> {r(Nw, s):8.3f}  ({dA[nm]:+.3f})")

    # B: leave-one-month-out at base level
    print("\nB. leave-one-month-out, base level, in-scope SSE (base = v45 treatment)")
    better, so_t, sn_t, gains = 0, 0.0, 0.0, []
    for m in sorted(d["mt"].unique().to_list()):
        tr, te = d.filter(pl.col("mt") != m), d.filter(pl.col("mt") == m)
        c45, c2 = L.fit(tr), fit2(tr)
        b, _, _ = L.apply(te, te["Pb"].to_numpy(), c45)
        n, kb = treat(te, b, c45, c2, te["twin"].to_numpy())
        yb = te["y"].to_numpy()
        so, sn = np.sum(((yb - b) ** 2)[kb]), np.sum(((yb - n) ** 2)[kb])
        gains.append(((yb - n) ** 2 - (yb - b) ** 2)[kb])
        better += sn < so
        so_t, sn_t = so_t + so, sn_t + sn
        print(f"   {m}: n {kb.sum():4d}  SSE {so:.4g} -> {sn:.4g}  ({sn / so:.3f})")
    g = np.sort(np.concatenate(gains))
    print(f"   pooled {so_t:.4g} -> {sn_t:.4g} ({sn_t / so_t:.3f}); months better {better}/12; gain without top-3 {g[3:].sum():.4g}")
    rules = {"A pooled < 0": dA["pooled"] < 0, "A Jan and Jul <= 0": dA["Jan"] <= 0 and dA["Jul"] <= 0,
             "B >= 8/12 and pooled lower": better >= 8 and sn_t < so_t, "B gain < 0 without top-3": g[3:].sum() < 0}
    for n_, ok in rules.items():
        print(f"   rule {n_}: {'PASS' if ok else 'FAIL'}")
    print(f"GATE: {'PASS' if all(rules.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
