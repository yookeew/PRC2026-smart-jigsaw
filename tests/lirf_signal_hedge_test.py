"""Gate for the LIRF NM-unmatched signal hedge (reports/lirf_signal_hedge_preregistration.md).

A  holdout, chain level: q fitted on the 10 non-holdout months, applied to in-scope Jan+Jul 2025 rows of v42's
   holdout chain; full RMSE over all holdout rows, pooled / Jan / Jul.
B  leave-one-month-out, base level: q fitted on the other 11 months, P = base stack; in-scope SSE per month;
   pooled gain with the 3 largest single-row gains removed.

Run:  .venv/Scripts/python.exe -u tests/lirf_signal_hedge_test.py
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
from post.lirf_signals import EVAL, RAW, I64, apply, fit, frame, in_scope, p_2025  # noqa: E402

HOLD = ["2025-01", "2025-07"]


def main() -> None:
    d = frame(str(RAW / "training_*.parquet")).join(p_2025(), on="MVT_ID_mvt", how="left")
    assert d.filter(pl.col("nmu"))["p"].null_count() == 0
    k_all = in_scope(d)
    print(f"LIRF NM-unmatched >= 1 h: {int(d['nmu'].sum()):,}; in scope (1-6 h, p < 0.99): {k_all.sum():,}; "
          f"echo rate in scope {d.filter(pl.Series(k_all))['echo'].mean():.3f}")

    # ---------------------------------------------------------------- A
    clf = fit(d.filter(~pl.col("mt").is_in(HOLD)))
    v42 = pl.read_parquet(EVAL / "v42_chain_holdout.parquet").select(I64, P="v42")
    lab = (pl.scan_parquet([str(RAW / f"training_2025-{a}-01_2025-{b}-01.parquet") for a, b in (("01", "02"), ("07", "08"))])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select(I64, y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), mt=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")).collect())
    h = v42.join(lab, on="MVT_ID_mvt")
    assert h.height == v42.height
    hd = h.join(d.filter(pl.col("mt").is_in(HOLD)).drop("y", "mt"), on="MVT_ID_mvt", how="inner")
    new, k, q = apply(hd, hd["P"].to_numpy(), clf)
    h = h.join(hd.select("MVT_ID_mvt", Pn=pl.Series(new)), on="MVT_ID_mvt", how="left").with_columns(
        Pn=pl.col("Pn").fill_null(pl.col("P")))
    y, P, Pn, mt = h["y"].to_numpy(), h["P"].to_numpy(), h["Pn"].to_numpy(), h["mt"].to_numpy()
    r = lambda z, m: float(np.sqrt(np.mean((y - z)[m] ** 2)))  # noqa: E731
    yk, Pk = hd["y"].to_numpy()[k], hd["P"].to_numpy()[k]
    print(f"\nA. holdout chain (10-month fit). In-scope rows {k.sum()} (echo {hd['echo'].to_numpy()[k].mean():.3f}); "
          f"mean p {hd['p'].to_numpy()[k].mean():.3f} -> q {q[k].mean():.3f}; "
          f"in-scope SSE {np.sum((yk - Pk) ** 2):.4g} -> {np.sum((yk - new[k]) ** 2):.4g}")
    deltas = {}
    for nm, m in (("pooled", np.ones(len(y), bool)), ("Jan", mt == "2025-01"), ("Jul", mt == "2025-07")):
        deltas[nm] = r(Pn, m) - r(P, m)
        print(f"   full RMSE {nm:6s} {r(P, m):8.3f} -> {r(Pn, m):8.3f}  ({deltas[nm]:+.3f})")

    # ---------------------------------------------------------------- B
    print("\nB. leave-one-month-out, base level (P = base stack), in-scope SSE")
    gains, months_better, tot_old, tot_new = [], 0, 0.0, 0.0
    for m in sorted(d["mt"].unique().to_list()):
        dm = d.filter(pl.col("mt") == m)
        new_b, kb, qb = apply(dm, dm["Pb"].to_numpy(), fit(d.filter(pl.col("mt") != m)))
        yb, Pb = dm["y"].to_numpy()[kb], dm["Pb"].to_numpy()[kb]
        so, sn = np.sum((yb - Pb) ** 2), np.sum((yb - new_b[kb]) ** 2)
        gains.append((yb - new_b[kb]) ** 2 - (yb - Pb) ** 2)
        months_better += sn < so
        tot_old, tot_new = tot_old + so, tot_new + sn
        print(f"   {m}: n {kb.sum():4d}  echo {dm['echo'].to_numpy()[kb].mean():.2f}  SSE {so:.4g} -> {sn:.4g}  ({sn / so:.3f})")
    g = np.sort(np.concatenate(gains))
    print(f"   pooled SSE {tot_old:.4g} -> {tot_new:.4g} ({tot_new / tot_old:.3f}); months better {months_better}/12; "
          f"gain without the 3 largest single-row gains {g[3:].sum():.4g}")

    rules = {"1 A pooled full delta < 0": deltas["pooled"] < 0,
             "2 A Jan and Jul <= 0": deltas["Jan"] <= 0 and deltas["Jul"] <= 0,
             "3 B pooled lower and >= 9/12 months": tot_new < tot_old and months_better >= 9,
             "4 B gain < 0 without top-3 rows": g[3:].sum() < 0}
    print()
    for n, ok in rules.items():
        print(f"   rule {n}: {'PASS' if ok else 'FAIL'}")
    print(f"GATE: {'PASS' if all(rules.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
