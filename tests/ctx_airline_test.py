"""Stage 1 of reports/ctx_airline_preregistration.md: block 8 (airline + stand rotation) in the corrector.

Both arms are the ensemble2 corrector with block 7 (runway) on, trained on the 10 OOF months and applied
to Jan+Jul 2025. Control = blocks 1-7; treatment = blocks 1-8. Also prints a 2025-vs-2026 shift check
(SMD) of the block-8 columns.

Run:  .venv/Scripts/python.exe tests/ctx_airline_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ["CTX_RWY"] = "1"
os.environ["CTX_AIR"] = "1"
os.environ["CTX_AIR_EXCL"] = "1,7"   # amendment 2026-10-05: holdout months never enter the airline table

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_corrector_test as CC  # noqa: E402
import oof_context_corrector_test as O  # noqa: E402

RANKING = ROOT / "data" / "ranking" / "ranking.parquet"


def shift_check() -> None:
    # p = 0 on both sides: al_res is then the airline's taxi level, so the check measures the operator mix
    dep, _ = CC.movements(str(CC.RAW / "training_*.parquet"))
    a = CC.airline_block(dep, np.zeros(dep.height)).with_columns(m=dep["m"]).filter(pl.col("m").is_in([1, 7]))
    dr, _ = CC.movements(str(RANKING))
    b = CC.airline_block(dr, np.zeros(dr.height)).with_columns(m=dr["m"])
    print("block-8 shift, 2025 vs 2026 (SMD per month):")
    for c in a.columns:
        if c in ("m", "al_op"):
            continue
        smd = []
        for mm in (1, 7):
            x, z = a.filter(pl.col("m") == mm)[c].drop_nans().drop_nulls(), b.filter(pl.col("m") == mm)[c].drop_nans().drop_nulls()
            sd = np.sqrt((x.var() + z.var()) / 2) or 1.0
            smd.append((z.mean() - x.mean()) / sd)
        print(f"  {c:14s} Jan {smd[0]:+.3f}  Jul {smd[1]:+.3f}")


def main() -> None:
    shift_check()
    P = O.prepare()
    cols = P["X_full"].columns
    ctrl_cols = [c for c in cols if not c.startswith(("al_", "ro_"))]
    s, y, m, nm, lr, is_ho, inner, apt = (P[k] for k in ("s", "y", "m", "nm", "line_rows", "is_ho", "inner", "apt"))
    fit = (~is_ho) & (~nm) & (y <= O.TRIM) & ~lr
    app = is_ho & ~lr
    win = P["win"]
    rot_cols = [c for c in cols if not c.startswith("al_")]
    res = {}
    for arm, cc in (("control", ctrl_cols), ("rotation", rot_cols), ("treatment", cols)):
        X = P["X_full"].select(cc).to_numpy().astype(np.float64)
        cats = [cc.index(c) for c in P["catn"] if c in cc]
        c, best = CC.fit_apply_ens(X, y - s, fit, inner, app, cats, apt)
        q = s.copy(); q[app] += c
        res[arm] = win(np.clip(q, 0, None))
        print(f"  {arm}: global best_iter {best}", flush=True)
    body = is_ho & (~nm) & (y <= 3600)
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))
    print(f"\n{'slice (holdout)':20s} {'control':>9s} {'rotation':>9s} {'treatment':>10s} {'rot-c':>7s} {'t-c':>7s}")
    rows = [("body pooled", body), ("body Jan", body & (m == 1)), ("body Jul", body & (m == 7)),
            ("trimmed", is_ho & (y <= O.TRIM))] + [(f"  body {a}", body & (apt == a)) for a in sorted(set(apt))]
    for name, k in rows:
        a1, a0, a2 = r(res["control"], k), r(res["rotation"], k), r(res["treatment"], k)
        print(f"{name:20s} {a1:9.2f} {a0:9.2f} {a2:10.2f} {a0 - a1:+7.2f} {a2 - a1:+7.2f}")
    tk = is_ho & (y <= O.TRIM)
    for arm in ("rotation", "treatment"):
        g = {n: r(res["control"], k) - r(res[arm], k) for n, k in (("pooled", body), ("Jan", body & (m == 1)), ("Jul", body & (m == 7)))}
        ok = g["pooled"] >= 0.5 and g["Jan"] > 0 and g["Jul"] > 0 and r(res[arm], tk) <= r(res["control"], tk)
        print(f"STAGE 1 {arm}: body gain {g} -> {'PROCEED to stage 2' if ok else 'STOP'}")


if __name__ == "__main__":
    main()
