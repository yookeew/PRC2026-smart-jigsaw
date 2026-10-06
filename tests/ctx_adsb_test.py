"""Stage 1 of reports/ctx_adsb_preregistration.md: block 9 (ADS-B track state) in the ensemble2 corrector.

Both arms are the adopted ensemble2 corrector (global + per-airport LightGBM on y - stack, trained on
the 10 OOF months, applied to Jan+Jul 2025) with v37's blocks 1-7. Control = blocks 1-7; treatment =
blocks 1-7 + block 9.

--shift prints only the 2025-vs-2026 check (SMD among matched rows, matched rate per airport). Per the
pre-registration it is run first; columns over the limit go in CTX_ADSB_DROP before stage 1 is run.

Run:  .venv/Scripts/python.exe tests/ctx_adsb_test.py --shift
      CTX_ADSB_DROP=<cols> .venv/Scripts/python.exe tests/ctx_adsb_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ["CTX_RWY"] = "1"
os.environ["CTX_ADSB"] = "1"

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_corrector_test as CC  # noqa: E402

RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
SMD_MAX = 0.25
FLAGS = ("ab_avail", "ab_cov", "ab_matched", "ab_fallback", "ab_tier")


def shift_check() -> None:
    dep, _ = CC.movements(str(CC.RAW / "training_*.parquet"))
    dep = dep.filter(pl.col("m").is_in([1, 7]))
    a = CC.adsb_block(dep, np.zeros(dep.height)).with_columns(m=dep["m"], apt=dep["apt"])
    dr, _ = CC.movements(str(RANKING))
    b = CC.adsb_block(dr, np.zeros(dr.height)).with_columns(m=dr["m"], apt=dr["apt"])
    print("matched rate per airport (2025 Jan/Jul -> 2026 Jan/Jul):")
    for ap in sorted(set(a["apt"])):
        r = [z.filter((pl.col("apt") == ap) & (pl.col("m") == mm))["ab_matched"].mean() for z in (a, b) for mm in (1, 7)]
        print(f"  {ap}  Jan {r[0]:.2f} -> {r[2]:.2f}   Jul {r[1]:.2f} -> {r[3]:.2f}")
    print(f"block-9 shift among matched rows, 2025 vs 2026 (SMD per month; |SMD| > {SMD_MAX} -> drop):")
    drop = []
    for c in a.columns:
        if c in ("m", "apt") or c in FLAGS or c.endswith("_rel"):
            continue
        smd = []
        for mm in (1, 7):
            x = a.filter((pl.col("m") == mm) & (pl.col("ab_matched") == 1))[c].drop_nans().drop_nulls()
            z = b.filter((pl.col("m") == mm) & (pl.col("ab_matched") == 1))[c].drop_nans().drop_nulls()
            sd = np.sqrt((x.var() + z.var()) / 2) or 1.0
            smd.append((z.mean() - x.mean()) / sd)
        bad = max(abs(s) for s in smd) > SMD_MAX
        drop += [c] if bad else []
        print(f"  {c:18s} Jan {smd[0]:+.3f}  Jul {smd[1]:+.3f}{'  DROP' if bad else ''}")
    rel = [c + "_rel" for c in drop if c + "_rel" in a.columns]
    print(f"CTX_ADSB_DROP={','.join(drop + rel)}")


def main() -> None:
    import oof_context_corrector_test as O
    print("dropped by the shift check:", CC.AB_DROP or "none")
    P = O.prepare()
    cols = P["X_full"].columns
    ctrl_cols = [c for c in cols if not c.startswith("ab_")]
    s, y, m, nm, lr, is_ho, inner, apt = (P[k] for k in ("s", "y", "m", "nm", "line_rows", "is_ho", "inner", "apt"))
    fit = (~is_ho) & (~nm) & (y <= O.TRIM) & ~lr
    app = is_ho & ~lr
    win = P["win"]
    print(f"ab_avail share: training rows {P['X_full']['ab_avail'].to_numpy()[fit].mean():.3f}, "
          f"holdout {P['X_full']['ab_avail'].to_numpy()[app].mean():.3f}")
    res = {}
    for arm, cc in (("control", ctrl_cols), ("treatment", cols)):
        X = P["X_full"].select(cc).to_numpy().astype(np.float64)
        cats = [cc.index(c) for c in P["catn"]]
        c, best = CC.fit_apply_ens(X, y - s, fit, inner, app, cats, apt)
        q = s.copy(); q[app] += c
        res[arm] = win(np.clip(q, 0, None))
        print(f"  {arm}: global best_iter {best}", flush=True)
    body = is_ho & (~nm) & (y <= 3600)
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))
    print(f"\n{'slice (holdout)':20s} {'control':>9s} {'treatment':>10s} {'delta':>7s}")
    rows = [("body pooled", body), ("body Jan", body & (m == 1)), ("body Jul", body & (m == 7)),
            ("trimmed", is_ho & (y <= O.TRIM))] + [(f"  body {a}", body & (apt == a)) for a in sorted(set(apt))]
    for name, k in rows:
        a1, a2 = r(res["control"], k), r(res["treatment"], k)
        print(f"{name:20s} {a1:9.2f} {a2:10.2f} {a2 - a1:+7.2f}")
    g = {n: r(res["control"], k) - r(res["treatment"], k) for n, k in (("pooled", body), ("Jan", body & (m == 1)), ("Jul", body & (m == 7)))}
    tk = is_ho & (y <= O.TRIM)
    ok = g["pooled"] >= 0.5 and g["Jan"] > 0 and g["Jul"] > 0 and r(res["treatment"], tk) <= r(res["control"], tk)
    print(f"\nSTAGE 1: body gain {g} -> {'PROCEED to stage 2' if ok else 'STOP'}")


if __name__ == "__main__":
    shift_check() if "--shift" in sys.argv else main()
