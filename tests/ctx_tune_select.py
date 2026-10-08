"""Selection stage of reports/ctx_tune_preregistration.md: corrector parameter configurations scored on Feb + Aug 2025
(OOF months, never the holdout). Arm C c1 inputs and training rows, r member (blocks 1-7), ensemble2.

Run:  ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 .venv/Scripts/python.exe -u tests/ctx_tune_select.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from post import ctx_armc as A  # noqa: E402

VAL = (2, 8)
BASE = dict(num_leaves=63, learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.9, lambda_l2=10.0)
CONFIGS = {
    "c0": {},
    "c0b": {},          # refit-noise yardstick (A1)
    "c1": dict(num_leaves=127),
    "c2": dict(num_leaves=255, min_data_in_leaf=400),
    "c3": dict(num_leaves=31),
    "c4": dict(learning_rate=0.025),
    "c5": dict(feature_fraction=0.7),
    "c6": dict(min_data_in_leaf=50),
    "c7": dict(lambda_l2=1.0),
    "c8": dict(num_leaves=127, learning_rate=0.025, feature_fraction=0.7),
}


def main() -> None:
    import context_corrector_test as CC
    import oof_context_corrector_test as O
    P = O.prepare()
    dep, s, y, m, nm, lr, is_ho, inner, apt = (P[k] for k in ("dep", "s", "y", "m", "nm", "line_rows", "is_ho", "inner", "apt"))
    Xdf = A.add_arm(P["X_full"], dep, None)
    cols = Xdf.columns
    X = Xdf.to_numpy().astype(np.float64)
    cats = [cols.index(c) for c in P["catn"]]
    trainable = (~is_ho) & A.arm_fit((~nm) & (y <= O.TRIM) & ~lr, nm, lr, s, y)
    val = (~is_ho) & np.isin(m, VAL)
    fit = trainable & ~val
    app = val & ~lr
    body = val & ~nm & (y <= 3600)
    trim = val & (y <= O.TRIM)
    print(f"fit rows {fit.sum():,}; validation rows {val.sum():,} (body {body.sum():,})", flush=True)
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))  # noqa: E731
    q0 = P["win"](np.clip(s, 0, None))
    print(f"stack alone: body {r(q0, body):.2f}  Feb {r(q0, body & (m == 2)):.2f}  Aug {r(q0, body & (m == 8)):.2f}", flush=True)
    res = {}
    for cid, ch in CONFIGS.items():
        t0 = time.time()
        g = {**BASE, **ch}
        CC.PARAMS = {**CC.PARAMS, **g}
        CC.PARAMS_APT = {**CC.PARAMS, "min_data_in_leaf": max(g["min_data_in_leaf"] // 2, 20)}
        c, best = CC.fit_apply_ens(X, y - s, fit, inner, app, cats, apt)
        q = s.copy(); q[app] += c
        q = P["win"](np.clip(q, 0, None))
        res[cid] = (r(q, body), r(q, body & (m == 2)), r(q, body & (m == 8)), r(q, trim))
        print(f"{cid} {ch}: best_iter {best}; body {res[cid][0]:.2f}  Feb {res[cid][1]:.2f}  Aug {res[cid][2]:.2f}  trimmed {res[cid][3]:.2f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    b0 = res["c0"]
    noise = abs(res["c0"][0] - res["c0b"][0])
    print(f"\nrefit noise |c0 - c0b| pooled body: {noise:.3f}")
    best = min((k for k in res if k != "c0b"), key=lambda k: res[k][0])
    gain = [b0[i] - res[best][i] for i in range(3)]
    ok = best != "c0" and gain[0] >= 0.3 and gain[1] > 0 and gain[2] > 0 and gain[0] > 2 * noise
    print(f"\nchosen {best}: gain vs c0 body {gain[0]:+.2f} (Feb {gain[1]:+.2f}, Aug {gain[2]:+.2f}) -> {'GO to the holdout gate' if ok else 'STOP'}")


if __name__ == "__main__":
    main()
