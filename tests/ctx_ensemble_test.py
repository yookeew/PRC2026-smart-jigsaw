"""Corrector ensemble screen (reports/ctx_ensemble_preregistration.md).

Same data, features and training rows as stage 1 of reports/oof_context_corrector_preregistration.md
(corrector on y - stack, trained on the 10 OOF months, applied to Jan+Jul 2025). Arms:
  single   : one global LightGBM (the adopted corrector)
  ensemble : equal-weight mean of the global LightGBM, one LightGBM per airport and a CatBoost
Decision on holdout body RMSE (NM-matched, y <= 1 h).

Run:  .venv/Scripts/python.exe tests/ctx_ensemble_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_corrector_test as CC  # noqa: E402
import oof_context_corrector_test as O  # noqa: E402

PARAMS_APT = {**CC.PARAMS, "min_data_in_leaf": 100}


def lgb_rounds(X, y, fit, inner, cats, params):
    tr = lgb.Dataset(X[fit & ~inner], y[fit & ~inner], categorical_feature=cats, free_raw_data=False)
    va = lgb.Dataset(X[fit & inner], y[fit & inner], categorical_feature=cats, reference=tr, free_raw_data=False)
    m = lgb.train(params, tr, 3000, valid_sets=[va], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(m.best_iteration, 1)
    return lgb.train(params, lgb.Dataset(X[fit], y[fit], categorical_feature=cats), best), best


def main() -> None:
    P = O.prepare()
    X = P["X_full"].to_numpy().astype(np.float64)
    cols = P["X_full"].columns
    cats = [cols.index(c) for c in P["catn"]]
    s, y, m, nm, lr, is_ho, inner, apt = (P[k] for k in ("s", "y", "m", "nm", "line_rows", "is_ho", "inner", "apt"))
    fit = (~is_ho) & (~nm) & (y <= O.TRIM) & ~lr
    app = is_ho & ~lr
    tgt = y - s
    g, bg = lgb_rounds(X, tgt, fit, inner, cats, CC.PARAMS)
    c_glob = g.predict(X[app])
    print(f"global: best_iter {bg}", flush=True)
    c_apt = c_glob.copy()
    apt_app = apt[app]
    for a in np.unique(apt):
        k = fit & (apt == a)
        if k.sum() < 20000:
            continue
        ma, ba = lgb_rounds(X, tgt, k, inner, cats, PARAMS_APT)
        sel = apt_app == a
        c_apt[sel] = ma.predict(X[app][sel])
        print(f"  {a}: best_iter {ba}", flush=True)
    try:
        from catboost import CatBoostRegressor, Pool
    except ModuleNotFoundError:
        CatBoostRegressor = None
    if CatBoostRegressor is None:
        print("catboost not installed: three-model arm skipped", flush=True)
        c_cat = None
    else:
        Xc = P["X_full"].with_columns([pl.col(c).fill_null(-1).cast(pl.Int64).cast(pl.Utf8) for c in P["catn"]]).to_pandas()
        trn, val = np.flatnonzero(fit & ~inner), np.flatnonzero(fit & inner)
        cb = CatBoostRegressor(iterations=3000, depth=8, learning_rate=0.08, loss_function="RMSE", random_seed=0,
                               od_type="Iter", od_wait=100, verbose=0, thread_count=-1)
        cb.fit(Pool(Xc.iloc[trn], tgt[trn], cat_features=P["catn"]), eval_set=Pool(Xc.iloc[val], tgt[val], cat_features=P["catn"]))
        bc = cb.get_best_iteration()
        cb = CatBoostRegressor(iterations=max(bc, 1), depth=8, learning_rate=0.08, loss_function="RMSE", random_seed=0,
                               verbose=0, thread_count=-1)
        cb.fit(Pool(Xc.iloc[np.flatnonzero(fit)], tgt[fit], cat_features=P["catn"]))
        c_cat = cb.predict(Xc.iloc[np.flatnonzero(app)])
        print(f"catboost: best_iter {bc}", flush=True)
    win = P["win"]
    def mk(c):
        q = s.copy()
        q[app] += c
        return win(np.clip(q, 0, None))
    arms = {"stack": s, "single": mk(c_glob), "per-airport": mk(c_apt), "ensemble2": mk((c_glob + c_apt) / 2)}
    if c_cat is not None:
        arms.update({"catboost": mk(c_cat), "ensemble": mk((c_glob + c_apt + c_cat) / 3)})
    body = is_ho & (~nm) & (y <= 3600)
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))
    print(f"\n{'arm':12s} {'body':>8s} {'Jan':>8s} {'Jul':>8s} {'trimmed':>9s}")
    for a, q in arms.items():
        print(f"{a:12s} {r(q, body):8.2f} {r(q, body & (m == 1)):8.2f} {r(q, body & (m == 7)):8.2f} {r(q, is_ho & (y <= O.TRIM)):9.2f}")
    sgl = arms["single"]
    for name in ("ensemble2", "ensemble"):
        if name not in arms:
            continue
        e = arms[name]
        g = {n: r(sgl, k) - r(e, k) for n, k in (("pooled", body), ("Jan", body & (m == 1)), ("Jul", body & (m == 7)))}
        ok = g["pooled"] >= 1.0 and g["Jan"] > 0 and g["Jul"] > 0 and r(e, is_ho & (y <= O.TRIM)) <= r(sgl, is_ho & (y <= O.TRIM))
        print(f"DECISION {name}: vs single body gain {g} -> {'ADOPT' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
