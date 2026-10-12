"""Scoping (no gate, no build): is an XGBoost corrector diverse enough from the LightGBM one to help in a blend?

Same inputs and training rows as `ctx_armc.py inputs` for the current member (env flags as for that member, e.g.
ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_FLOW=1 PRC_ARMC_TAG=_rf). Fit on the 10 OOF months
(holdout rows never in training), early stopping on the same inner day split, apply to the Jan/Jul holdout; compare with
the LightGBM member's holdout output (cache/eval/<NAME>_holdout_ev.parquet) and their 50/50 mean. Corrector level.

Run:  <member env> .venv/Scripts/python.exe tests/xgb_corrector_scope.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "post")); sys.path.insert(0, str(ROOT / "tests"))
import ctx_armc as A  # noqa: E402
import oof_context_corrector_test as O  # noqa: E402

PARAMS = dict(objective="reg:squarederror", tree_method="hist", device=os.environ.get("XGB_DEVICE", "cuda"),
              max_depth=int(os.environ.get("XGB_DEPTH", "9")), learning_rate=0.05, min_child_weight=200,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, max_bin=127, seed=42)


def main() -> None:
    t0 = time.time()
    P = O.prepare()
    dep, s, y, m, nm, lr, is_ho, inner = (P[k] for k in ("dep", "s", "y", "m", "nm", "line_rows", "is_ho", "inner"))
    Xdf = A.add_arm(P["X_full"], dep, None)
    X = Xdf.to_numpy().astype(np.float32)
    base_fit = (~nm) & (y <= O.TRIM) & ~lr
    trainable = (~is_ho) & A.arm_fit(base_fit, nm, lr, s, y)
    app = is_ho & ~lr
    print(f"{A.NAME}: features {X.shape[1]}, trainable {trainable.sum():,}, build {time.time() - t0:.0f}s", flush=True)
    r = y - s
    dtr = xgb.DMatrix(X[trainable & ~inner], r[trainable & ~inner], missing=np.nan)
    dva = xgb.DMatrix(X[trainable & inner], r[trainable & inner], missing=np.nan)
    t1 = time.time()
    b = xgb.train(PARAMS, dtr, 4000, evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
    best = b.best_iteration + 1
    full = xgb.train(PARAMS, xgb.DMatrix(X[trainable], r[trainable], missing=np.nan), best)
    c = full.predict(xgb.DMatrix(X[app], missing=np.nan))
    print(f"xgb best_iter {best}, fit {time.time() - t1:.0f}s", flush=True)
    out = s.copy(); out[app] = s[app] + c
    out = P["win"](np.clip(out, 0, None))
    ids = dep["MVT_ID_mvt"].to_numpy()
    xg = pl.DataFrame({"MVT_ID_mvt": ids[is_ho].astype(np.int64), "xgb": out[is_ho], "y": y[is_ho], "m": m[is_ho],
                       "nm": nm[is_ho], "apt": dep["apt"].to_numpy()[is_ho]})
    lg = pl.read_parquet(ROOT / "cache" / "eval" / f"{A.NAME}_holdout_ev.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), lgb=pl.col("pred").cast(pl.Float64))
    z = xg.join(lg, on="MVT_ID_mvt", how="inner").filter(pl.col("y") <= O.TRIM)
    z.write_parquet(ROOT / "cache" / "analysis" / f"xgb_scope_{A.NAME}.parquet")
    yy, a, g = z["y"].to_numpy(), z["lgb"].to_numpy(), z["xgb"].to_numpy()
    mm, nmm, ap = z["m"].to_numpy(), z["nm"].to_numpy(), z["apt"].to_numpy()
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
    print(f"corr(lgb - y, xgb - y) {np.corrcoef(a - yy, g - yy)[0, 1]:.4f}; RMS(lgb - xgb) {rm(a - g):.1f}")
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = (1 - w) * a + w * g
        body = (~nmm) & (yy <= 3600)
        print(f"  w_xgb {w:.2f}: trimmed {rm(yy - q):.3f}  Jan {rm((yy - q)[mm == 1]):.3f}  Jul {rm((yy - q)[mm == 7]):.3f}  "
              f"body {rm((yy - q)[body]):.3f}  non-LIRF {rm((yy - q)[ap != 'LIRF']):.3f}")


if __name__ == "__main__":
    main()
