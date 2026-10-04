"""Context corrector on the base stack, production fit (reports/oof_context_corrector_preregistration.md).

Stack s = NNLS(lgb_mixed_nb, cat_mixed) (weights from the 10 OOF months) + LOBT window. The
corrector (LightGBM L2 on y - s, the six context blocks of tests/context_corrector_test.py) is
fitted on all 12 months of out-of-sample stack predictions: the 10 OOF months (fold files) and
Jan+Jul 2025 (holdout predictions of the 10-month fits). Cell statistics are leave-one-month-out
on training rows and all 12 months for the ranking. Rounds from the inner day % 5 split.

Writes data/submissions/ctxcorr.parquet (ranking, the corrected stack) and fills in the chain
columns of cache/eval/ctxcorr_holdout_ev.parquet, so that PRC_LGB=ctxcorr runs v30's chain on it.

CTX_ENS=2 switches to the ensemble of reports/ctx_ensemble_preregistration.md (mean of the global
LightGBM and one LightGBM per airport) and writes ctxcorr2.parquet / ctxcorr2_holdout_ev.parquet.

Run:  .venv/Scripts/python.exe src/post/ctx_corrector.py
      CTX_ENS=2 .venv/Scripts/python.exe src/post/ctx_corrector.py
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import context_corrector_test as CC  # noqa: E402
from post import nm_lines as NL  # noqa: E402

EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
ENS = os.environ.get("CTX_ENS") == "2"
NAME = "ctxcorr2" if ENS else "ctxcorr"
OUT = SUB / f"{NAME}.parquet"
EV = EVAL / f"{NAME}_holdout_ev.parquet"
TRIM = 18000


def _rd(path, c, col="pred"):
    return pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col(col).cast(pl.Float64).alias(c))


def main() -> None:
    # ---- training rows: 12 months of out-of-sample stack predictions
    oof = _rd(OOF / "lgb_mixed_nb" / "*.parquet", "lgb").join(_rd(OOF / "cat_mixed" / "*.parquet", "cat"), on="MVT_ID_mvt")
    ho = _rd(EVAL / "lgb_mixed_nb_holdout_ev.parquet", "lgb").join(_rd(EVAL / "cat_mixed_holdout_ev.parquet", "cat"), on="MVT_ID_mvt")
    w, _ = nnls(oof.join(NL.frame(str(CC.RAW / "training_*.parquet")).select("MVT_ID_mvt", "y"), on="MVT_ID_mvt")
                .select("lgb", "cat").to_numpy(),
                oof.join(NL.frame(str(CC.RAW / "training_*.parquet")).select("MVT_ID_mvt", "y"), on="MVT_ID_mvt")["y"].to_numpy())
    print("stack weights (OOF months):", np.round(w, 3))
    tr = pl.concat([oof, ho])
    dep_all, arr_all = CC.movements(str(CC.RAW / "training_*.parquet"))
    dep = tr.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="inner")
    tr = dep.select("MVT_ID_mvt").join(tr, on="MVT_ID_mvt", how="left")
    lt = (dep["T"] - dep["t_LOBT_flt"]).to_numpy()
    win = lambda q, lt=lt: np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))
    s = win(tr.select("lgb", "cat").to_numpy() @ w)
    arr = arr_all.filter(pl.col("L").is_not_null())
    X, ctrl, catn = CC.build(dep, arr, s, CC.cell_table(dep_all))
    m = dep["m"].to_numpy()
    cel_cols = ["cel_p50", "cel_p90", "cel_sd", "cel_n"]
    cel = X.select(cel_cols).to_numpy().copy()
    for mm in sorted(set(m)):
        k = m == mm
        cel[k] = CC.apply_cells(dep.filter(pl.Series(k)), CC.cell_table(dep_all.filter(pl.col("m") != mm))).to_numpy()
    X = X.with_columns([pl.Series(c, cel[:, i]) for i, c in enumerate(cel_cols)]).with_columns(cel_rel=pl.Series(s) - pl.col("cel_p50"))
    cols = X.columns
    y = dep["y"].to_numpy(); nm = dep["nm"].to_numpy()
    d = (dep["T"] - dep["t_SCHED_TIME_UTC_mvt"]).to_numpy()
    line_rows = nm & (np.nan_to_num(d, nan=-np.inf) > NL.LINE_MIN_D)
    fit = (~nm) & (y <= TRIM) & ~line_rows
    inner = (dep["T"].to_numpy() // 86400).astype(int) % 5 == 0
    print(f"training rows {fit.sum():,} of {len(y):,} (12 months); features {len(cols)}")

    # ---- ranking rows
    rk_l = pl.read_parquet(SUB / "lgb_mixed_nb.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    rk_c = pl.read_parquet(SUB / "cat_mixed.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), cat=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    dep_r, arr_r = CC.movements(str(RANKING))
    rk = dep_r.select("MVT_ID_mvt").join(rk_l, on="MVT_ID_mvt", how="left").join(rk_c, on="MVT_ID_mvt", how="left")
    assert rk["lgb"].null_count() == 0 and rk["cat"].null_count() == 0
    lt_r = (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy()
    s_r = win(rk.select("lgb", "cat").to_numpy() @ w, lt_r)
    Xr, _, _ = CC.build(dep_r, arr_r.filter(pl.col("L").is_not_null()), s_r, CC.cell_table(dep_all))
    Xr = Xr.with_columns(cel_rel=pl.Series(s_r) - pl.col("cel_p50")).select(cols)
    nm_r = dep_r["nm"].to_numpy(); d_r = (dep_r["T"] - dep_r["t_SCHED_TIME_UTC_mvt"]).to_numpy()
    app = ~(nm_r & (np.nan_to_num(d_r, nan=-np.inf) > NL.LINE_MIN_D))

    Xa = X.to_numpy().astype(np.float64)
    cats = [cols.index(c) for c in catn]
    import lightgbm as lgb
    if ENS:
        Xb = np.vstack([Xa, Xr.to_numpy().astype(np.float64)])
        n_tr, n_r = len(y), Xr.height
        z = np.zeros(n_r, bool)
        apt = np.concatenate([dep["apt"].to_numpy(), dep_r["apt"].to_numpy()])
        corr, best = CC.fit_apply_ens(Xb, np.concatenate([y - s, np.zeros(n_r)]), np.concatenate([fit, z]),
                                      np.concatenate([inner, z]), np.concatenate([np.zeros(n_tr, bool), app]), cats, apt)
        print(f"corrector (ensemble2): global best_iter {best}")
        out = s_r.copy()
        out[app] = s_r[app] + corr
    else:
        tr_ds = lgb.Dataset(Xa[fit & ~inner], (y - s)[fit & ~inner], categorical_feature=cats, free_raw_data=False)
        va_ds = lgb.Dataset(Xa[fit & inner], (y - s)[fit & inner], categorical_feature=cats, reference=tr_ds, free_raw_data=False)
        mdl = lgb.train(CC.PARAMS, tr_ds, 3000, valid_sets=[va_ds], callbacks=[lgb.early_stopping(100, verbose=False)])
        best = max(mdl.best_iteration, 1)
        full = lgb.train(CC.PARAMS, lgb.Dataset(Xa[fit], (y - s)[fit], categorical_feature=cats), best)
        print(f"corrector: best_iter {best}")
        out = s_r.copy()
        out[app] = s_r[app] + full.predict(Xr.to_numpy().astype(np.float64)[app])
    out = win(np.clip(out, 0, None), lt_r)
    print(f"ranking: RMS change vs stack {np.sqrt(np.mean((out - s_r) ** 2)):.1f} s; mean {np.mean(out - s_r):+.1f} s")
    sub = pl.DataFrame({"MVT_ID_mvt": dep_r["MVT_ID_mvt"], "TAXITIME_SEC_mvt": np.round(out).astype(np.int32)})
    tpl = pl.read_parquet(TEMPLATE)
    sub = tpl.select("MVT_ID_mvt").join(sub.with_columns(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
    assert sub.height == tpl.height and sub["TAXITIME_SEC_mvt"].null_count() == 0
    sub.write_parquet(OUT)
    print(f"wrote {OUT.name}")

    # ---- holdout ev in the chain's format (pred = the stage-2 corrected stack, fit on the 10 OOF months)
    base = pl.read_parquet(EVAL / "lgb_mixed_nb_holdout_ev.parquet").drop("pred")
    ev = pl.read_parquet(EV).select(pl.col("MVT_ID_mvt").cast(pl.Float64), "pred")
    base.join(ev, on="MVT_ID_mvt", how="inner").write_parquet(EV)
    print(f"rewrote {EV.name} with the chain columns")


if __name__ == "__main__":
    main()
