"""2025 -> 2026 drift map on the corrector's inputs (step 1 of the importance-weighting plan, PROGRESS.md §87).

Adversarial LightGBM: Jan+Jul 2025 departures (label 0) vs Jan+Jul 2026 ranking departures (label 1), on the
corrector feature matrix built exactly as src/post/ctx_corrector.py builds it (blocks 1-7 + block 9). Same
months on both sides, so seasonality cannot separate them. Reports out-of-fold AUC, the most drifted
features (gain and single-feature AUC), and writes importance weights w = p / (1 - p) for the 2025
holdout rows (clipped at their 95th percentile, normalised to mean 1) to cache/eval/iw_holdout.parquet.

Run:  .venv/Scripts/python.exe -u tests/drift_map.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ["CTX_RWY"] = "1"
os.environ["CTX_ADSB"] = "1"

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_corrector_test as CC  # noqa: E402
from post import nm_lines as NL  # noqa: E402

EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
PARAMS = dict(objective="binary", num_leaves=63, learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.8,
              seed=42, deterministic=True, force_row_wise=True, verbose=-1, max_bin=127)


def _rd(path, c, col="pred"):
    return pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col(col).cast(pl.Float64).alias(c))


def matrices() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(X for Jan+Jul 2025 holdout rows, X for the 2026 ranking rows, holdout ids/airport/day)."""
    oof = _rd(OOF / "lgb_mixed_nb" / "*.parquet", "lgb").join(_rd(OOF / "cat_mixed" / "*.parquet", "cat"), on="MVT_ID_mvt")
    y_all = NL.frame(str(CC.RAW / "training_*.parquet")).select("MVT_ID_mvt", "y")
    o = oof.join(y_all, on="MVT_ID_mvt")
    w, _ = nnls(o.select("lgb", "cat").to_numpy(), o["y"].to_numpy())
    ho = _rd(EVAL / "lgb_mixed_nb_holdout_ev.parquet", "lgb").join(_rd(EVAL / "cat_mixed_holdout_ev.parquet", "cat"), on="MVT_ID_mvt")
    dep_all, arr_all = CC.movements(str(CC.RAW / "training_*.parquet"))
    dep = ho.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="inner")
    ho = dep.select("MVT_ID_mvt").join(ho, on="MVT_ID_mvt", how="left")
    win = lambda q, lt: np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))
    s = win(ho.select("lgb", "cat").to_numpy() @ w, (dep["T"] - dep["t_LOBT_flt"]).to_numpy())
    cells = CC.cell_table(dep_all)          # all 12 months, as the ranking side uses
    X = CC.build(dep, arr_all.filter(pl.col("L").is_not_null()), s, cells)[0]
    rk_l = pl.read_parquet(SUB / "lgb_mixed_nb.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    rk_c = pl.read_parquet(SUB / "cat_mixed.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), cat=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    dep_r, arr_r = CC.movements(str(RANKING))
    rk = dep_r.select("MVT_ID_mvt").join(rk_l, on="MVT_ID_mvt", how="left").join(rk_c, on="MVT_ID_mvt", how="left")
    s_r = win(rk.select("lgb", "cat").to_numpy() @ w, (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy())
    Xr = CC.build(dep_r, arr_r.filter(pl.col("L").is_not_null()), s_r, cells)[0].select(X.columns)
    meta = dep.select("MVT_ID_mvt", "apt", "m", day=(pl.col("T") // 86400).cast(pl.Int64))
    meta_r = dep_r.select(day=(pl.col("T") // 86400).cast(pl.Int64))
    return X, Xr, meta, meta_r


def main() -> None:
    X, Xr, meta, meta_r = matrices()
    cols = X.columns
    A = np.vstack([X.to_numpy().astype(np.float64), Xr.to_numpy().astype(np.float64)])
    lab = np.concatenate([np.zeros(X.height), np.ones(Xr.height)])
    # folds by UTC day, so flights of the same day never sit on both sides of a split
    grp = np.concatenate([meta["day"].to_numpy(), meta_r["day"].to_numpy()])
    fold = (grp * 2654435761 % 2**32) % 5
    cats = [cols.index(c) for c in ("apt_c", "stand_pfx") if c in cols]
    p = np.zeros(len(lab))
    gain = np.zeros(len(cols))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(A[tr], lab[tr], categorical_feature=cats), 300)
        p[te] = m.predict(A[te])
        gain += m.feature_importance("gain")
    print(f"rows: 2025 Jan+Jul {X.height:,}, 2026 Jan+Jul {Xr.height:,}")
    print(f"adversarial out-of-fold AUC: {roc_auc_score(lab, p):.3f}")
    for mm in (1, 7):
        k = np.concatenate([meta["m"].to_numpy() == mm, np.ones(Xr.height, bool)])
        print(f"  ... 2025 month {mm} rows only vs all 2026: AUC {roc_auc_score(lab[k], p[k]):.3f}")
    top = np.argsort(-gain)[:15]
    print("most drifted features (share of gain | single-feature AUC, 0.5 = no drift):")
    for i in top:
        x = A[:, i]; ok = ~np.isnan(x)
        auc = roc_auc_score(lab[ok], x[ok]) if ok.sum() > 1000 and len(np.unique(lab[ok])) == 2 else np.nan
        print(f"  {cols[i]:18s} {gain[i] / gain.sum():6.1%} | {max(auc, 1 - auc):.3f}")
    ph = np.clip(p[: X.height], 1e-4, 1 - 1e-4)
    w = ph / (1 - ph)
    w = np.minimum(w, np.quantile(w, 0.95))
    w = w / w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum()
    print(f"holdout weights: effective sample size {ess:,.0f} of {len(w):,} ({ess / len(w):.1%})")
    print("mean weight by airport:", {a: round(float(w[meta['apt'].to_numpy() == a].mean()), 2) for a in sorted(set(meta['apt']))})
    meta.select("MVT_ID_mvt").with_columns(w=pl.Series(w)).write_parquet(EVAL / "iw_holdout.parquet")
    print("wrote iw_holdout.parquet")


if __name__ == "__main__":
    main()
