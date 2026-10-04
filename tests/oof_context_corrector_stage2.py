"""Stage 2 of reports/oof_context_corrector_preregistration.md: the corrected stack through v30's chain.

1. Corrected-stack predictions in the chain's input format:
     holdout  : corrector fit on all 10 OOF months   -> cache/eval/ctxcorr_holdout_ev.parquet
     OOF      : two-month-block cross-fit             -> cache/oof/ctxcorr/fold=<block>.parquet
2. v30's chain cross-fit with them as the LightGBM input vs the current v31 chain; LOBT window and
   NM-unmatched lines on both; decision on body RMSE.

Run:  .venv/Scripts/python.exe tests/oof_context_corrector_stage2.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_corrector_test as CC  # noqa: E402
import oof_context_corrector_test as O  # noqa: E402
from post import nm_lines as NL  # noqa: E402

ENS = os.environ.get("CTX_ENS") == "2"   # reports/ctx_ensemble_preregistration.md: global + per-airport mean
NAME = "ctxcorr2" if ENS else "ctxcorr"
EV = ROOT / "cache" / "eval" / f"{NAME}_holdout_ev.parquet"
OOFD = ROOT / "cache" / "oof" / NAME
BLOCKS = [(2, 3), (4, 5), (6, 8), (9, 10), (11, 12)]


def build_inputs() -> None:
    P = O.prepare()
    X = P["X_full"].to_numpy().astype(np.float64)
    cols = P["X_full"].columns
    cats = [cols.index(c) for c in P["catn"]]
    s, y, m, nm, lr, is_ho, inner = (P[k] for k in ("s", "y", "m", "nm", "line_rows", "is_ho", "inner"))
    trainable = (~is_ho) & (~nm) & (y <= O.TRIM) & ~lr
    out = s.copy()
    app = is_ho & ~lr
    fa = (lambda *a: CC.fit_apply_ens(*a, P["apt"])) if ENS else CC.fit_apply
    c, best = fa(X, y - s, trainable, inner, app, cats)
    out[app] = s[app] + c
    print(f"holdout corrector: best_iter {best}", flush=True)
    for blk in BLOCKS:
        inb = np.isin(m, blk) & ~is_ho
        c, best = fa(X, y - s, trainable & ~inb, inner, inb & ~lr, cats)
        out[inb & ~lr] = s[inb & ~lr] + c
        print(f"OOF block {blk}: best_iter {best}", flush=True)
    out = P["win"](np.clip(out, 0, None))
    ids = P["dep"]["MVT_ID_mvt"]
    pl.DataFrame({"MVT_ID_mvt": ids.filter(pl.Series(is_ho)), "pred": out[is_ho]}).write_parquet(EV)
    OOFD.mkdir(parents=True, exist_ok=True)
    for f in OOFD.glob("*.parquet"):
        f.unlink()
    for blk in BLOCKS:
        k = np.isin(m, blk) & ~is_ho
        pl.DataFrame({"MVT_ID_mvt": ids.filter(pl.Series(k)), "pred": out[k]}).write_parquet(OOFD / f"fold={blk[0]:02d}-{blk[1]:02d}.parquet")


def main() -> None:
    if not EV.exists():
        build_inputs()
    import base_neighbour_chain_test as B
    import neighbour_state_test as N
    B.NEW_EV, B.NEW_OOF = EV, OOFD
    df = B.A.load()
    ex, lag = N.train_extra()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    base = B.v30_method(df, ex, lag, arr)
    # with_new_lgb asserts 10 monthly folds; ours are 5 two-month blocks
    folds = sorted(OOFD.glob("fold=*.parquet")); assert len(folds) == len(BLOCKS)
    ev = pl.read_parquet(EV).select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    df2 = df.join(ev, on="MVT_ID_mvt", how="left"); assert df2["lgb_new"].null_count() == 0
    df2 = df2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    oof = pl.concat([pl.read_parquet(p) for p in folds]).select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    ex2 = ex.join(oof, on="MVT_ID_mvt", how="left"); assert ex2["lgb_new"].null_count() == 0
    ex2 = ex2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    from scipy.optimize import nnls
    w, _ = nnls(ex2.select("lgb", "cat").to_numpy(), ex2["taxi"].to_numpy())
    print("treatment OOF stack weights", np.round(w, 3))
    ex2 = ex2.with_columns(s=pl.Series(ex2.select("lgb", "cat").to_numpy() @ w))
    new = B.v30_method(df2, ex2, lag, arr)
    # LOBT window + NM-unmatched lines (10-month fit) on both, as v33's method
    raw = NL.frame(str(ROOT / "data" / "raw" / "training_*.parquet"))
    prm = NL.fit(raw.filter(~pl.col("m").is_in([1, 7])))
    h = df.select("MVT_ID_mvt").join(raw, on="MVT_ID_mvt", how="left")
    lt = h["lt"].to_numpy()
    fin = lambda q: NL.apply(h, np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606)), prm)[0]
    b, t = fin(base), fin(new)
    y = df["taxi"].to_numpy(); ym = df["ym"].to_numpy(); apt = df["ADEP_mvt"].to_numpy()
    body = (~h["nm"].to_numpy()) & (y <= 3600)
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))
    print(f"\n{'slice':20s} {'v33 method':>11s} {'+ctx corrector':>15s} {'delta':>7s}")
    rows = [("body pooled", body), ("body Jan", body & (ym == "2025-01")), ("body Jul", body & (ym == "2025-07")),
            ("trimmed", y <= 18000), ("full", np.ones_like(body))] + [(f"  body {a}", body & (apt == a)) for a in sorted(set(apt))]
    for name, k in rows:
        print(f"{name:20s} {r(b, k):11.2f} {r(t, k):15.2f} {r(t, k) - r(b, k):+7.2f}")
    g = {n: r(b, k) - r(t, k) for n, k in (("pooled", body), ("Jan", body & (ym == "2025-01")), ("Jul", body & (ym == "2025-07")))}
    ok = g["pooled"] >= 1.5 and g["Jan"] > 0 and g["Jul"] > 0 and r(t, y <= 18000) <= r(b, y <= 18000)
    print(f"\nSTAGE 2: body gain {g} -> {'ADOPT' if ok else 'REJECT'}")
    pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], "base": b, "new": t}).write_parquet(ROOT / "cache" / "eval" / "ctxcorr_chain_holdout.parquet")


if __name__ == "__main__":
    main()
