"""G3: G2's day-shift slope model plus per-flight inputs (reports/day_alloc_g3_preregistration.md).

Gate vs G2x8 (v63's recipe), 2025, 12 months, 8 airports, leave-period-out as tests/day_alloc_gbm.py.

Run:  .venv/Scripts/python.exe tests/day_alloc_g3.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402
import day_alloc_gbm as G  # noqa: E402

CAT = {"WK_TBL_CAT_flt": ["H", "J", "L", "M"],
       "MARKET_SEGMENT_flt": ["All-Cargo", "Business Aviation", "Charter", "Low-Cost", "Mainline", "Military", "Not classified",
                              "Other", "Regional"]}
NUM = ["n_dep_30m_prev", "dep_pressure", "inbound_arr_delay", "eobt_delay", "sched_ground", "aobt3_taxi",
       "deicing_risk", "wx_snow", "below_freezing"]


def add_feats(x: pl.DataFrame, path: Path) -> pl.DataFrame:
    """Join the extra inputs on MVT_ID_mvt (keeps x's row order and dtype)."""
    f = pl.read_parquet(path, columns=["MVT_ID_mvt", *CAT, *NUM]).with_columns(
        pl.col("MVT_ID_mvt").cast(x["MVT_ID_mvt"].dtype),
        *[pl.col(c).replace_strict(v, list(range(len(v))), default=-1).cast(pl.Float64).alias(c) for c, v in CAT.items()],
        *[pl.col(c).cast(pl.Float64) for c in NUM])
    out = x.join(f, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert out.height == x.height
    return out


def inputs3(x, lam0):
    extra = np.column_stack([x[c].to_numpy().astype(np.float64) for c in [*CAT, *NUM]])
    return np.column_stack([G.inputs(x, lam0, True), extra])


def slope_fit3(X, r, lam, m):
    ds = lgb.Dataset(X[m], label=r[m] / lam[m], weight=lam[m] ** 2, init_score=np.full(m.sum(), G.INIT),
                     categorical_feature=[0, 9, 10], free_raw_data=False)
    return lgb.train(G.PARAMS, ds, G.ROUNDS)


def production_model():
    x = add_feats(T.frame25(), ROOT / "cache/features/train2025.parquet")
    lam, _ = L.lam_rows(x)
    p, y, apt = x["p"].to_numpy().astype(float), x["y"].to_numpy(), x["apt"].to_numpy()
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    m = np.isin(apt, T.A8) & np.isfinite(lam) & (y <= 18000) & (np.abs(lam0) >= 1)
    return slope_fit3(inputs3(x, lam0), p - y, lam0, m)


def main() -> None:
    x = T.frame25()
    x = add_feats(x, ROOT / "cache/features/train2025.parquet")
    print("extra inputs, null share:", {c: round(x[c].null_count() / x.height, 3) for c in [*CAT, *NUM]}, flush=True)
    lam, q0 = L.lam_rows(x)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    in8 = np.isin(apt, T.A8)
    ok = in8 & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    body, trim = ~nmu & (y <= 3600), y <= 18000
    r = p - y
    X2, X3 = G.inputs(x, lam0, True), inputs3(x, lam0)
    qs = {"G2x8 (v63)": p.copy(), "G3x8": p.copy()}
    for pk in T.PERIODS:
        mfit = ok & trim & (per != pk) & (np.abs(lam0) >= 1)
        k = ok & (per == pk)
        for name, X, fit in (("G2x8 (v63)", X2, G.slope_fit), ("G3x8", X3, slope_fit3)):
            bst = fit(X, r, lam0, mfit)
            qs[name][k] = np.maximum(p[k] - np.clip(bst.predict(X[k], raw_score=True) + G.INIT, 0, 1) * lam0[k], 0)
        print("period", pk, "done", flush=True)
    qb = qs["G2x8 (v63)"]
    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    months = sorted(set(mo))
    rows = []
    for name, q in list(qs.items()) + [("R0", q0)]:
        res = {"cand": name, "full": rm(q, in8) - rm(qb, in8), "body": rm(q, in8 & body) - rm(qb, in8 & body),
               "trim": rm(q, in8 & trim) - rm(qb, in8 & trim),
               "mo_better": int(sum(rm(q, in8 & (mo == m)) < rm(qb, in8 & (mo == m)) for m in months))}
        for m in months:
            res[f"f{m[5:]}"] = round(rm(q, in8 & (mo == m)) - rm(qb, in8 & (mo == m)), 2)
        res["pass"] = (res["full"] <= -0.10 and res["body"] <= 0 and res["trim"] <= 0 and res["mo_better"] >= 8
                       and res["f02"] <= 0 and res["f06"] <= 0)
        rows.append(res)
    with pl.Config(tbl_rows=10, tbl_cols=40, float_precision=3, tbl_width_chars=300):
        print("vs G2x8 (v63's recipe), 2025, 12 months, 8 airports' rows; fMM = full by month")
        print(pl.DataFrame(rows))
    q = qs["G3x8"]
    print("G3x8 per airport full:", {a: round(rm(q, apt == a) - rm(qb, apt == a), 2) for a in T.A8})
    print("G3x8 per airport body:", {a: round(rm(q, (apt == a) & body) - rm(qb, (apt == a) & body), 2) for a in T.A8})
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    dmse = sum(n26[a] * np.mean((y[apt == a] - q[apt == a]) ** 2 - (y[apt == a] - qb[apt == a]) ** 2) for a in T.A8) / sum(n26.values())
    print(f"G3x8 2026 projection vs v63: {np.sqrt(238.3173 ** 2 + dmse) - 238.3173:+.3f} s")


if __name__ == "__main__":
    main()
