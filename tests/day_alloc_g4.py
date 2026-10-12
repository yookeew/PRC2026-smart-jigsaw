"""G4: G2's day-shift slope model plus day-type inputs (reports/day_alloc_g4_preregistration.md).

Gate vs G2x8 (v63's recipe), 2025, 12 months, 8 airports, leave-period-out as tests/day_alloc_gbm.py.

Run:  .venv/Scripts/python.exe tests/day_alloc_g4.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
import v57_dsm as T  # noqa: E402
import v61_learned_dayshift as L  # noqa: E402
import day_alloc_gbm as G  # noqa: E402

EXT = ROOT / "external-data"


def day_type(year: int) -> pl.DataFrame:
    rd = lambda f, cols: pl.read_csv(EXT / f, columns=["FLT_DATE", "APT_ICAO", *cols], infer_schema_length=0)  # noqa: E731
    a = rd(f"all_pre_departure_delays_{year}.csv", ["DLY_ALL_PRE_2"])
    c = rd(f"atc_pre_departure_delays_{year}.csv", ["DLY_ATC_PRE_2"])
    s = rd(f"atfm_slot_adherence_{year}.csv", ["FLT_DEP_1", "FLT_DEP_REG_1"])
    k = ["FLT_DATE", "APT_ICAO"]
    d = a.join(c, on=k, how="left").join(s, on=k, how="left").with_columns(
        [pl.col(x).cast(pl.Float64, strict=False) for x in ("DLY_ALL_PRE_2", "DLY_ATC_PRE_2", "FLT_DEP_1", "FLT_DEP_REG_1")])
    return d.select(apt="APT_ICAO", day=pl.col("FLT_DATE").str.slice(0, 10).str.to_date(),
                    atc_share=(pl.col("DLY_ATC_PRE_2").fill_null(0) / pl.col("DLY_ALL_PRE_2")).clip(0, 1),
                    reg_share=pl.col("FLT_DEP_REG_1") / pl.col("FLT_DEP_1")).unique(["apt", "day"])


def add_day_type(x: pl.DataFrame, year: int) -> pl.DataFrame:
    out = x.join(day_type(year), on=["apt", "day"], how="left", maintain_order="left")
    assert out.height == x.height
    return out


def inputs4(x, lam0):
    extra = np.column_stack([x[c].to_numpy().astype(np.float64) for c in ("atc_share", "reg_share")])
    return np.column_stack([G.inputs(x, lam0, True), extra])


def production_model():
    x = add_day_type(T.frame25(), 2025)
    lam, _ = L.lam_rows(x)
    p, y, apt = x["p"].to_numpy().astype(float), x["y"].to_numpy(), x["apt"].to_numpy()
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    m = np.isin(apt, T.A8) & np.isfinite(lam) & (y <= 18000) & (np.abs(lam0) >= 1)
    return G.slope_fit(inputs4(x, lam0), p - y, lam0, m)


def main() -> None:
    x = add_day_type(T.frame25(), 2025)
    in8 = np.isin(x["apt"].to_numpy(), T.A8)
    print("day-type inputs on the 8 airports' rows: null share",
          {c: round(float(x.filter(pl.Series(in8))[c].is_null().mean()), 4) for c in ("atc_share", "reg_share")},
          "| mean", {c: round(float(x.filter(pl.Series(in8))[c].mean()), 3) for c in ("atc_share", "reg_share")}, flush=True)
    lam, q0 = L.lam_rows(x)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    ok = in8 & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    body, trim = ~nmu & (y <= 3600), y <= 18000
    r = p - y
    X2, X4 = G.inputs(x, lam0, True), inputs4(x, lam0)
    qs = {"G2x8 (v63)": p.copy(), "G4x8": p.copy()}
    for pk in T.PERIODS:
        mfit = ok & trim & (per != pk) & (np.abs(lam0) >= 1)
        k = ok & (per == pk)
        for name, X in (("G2x8 (v63)", X2), ("G4x8", X4)):
            bst = G.slope_fit(X, r, lam0, mfit)
            qs[name][k] = np.maximum(p[k] - np.clip(bst.predict(X[k], raw_score=True) + G.INIT, 0, 1) * lam0[k], 0)
        print("period", pk, "done", flush=True)
    qb = qs["G2x8 (v63)"]
    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    months = sorted(set(mo))
    nf = mo != "2025-02"
    rows = []
    for name, q in list(qs.items()) + [("R0", q0)]:
        res = {"cand": name, "full": rm(q, in8) - rm(qb, in8), "body": rm(q, in8 & body) - rm(qb, in8 & body),
               "trim": rm(q, in8 & trim) - rm(qb, in8 & trim), "full_noFeb": rm(q, in8 & nf) - rm(qb, in8 & nf),
               "mo_better": int(sum(rm(q, in8 & (mo == m)) < rm(qb, in8 & (mo == m)) for m in months))}
        for m in months:
            res[f"f{m[5:]}"] = round(rm(q, in8 & (mo == m)) - rm(qb, in8 & (mo == m)), 2)
        res["pass"] = (res["full"] <= -0.10 and res["body"] <= 0 and res["trim"] <= 0 and res["mo_better"] >= 8
                       and res["f02"] <= 0 and res["f06"] <= 0 and res["full_noFeb"] <= -0.05)
        rows.append(res)
    with pl.Config(tbl_rows=10, tbl_cols=40, float_precision=3, tbl_width_chars=300):
        print("vs G2x8 (v63's recipe), 2025, 12 months, 8 airports' rows; fMM = full by month")
        print(pl.DataFrame(rows))
    q = qs["G4x8"]
    print("G4x8 per airport full:", {a: round(rm(q, apt == a) - rm(qb, apt == a), 2) for a in T.A8})
    print("G4x8 per airport body:", {a: round(rm(q, (apt == a) & body) - rm(qb, (apt == a) & body), 2) for a in T.A8})
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    dmse = sum(n26[a] * np.mean((y[apt == a] - q[apt == a]) ** 2 - (y[apt == a] - qb[apt == a]) ** 2) for a in T.A8) / sum(n26.values())
    print(f"G4x8 2026 projection vs v63: {np.sqrt(238.3173 ** 2 + dmse) - 238.3173:+.3f} s")


if __name__ == "__main__":
    main()
