"""v61 (reports/v61_learned_dayshift_preregistration.md): learned day-shift allocation variants S1-S4 vs R0,
2025 all 12 months, 8 airports, slopes leave-period-out. Saves the per-row 2025 lambda (cache/v57/lam_row_2025.parquet)
and the production slopes of the chosen variant (cache/v57/v61_slopes.json).

Run:  python tests/v61_learned_dayshift.py
"""
import os, sys, json
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
from post import pru_recon2 as R  # noqa: E402
from post import pru_dsm as M  # noqa: E402
import v57_dsm as T  # noqa: E402

LAM_CACHE = ROOT / "cache/v57/lam_row_2025.parquet"
VARIANTS = ["S1", "S2", "S3", "S4"]


def band(dh):
    return np.digitize(dh, [0, 900, 3600])                 # 0: < 0, 1: 0-15 min, 2: 15-60, 3: >= 60


def tod(hour):
    return np.digitize(hour, [9, 15])                      # 0: [0,9), 1: [9,15), 2: [15,24)


def cells(variant, apt, b, td, lam):
    a = np.searchsorted(np.array(T.A8), apt)
    if variant == "S1":
        return a
    if variant in ("S2",):
        return a * 100 + b * 10 + (np.abs(lam) > 60).astype(int)
    if variant == "S4":
        return a * 100 + b * 10 + td
    return a * 100 + b * 10                                 # S3


def fit(variant, cell, acell, lam, r, dayk, mask):
    """Slopes per cell (OLS through the origin), fallback to the airport slope for thin cells."""
    def ols(c_arr, m):
        out = {}
        df = pl.DataFrame({"c": c_arr[m], "lr": lam[m] * r[m], "ll": lam[m] ** 2, "d": dayk[m]})
        g = df.group_by("c").agg(num=pl.col("lr").sum(), den=pl.col("ll").sum(), n=pl.len(), nd=pl.col("d").n_unique())
        for row in g.iter_rows(named=True):
            out[row["c"]] = (row["num"] / row["den"] if row["den"] > 0 else 0.0, row["n"], row["nd"])
        return out
    base = {k: v[0] for k, v in ols(acell, mask).items()}
    raw = ols(cell, mask)
    s = {}
    for k, (v, n, nd) in raw.items():
        v = v if (n >= 500 and nd >= 10) else base.get(k // 100 if variant != "S1" else k, 0.0)
        s[k] = float(np.clip(v, 0, 1)) if variant == "S3" else float(v)
    return s, base


def lam_rows(x: pl.DataFrame) -> np.ndarray:
    if LAM_CACHE.exists():
        c = pl.read_parquet(LAM_CACHE)
        if c.height == x.height and (c["MVT_ID_mvt"] == x["MVT_ID_mvt"]).all():
            return c["lam"].to_numpy(), c["q0"].to_numpy()
    p, mo = x["p"].to_numpy().astype(float), x["mo"].to_numpy()
    lam_row, q0 = np.full(x.height, np.nan), p.copy()
    for pk, (m1, m2) in T.PERIODS.items():
        for m in (m1, m2):
            mk = mo == m
            cal = x.filter(pl.col("mo").is_in([m1, m2]) & (pl.col("mo") != m)) if pk == "P01" else x.filter(~pl.col("period").is_in(["P01", pk]))
            xm = x.filter(pl.Series(mk)); idx = np.flatnonzero(mk)
            for d in M.day_table(xm, T.dists_from(cal), set(T.A8)):
                lam_row[idx[d["g"]]] = d["lam"]
            q0[idx] = R.apply(xm, R.solve_days(xm, None, skip=("LIRF", "LSZH")), 0.75)
    pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "lam": lam_row, "q0": q0}).write_parquet(LAM_CACHE)
    return lam_row, q0


def main():
    x = T.frame25()
    lam, q0 = lam_rows(x)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    b, td = band(x["ts"].to_numpy() - p), tod(x["hour"].to_numpy())
    dayk = (x["apt"] + "_" + x["day"].cast(pl.Utf8)).to_numpy()
    r = p - y
    in8 = np.isin(apt, T.A8)
    ok = in8 & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    acell = np.where(in8, np.searchsorted(np.array(T.A8), apt), -1)
    body, trim = ~nmu & (y <= 3600), y <= 18000
    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    months = sorted(set(mo))
    rows, qs = [], {}
    for v in VARIANTS:
        cell = np.where(in8, cells(v, apt, b, td, lam0), -1)
        q = p.copy()
        for pk in T.PERIODS:
            s, _ = fit(v, cell, acell, lam0, r, dayk, ok & trim & (per != pk))
            k = ok & (per == pk)
            sl = np.array([s.get(c, 0.0) for c in cell[k]])
            q[k] = np.maximum(p[k] - sl * lam0[k], 0)
        qs[v] = q
        res = {"variant": v, "full": rm(q, in8) - rm(q0, in8), "body": rm(q, in8 & body) - rm(q0, in8 & body),
               "trim": rm(q, in8 & trim) - rm(q0, in8 & trim),
               "months_full": int(sum(rm(q, in8 & (mo == m)) < rm(q0, in8 & (mo == m)) for m in months))}
        res["pass"] = res["full"] <= -0.10 and res["body"] <= 0 and res["trim"] <= 0 and res["months_full"] >= 8
        for m in months:
            res[f"f{m[5:]}"] = round(rm(q, in8 & (mo == m)) - rm(q0, in8 & (mo == m)), 2)
        for m in months:
            res[f"b{m[5:]}"] = round(rm(q, in8 & body & (mo == m)) - rm(q0, in8 & body & (mo == m)), 2)
        rows.append(res)
    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=10, tbl_cols=40, float_precision=3, tbl_width_chars=300):
        print("vs R0 (2025, 12 months, 8 airports): full / body / trimmed, months full better; fMM full, bMM body by month")
        print(res)
    passing = res.filter(pl.col("pass")).sort("full")
    chosen = passing["variant"][0] if passing.height else None
    print(f"passing: {passing['variant'].to_list()}; chosen: {chosen}")
    if chosen is None:
        return
    q = qs[chosen]
    print("  per airport full vs R0:", {a: round(rm(q, apt == a) - rm(q0, apt == a), 2) for a in T.A8})
    print("  per airport body vs R0:", {a: round(rm(q, (apt == a) & body) - rm(q0, (apt == a) & body), 2) for a in T.A8})
    # 2026 projection: per (airport) dMSE per row vs R0 x 2026 rows
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    dmse = sum(n26[a] * np.mean((y[apt == a] - q[apt == a]) ** 2 - (y[apt == a] - q0[apt == a]) ** 2) for a in T.A8) / sum(n26.values())
    print(f"  2026 projection vs R0: board {np.sqrt(239.40 ** 2 + dmse) - 239.40:+.3f} s")
    # production slopes from all six periods
    cell = np.where(in8, cells(chosen, apt, b, td, lam0), -1)
    s, base = fit(chosen, cell, acell, lam0, r, dayk, ok & trim)
    json.dump({"variant": chosen, "slopes": {str(k): v for k, v in s.items()}, "airport": {str(k): v for k, v in base.items()},
               "A8": T.A8}, open(ROOT / "cache/v57/v61_slopes.json", "w"), indent=1)
    print("  production slopes:", {T.A8[int(k) // 100] + f"|b{(int(k) // 10) % 10}|{int(k) % 10}" if chosen != "S1" else T.A8[int(k)]: round(v, 2)
                                   for k, v in sorted(s.items())})


if __name__ == "__main__":
    main()
