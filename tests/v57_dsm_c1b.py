"""v57 C1b (reports/v57_daysum_model_preregistration.md, amendment A1): day shift allocated by learned slopes
s_{airport, delay band} on the expectation-solve lam, leave-period-out over the six 2025 periods, vs R0.

Run:  python tests/v57_dsm_c1b.py
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

BANDS = [-np.inf, 0, 900, 3600, np.inf]


def band(dh):
    return np.digitize(dh, BANDS[1:-1])        # 0: < 0, 1: 0-15 min, 2: 15-60, 3: >= 60


def fit_slopes(lam_row, r, apt, b, mask):
    s = {}
    for a in T.A8:
        for k in range(4):
            m = mask & (apt == a) & (b == k) & np.isfinite(lam_row)
            den = float(np.sum(lam_row[m] ** 2))
            s[(a, k)] = float(np.sum(lam_row[m] * r[m]) / den) if den > 0 else 0.0
    return s


def main():
    x = T.frame25()
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    b = band(x["ts"].to_numpy() - p)
    lam_row = np.full(x.height, np.nan)
    q0 = p.copy()
    for pk, (m1, m2) in T.PERIODS.items():
        for m in (m1, m2):
            mk = (mo == m)
            cal = x.filter(pl.col("mo").is_in([m1, m2]) & (pl.col("mo") != m)) if pk == "P01" else x.filter(~pl.col("period").is_in(["P01", pk]))
            xm = x.filter(pl.Series(mk)); idx = np.flatnonzero(mk)
            for d in M.day_table(xm, T.dists_from(cal), set(T.A8)):
                lam_row[idx[d["g"]]] = d["lam"]
            q0[idx] = R.apply(xm, R.solve_days(xm, None, skip=("LIRF", "LSZH")), 0.75)
    r = p - y
    q = p.copy()
    fitmask = (y <= 18000)
    slopes_by_p = {}
    for pk in T.PERIODS:
        s = fit_slopes(lam_row, r, apt, b, fitmask & (per != pk))
        slopes_by_p[pk] = s
        k = (per == pk) & np.isfinite(lam_row) & np.isin(apt, T.A8)
        sl = np.array([s[(a, bb)] for a, bb in zip(apt[k], b[k])])
        q[k] = np.maximum(p[k] - sl * lam_row[k], 0)
    in8, body = np.isin(apt, T.A8), ~nmu & (y <= 3600)
    rm = lambda qq, m: float(np.sqrt(np.mean((y[m] - qq[m]) ** 2)))  # noqa: E731
    months = sorted(set(mo))
    res = {"full_vs_base": rm(q, in8) - rm(p, in8), "R0_full_vs_base": rm(q0, in8) - rm(p, in8),
           "full_vs_R0": rm(q, in8) - rm(q0, in8), "body_vs_R0": rm(q, in8 & body) - rm(q0, in8 & body),
           "trim_vs_R0": rm(q, in8 & (y <= 18000)) - rm(q0, in8 & (y <= 18000)),
           "months_full_beat_R0": int(sum(rm(q, in8 & (mo == m)) < rm(q0, in8 & (mo == m)) for m in months))}
    print("C1b vs R0 (2025, 12 months, 8 airports):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    print("  full vs R0 by month:", {m[5:]: round(rm(q, in8 & (mo == m)) - rm(q0, in8 & (mo == m)), 2) for m in months})
    print("  body vs R0 by month:", {m[5:]: round(rm(q, in8 & body & (mo == m)) - rm(q0, in8 & body & (mo == m)), 2) for m in months})
    print("  full vs R0 by airport:", {a: round(rm(q, apt == a) - rm(q0, apt == a), 2) for a in T.A8})
    gate = [res["full_vs_R0"] <= -0.10, res["body_vs_R0"] <= 0, res["months_full_beat_R0"] >= 8]
    print(f"  C1b GATE (full<=-0.10, body<=0, >=8/12 months): {gate} -> {'PASS' if all(gate) else 'FAIL'}")
    s_all = fit_slopes(lam_row, r, apt, b, fitmask)
    print("  production slopes (all periods), bands <0 / 0-15 / 15-60 / >=60 min:",
          {a: [round(s_all[(a, k)], 2) for k in range(4)] for a in T.A8})
    json.dump({f"{a}|{k}": v for (a, k), v in s_all.items()}, open(ROOT / "cache/v57/c1b_slopes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
