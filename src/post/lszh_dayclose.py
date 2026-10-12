"""LSZH level from the daily sums (reports/lszh_dayclose_preregistration.md).

LSZH rows only: per airport-day, the production engine's point solve (pru_recon2.solve_days, R0) gives lam_d with
sum max(0, T - S - p + lam_d) = 60 D; shift s_d = lam_m + a (lam_d - lam_m), lam_m = median lam_d of the month;
new taxi = max(p - s_d, 0), then LOBT window, clip, round (final4m_v62.finish). Every other row keeps the input value.

Run:  .venv/Scripts/python.exe src/post/lszh_dayclose.py --check2025              (picks a by the registered rule)
      .venv/Scripts/python.exe src/post/lszh_dayclose.py IN.parquet OUT.parquet A
      PRC_SCOPE=final4m ... (four-month template / ranking, src/scope.py)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
import scope  # noqa: E402
from post import pru_recon2 as R  # noqa: E402
import final4m_v62 as F  # noqa: E402

APT = "LSZH"


def day_lams(x: pl.DataFrame) -> list:
    """Solved LSZH days of x (rows with p, ts, apt, day, D, nD, mo): dicts with g (row indices), lam, per (month)."""
    skip = tuple(a for a in set(x["apt"].to_list()) if a != APT)
    return R.solve_days(x, None, skip=skip, period_col="mo")


def shifted(x: pl.DataFrame, days: list, a: float) -> np.ndarray:
    p = x["p"].to_numpy().astype(float)
    q = p.copy()
    lam_m = {m: float(np.median([d["lam"] for d in days if d["per"] == m])) for m in {d["per"] for d in days}}
    for d in days:
        s = lam_m[d["per"]] + a * (d["lam"] - lam_m[d["per"]])
        q[d["g"]] = np.maximum(p[d["g"]] - s, 0.0)
    return q


def month_medians(x: pl.DataFrame) -> dict:
    days = day_lams(x)
    return {m: float(np.median([d["lam"] for d in days if d["per"] == m])) for m in sorted({d["per"] for d in days})}


def check2025() -> None:
    import v57_dsm as T
    x = T.frame25().filter(pl.col("apt") == APT)
    y, p, mo = x["y"].to_numpy(), x["p"].to_numpy().astype(float), x["mo"].to_numpy()
    nmu = x["nmu"].to_numpy()
    days = day_lams(x)
    rm = lambda q, m: float(np.sqrt(np.mean((y[m] - q[m]) ** 2)))  # noqa: E731
    allm, trim, body = np.ones(len(y), bool), y <= 18000, ~nmu & (y <= 3600)
    months = sorted(set(mo))
    print(f"2025 LSZH rows {len(y):,}; solved days {len(days)}; no stage: full {rm(p, allm):.3f}, trimmed {rm(p, trim):.3f}, "
          f"body {rm(p, body):.3f}")
    res = {}
    for a in (0.5, 0.75, 1.0):
        q = shifted(x, days, a)
        better = sum(rm(q, mo == m) < rm(p, mo == m) for m in months)
        res[a] = (rm(q, allm), better)
        print(f"  a {a:4.2f}: full {rm(q, allm) - rm(p, allm):+.3f}  trimmed {rm(q, trim) - rm(p, trim):+.3f}  "
              f"body {rm(q, body) - rm(p, body):+.3f}  months better {better}/12  "
              + " ".join(f"{m[5:]}:{rm(q, mo == m) - rm(p, mo == m):+.2f}" for m in months))
    if res[1.0][0] <= min(res[0.5][0], res[0.75][0]) and res[1.0][1] >= 9:
        a = 1.0
    else:
        a = 0.5 if res[0.5][0] <= res[0.75][0] else 0.75
    print(f"registered rule -> a = {a}")


def build(in_f: str, out_f: str, a: float) -> None:
    x = F.frame(in_f)
    p0 = x["p0"].to_numpy()
    x = F.with_p(x, p0)
    apt, mo = x["apt"].to_numpy(), x["mo"].to_numpy()
    lz = apt == APT
    days = day_lams(x)
    on_day = np.zeros(len(p0), bool)
    for d in days:
        on_day[d["g"]] = True
    print(f"input {Path(in_f).name}: LSZH rows {lz.sum():,}, on solved days {(lz & on_day).sum():,}; "
          f"month medians before " + ", ".join(f"m{m} {v:+.1f}" for m, v in month_medians(x).items()))
    q = F.finish(x, shifted(x, days, a))
    out_v = p0.astype(np.int64).copy()
    out_v[lz] = q[lz]
    dd = out_v - p0.astype(np.int64)
    assert np.all(dd[~lz] == 0)
    for m in sorted(set(mo[lz])):
        k = lz & (mo == m)
        print(f"  m{m}: rows {k.sum():,}, changed {(dd[k] != 0).sum():,}, mean {dd[k].mean():+.2f}, RMS {np.sqrt(np.mean(dd[k] ** 2.0)):.2f}")
    print(f"all rows: RMS {np.sqrt(np.mean(dd ** 2.0)):.2f}, max |d| {np.abs(dd).max()}")
    # stop rule 2: re-solve on the output
    x2 = F.with_p(x, out_v.astype(float))
    print("month medians after " + ", ".join(f"m{m} {v:+.1f}" for m, v in month_medians(x2).items()))
    # stop rule 1: LOBT window, template
    lt = (x["MVT_TIME_UTC_mvt"] - x["LOBT_flt"]).dt.total_seconds().cast(pl.Float64).to_numpy()
    win = ~np.isnan(lt)
    viol = win & ((out_v < np.clip(lt - 3606, 0, None) - 0.5) | (out_v > lt + 3606 + 0.5))
    print(f"LOBT-window violations: {viol.sum()} (LSZH {(viol & lz).sum()})")
    tpl = pl.read_parquet(scope.TEMPLATE)
    out = tpl.select("MVT_ID_mvt").join(
        pl.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"], "TAXITIME_SEC_mvt": out_v.astype(np.int32)}).with_columns(
            pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
    assert out.height == tpl.height and out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
    assert out["MVT_ID_mvt"].n_unique() == out.height
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    if sys.argv[1] == "--check2025":
        check2025()
    else:
        build(sys.argv[1], sys.argv[2], float(sys.argv[3]))
