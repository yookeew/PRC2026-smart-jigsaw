"""Second gate for v62's day-level recipe on Feb/Jun-type months (PROGRESS.md §132).

v62 applies S3 (learned day shift, tests/v61_learned_dayshift.py) at seven airports and keeps R0 (v55's uniform 0.75
shift) at LTFM. §131's table is for S3 at all eight airports. This reruns the same leave-period-out evaluation with the
same cached 2025 lambdas and reports, per month, S3-at-seven-airports + R0-at-LTFM vs R0 at all eight, over the eight
airports' rows (full and body), with February and June singled out.

Run:  .venv/Scripts/python.exe tests/final4m_s3_7apt_check.py
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

A7 = [a for a in T.A8 if a != "LTFM"]


def main() -> None:
    x = T.frame25()
    lam, q0 = L.lam_rows(x)
    p, y = x["p"].to_numpy().astype(float), x["y"].to_numpy()
    apt, mo, per, nmu = x["apt"].to_numpy(), x["mo"].to_numpy(), x["period"].to_numpy(), x["nmu"].to_numpy()
    b = L.band(x["ts"].to_numpy() - p)
    td = L.tod(x["hour"].to_numpy())
    dayk = (x["apt"] + "_" + x["day"].cast(pl.Utf8)).to_numpy()
    in8, in7 = np.isin(apt, T.A8), np.isin(apt, A7)
    ok = in8 & np.isfinite(lam)
    lam0 = np.where(np.isfinite(lam), lam, 0.0)
    acell = np.where(in8, np.searchsorted(np.array(T.A8), apt), -1)
    cell = np.where(in8, L.cells("S3", apt, b, td, lam0), -1)
    body, trim = ~nmu & (y <= 3600), y <= 18000
    q8 = p.copy()
    for pk in T.PERIODS:
        s, _ = L.fit("S3", cell, acell, lam0, p - y, dayk, ok & trim & (per != pk))
        k = ok & (per == pk)
        q8[k] = np.maximum(p[k] - np.array([s.get(c, 0.0) for c in cell[k]]) * lam0[k], 0)
    q7 = np.where(in7, q8, q0)                    # v62: S3 at seven airports, R0 at LTFM
    rm = lambda q, m: float(np.sqrt(np.mean((y[m] - q[m]) ** 2)))  # noqa: E731
    print("vs R0 at all eight airports (2025, leave-period-out slopes); rows = the eight airports")
    print("month   full S3x8  full S3x7(v62)  body S3x8  body S3x7(v62)")
    for m in sorted(set(mo)):
        k = in8 & (mo == m)
        print(f"{m}  {rm(q8, k) - rm(q0, k):+9.3f}  {rm(q7, k) - rm(q0, k):+13.3f}  {rm(q8, k & body) - rm(q0, k & body):+9.3f}  "
              f"{rm(q7, k & body) - rm(q0, k & body):+13.3f}")
    k = in8
    print(f"all     {rm(q8, k) - rm(q0, k):+9.3f}  {rm(q7, k) - rm(q0, k):+13.3f}  {rm(q8, k & body) - rm(q0, k & body):+9.3f}  "
          f"{rm(q7, k & body) - rm(q0, k & body):+13.3f}")
    for m in ("2025-02", "2025-06"):
        k = mo == m
        print(f"{m} per airport, S3x7 vs R0 (full / body):",
              {a: (round(rm(q7, k & (apt == a)) - rm(q0, k & (apt == a)), 2),
                   round(rm(q7, k & (apt == a) & body) - rm(q0, k & (apt == a) & body), 2)) for a in A7})


if __name__ == "__main__":
    main()
