"""Neighbour-state transfer decay, per reports/neighbour_decay_preregistration.md.

v30's part-(b) corrector (day-demeaned) fit on three 2025 months and scored on a month one
away (near) and three months 5-8 away (far), forward and backward. alpha = sum(c*r)/sum(c^2)
on each eval set is the SSE-optimal strength of the correction there.

Run:  .venv/Scripts/python.exe tests/neighbour_decay_test.py
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
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402
import neighbour_state_test as N  # noqa: E402
import neighbour_state_v2_test as N2  # noqa: E402

SPLITS = {
    "forward": (["2025-02", "2025-03", "2025-04"], ["2025-05"], ["2025-10", "2025-11", "2025-12"]),
    "backward": (["2025-10", "2025-11", "2025-12"], ["2025-09"], ["2025-02", "2025-03", "2025-04"]),
}


def alpha(c: np.ndarray, r: np.ndarray) -> float:
    return float((c * r).sum() / (c * c).sum())


def main() -> None:
    ex, lag = N.train_extra()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    airports = sorted(ex["ADEP_mvt"].unique().to_list())
    s = ex["s"].to_numpy()
    X = N.x_b(ex, s, N.nb_frame(ex, s, lag, arr), airports)
    y = (ex["taxi"] - ex["s"]).to_numpy()
    ok = (ex["taxi"] <= C.TRIM).to_numpy()
    inner = (ex["day"].dt.day() % 5 == 0).to_numpy()
    pop = ex.select(C2.has_adsb()).to_series().to_numpy()
    ym, apt, day = ex["ym"].to_numpy(), ex["ADEP_mvt"].to_numpy(), ex["day"].to_numpy()

    far_alphas = []
    for name, (fit_m, near_m, far_m) in SPLITS.items():
        tr = ok & np.isin(ym, fit_m)
        m, best = N.fit(X[tr], y[tr], inner[tr], N.CAT_B)
        print(f"\n== {name}: fit {fit_m} ({tr.sum():,} rows, best_iter {best})")
        for lab, months in (("near", near_m), ("far", far_m)):
            app = np.isin(ym, months) & ~pop
            c = np.zeros(len(y))
            c[app] = m.predict(X[app])
            ct = N2.day_demean(c, apt, day, app)
            ev = app & ok
            r, cc = y[ev], ct[ev]
            a = alpha(cc, r)
            base, new = np.sqrt(np.mean(r ** 2)), np.sqrt(np.mean((r - cc) ** 2))
            print(f"  {lab:<4} {months}: n={ev.sum():,}  alpha={a:.3f}  resid RMS {base:.2f} -> {new:.2f} "
                  f"({(new / base - 1) * 100:+.2f}%)  at alpha: {np.sqrt(np.mean((r - a * cc) ** 2)):.2f}")
            if lab == "far":
                far_alphas.append(a)
                t = pl.DataFrame({"a": apt[ev], "c": cc, "r": r}).group_by("a").agg(
                    n=pl.len(), alpha=(pl.col("c") * pl.col("r")).sum() / (pl.col("c") ** 2).sum()).sort("a")
                print("   per-airport far alpha:", [(k, n, round(v, 2)) for k, n, v in t.iter_rows()])

    a_far = float(np.mean(far_alphas))
    build = all(a < 0.8 for a in far_alphas) and a_far > 0.2
    print(f"\nalpha_far by direction {np.round(far_alphas, 3)}; mean {a_far:.3f}")
    print(f"DECISION: {'BUILD v31 = v27 + %.3f * (v30 - v27)' % a_far if build else 'NO v31 from this'}")


if __name__ == "__main__":
    main()
