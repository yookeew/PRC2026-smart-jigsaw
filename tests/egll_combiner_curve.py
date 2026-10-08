"""Scoping: would more EGLL ADS-B training days help the combiner? (PROGRESS §109; no new data fetched)

The production combiner (src/post/adsb_combiner.py --full) is trained on the Jan+Jul 2025 holdout plus the 115 OOF-month
ADS-B days, all non-LIRF airports pooled (airport is a feature). In 2026 it serves ~19.6k EGLL rows with detected
pushbacks, from thin EGLL training support (§95).
Learning curve on the OOF-month ADS-B rows (honest OOF stack s), labels <= 5 h, non-LIRF: EGLL days are split into
5 folds by day; for each fold, the combiner (adsb_combiner_test.features / fit_combiner) is trained on all other
airports' rows plus a fraction f in {0.25, 0.5, 1.0} of the remaining EGLL days, and scored on the held-out EGLL day
fold (rows with an ADS-B tier). Reports EGLL RMSE of s and of s + combiner per f, and the EGLL training rows.

Run:  PRC_LGB=lgb_mixed .venv/Scripts/python.exe -u tests/egll_combiner_curve.py
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
sys.argv = [sys.argv[0], "--full"]
import adsb_combiner_test as C  # noqa: E402
import adsb_combiner_v2_test as C2  # noqa: E402


def main() -> None:
    import adsb_combiner_12m_test as T12
    import adsb_combiner_fullyear_test as TF
    T12.MONTHS = TF.FULL
    ex = T12.oof_frame().filter((pl.col("ADEP_mvt") != "LIRF") & (pl.col("taxi") <= C.TRIM))
    airports = sorted(ex["ADEP_mvt"].unique().to_list())
    X = C.features(ex, ex["s"].to_numpy(), airports).to_numpy().astype(np.float64)
    y = (ex["taxi"] - ex["s"]).to_numpy()
    has = ex.select(C2.has_adsb()).to_series().to_numpy()
    apt = ex["ADEP_mvt"].to_numpy()
    day = ex["day"].cast(pl.Utf8).to_numpy()
    inner = (pd := np.array([int(d[-2:]) for d in day])) % 5 == 0
    eg_days = np.array(sorted(set(day[apt == "EGLL"])))
    rng = np.random.default_rng(0)
    rng.shuffle(eg_days)
    folds = np.array_split(eg_days, 5)
    print(f"OOF ADS-B rows (non-LIRF, <= 5 h) {len(y):,}; EGLL {int((apt == 'EGLL').sum()):,} on {len(eg_days)} days, "
          f"with an ADS-B tier {int(((apt == 'EGLL') & has).sum()):,}")
    res = {f: [] for f in (0.25, 0.5, 1.0)}
    base_se, n_te = 0.0, 0
    for i, fd in enumerate(folds):
        te = (apt == "EGLL") & np.isin(day, fd)
        other_eg = [d for d in eg_days if d not in set(fd)]
        k_te = te & has
        base_se += float(np.sum(y[k_te] ** 2)); n_te += int(k_te.sum())
        for f in res:
            keep_days = set(other_eg[: int(round(f * len(other_eg)))])
            tr = (apt != "EGLL") | np.isin(day, list(keep_days))
            tr &= ~te
            m, best = C.fit_combiner(X[tr], y[tr], inner[tr])
            p = m.predict(X[k_te])
            res[f].append((float(np.sum((y[k_te] - p) ** 2)), int(((apt == "EGLL") & tr & has).sum())))
            print(f"  fold {i} f {f:.2f}: test rows {k_te.sum():,}; RMSE s {np.sqrt(np.mean(y[k_te] ** 2)):.1f} -> s + combiner "
                  f"{np.sqrt(res[f][-1][0] / k_te.sum()):.1f} (EGLL ADS-B training rows {res[f][-1][1]:,})", flush=True)
    print(f"\nEGLL held-out ADS-B rows {n_te:,}: RMSE of s alone {np.sqrt(base_se / n_te):.1f}")
    for f, v in res.items():
        print(f"  EGLL training fraction {f:.2f} (mean EGLL ADS-B training rows {np.mean([b for _, b in v]):,.0f}): "
              f"RMSE s + combiner {np.sqrt(sum(a for a, _ in v) / n_te):.1f}")


if __name__ == "__main__":
    main()
