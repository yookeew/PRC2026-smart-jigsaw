"""Stop rule of reports/ctx_sel_eddm_preregistration.md: block-9 inputs (own-row ADS-B, context_corrector_test.adsb_block)
at EDDM and LEMD, 2025 rows on fetched ADS-B days vs 2026 ranking rows (OOF / holdout stack on 2025, production stack on
2026, as tests/ctx_hour2_drift.py). Stop if |SMD| > 0.25 for any input at an airport.

Run:  .venv/Scripts/python.exe tests/ctx_sel_drift.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import context_corrector_test as CC  # noqa: E402
import ctx_hour2_drift as H  # noqa: E402
import oof_context_corrector_test as O  # noqa: E402

APTS = ("EDDM", "LEMD")


def main() -> None:
    dep_all, _ = CC.movements(str(CC.RAW / "training_*.parquet"))
    pr = O.preds()
    dep = pr.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="left")
    s25 = H.win(pr.select("lgb", "cat").to_numpy() @ H.W, (dep["T"] - dep["t_LOBT_flt"]).to_numpy())
    rd = lambda f, c: pl.read_parquet(ROOT / "data/submissions" / f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    dep_r, _ = CC.movements(str(CC.RANKING))
    rk = dep_r.select("MVT_ID_mvt").join(rd("lgb_mixed_nb.parquet", "lgb"), on="MVT_ID_mvt", how="left").join(rd("cat_mixed.parquet", "cat"), on="MVT_ID_mvt", how="left")
    s26 = H.win(rk.select("lgb", "cat").to_numpy() @ H.W, (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy())
    a = pl.concat([dep.select("apt"), CC.adsb_block(dep, s25)], how="horizontal").filter(pl.col("ab_avail") == 1)
    b = pl.concat([dep_r.select("apt"), CC.adsb_block(dep_r, s26)], how="horizontal").filter(pl.col("ab_avail") == 1)
    cols = [c for c in a.columns if c not in ("apt", "ab_avail")]
    worst = {}
    print(f"{'airport':8s} {'input':16s} {'mean25':>9s} {'mean26':>9s} {'sd25':>8s} {'SMD':>7s} {'null25':>7s} {'null26':>7s}")
    for ap in APTS:
        xa, xb = a.filter(pl.col("apt") == ap), b.filter(pl.col("apt") == ap)
        worst[ap] = 0.0
        for c in cols:
            x, z = xa[c].drop_nulls().drop_nans().to_numpy(), xb[c].drop_nulls().drop_nans().to_numpy()
            if len(x) < 100 or len(z) < 100 or x.std() == 0:
                continue
            smd = (z.mean() - x.mean()) / x.std()
            worst[ap] = max(worst[ap], abs(smd))
            print(f"{ap:8s} {c:16s} {x.mean():9.1f} {z.mean():9.1f} {x.std():8.1f} {smd:+7.3f} "
                  f"{xa[c].null_count() / xa.height:7.3f} {xb[c].null_count() / xb.height:7.3f}")
    for ap, w in worst.items():
        print(f"{ap}: max |SMD| {w:.3f} -> {'pass' if w <= 0.25 else 'STOP'}")


if __name__ == "__main__":
    main()
