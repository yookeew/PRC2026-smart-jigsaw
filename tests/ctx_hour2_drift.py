"""Condition (f) of reports/ctx_hour2_preregistration.md: block-12 inputs, 2025 training rows vs 2026 ranking rows.

Neighbours' p = the stack each frame's corrector sees: OOF / holdout stack on 2025 (as ctx_armc.py prod's training
rows), the production stack on the ranking rows. SMD = (mean 2026 - mean 2025) / sd 2025, pooled and per airport.

Run:  .venv/Scripts/python.exe tests/ctx_hour2_drift.py
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
import oof_context_corrector_test as O  # noqa: E402
from post import hour_feats as HF  # noqa: E402

W = np.array([0.577, 0.428])     # OOF stack weights (logs/v64/armc_prod_*.log)


def win(q, lt):
    return np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))


def block(dep, s):
    d = pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], "apt": dep["apt"], "RUNWAY_mvt": dep["rwy"], "t": dep["T"],
                      "lo": dep["t_LOBT_flt"], "a3": dep["t_AOBT_3_flt"], "s": dep["t_SCHED_TIME_UTC_mvt"], "p": s})
    return d.select("MVT_ID_mvt", "apt").join(HF.build_rwy(d), on="MVT_ID_mvt", how="left")


def main() -> None:
    dep_all, _ = CC.movements(str(CC.RAW / "training_*.parquet"))
    pr = O.preds()
    dep = pr.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="left")
    s25 = win(pr.select("lgb", "cat").to_numpy() @ W, (dep["T"] - dep["t_LOBT_flt"]).to_numpy())
    rd = lambda f, c: pl.read_parquet(ROOT / "data/submissions" / f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    dep_r, _ = CC.movements(str(CC.RANKING))
    rk = dep_r.select("MVT_ID_mvt").join(rd("lgb_mixed_nb.parquet", "lgb"), on="MVT_ID_mvt", how="left").join(rd("cat_mixed.parquet", "cat"), on="MVT_ID_mvt", how="left")
    assert rk["lgb"].null_count() == 0 and rk["cat"].null_count() == 0
    s26 = win(rk.select("lgb", "cat").to_numpy() @ W, (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy())
    a, b = block(dep, s25), block(dep_r, s26)
    worst = 0.0
    print(f"{'input':10s} {'mean25':>9s} {'mean26':>9s} {'sd25':>8s} {'SMD':>7s} {'null25':>7s} {'null26':>7s}  max |SMD| by airport")
    for c in HF.RWY_COLS:
        x, z = a[c].drop_nulls().to_numpy(), b[c].drop_nulls().to_numpy()
        smd = (z.mean() - x.mean()) / x.std()
        worst = max(worst, abs(smd))
        per = {}
        for ap in sorted(a["apt"].unique().to_list()):
            xa = a.filter(pl.col("apt") == ap)[c].drop_nulls().to_numpy(); za = b.filter(pl.col("apt") == ap)[c].drop_nulls().to_numpy()
            if len(xa) > 100 and len(za) > 100:
                per[ap] = (za.mean() - xa.mean()) / xa.std()
        k = max(per, key=lambda q: abs(per[q]))
        print(f"{c:10s} {x.mean():9.1f} {z.mean():9.1f} {x.std():8.1f} {smd:+7.3f} {a[c].null_count() / a.height:7.3f} "
              f"{b[c].null_count() / b.height:7.3f}  {k} {per[k]:+.2f}")
    print(f"max pooled |SMD| {worst:.3f} -> condition (f) {'PASS' if worst <= 0.25 else 'FAIL'}")


if __name__ == "__main__":
    main()
