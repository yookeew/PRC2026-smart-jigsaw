"""Condition (f) of reports/ctx_flow_preregistration.md: block-13 inputs, 2025 training rows vs 2026 ranking rows.

The inputs need no prediction, so each frame is the corrector's own: the 2025 departures with an OOF / holdout stack
(as ctx_armc.py prod's training rows) and the ranking departures. SMD = (mean 2026 - mean 2025) / sd 2025, pooled and
per airport.

Run:  .venv/Scripts/python.exe tests/ctx_flow_drift.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ["CTX_FLOW"] = "1"
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import context_corrector_test as CC  # noqa: E402
import oof_context_corrector_test as O  # noqa: E402
from post import flow_feats as FF  # noqa: E402


def block(dep):
    d = dep.select("MVT_ID_mvt", "apt", "ades", t="T", s="t_SCHED_TIME_UTC_mvt", lo="t_LOBT_flt")
    return d.select("MVT_ID_mvt", "apt").join(FF.build(d), on="MVT_ID_mvt", how="left")


def main() -> None:
    dep_all, _ = CC.movements(str(CC.RAW / "training_*.parquet"))
    dep = O.preds().select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="left")
    dep_r, _ = CC.movements(str(CC.RANKING))
    a, b = block(dep), block(dep_r)
    worst = 0.0
    print(f"{'input':10s} {'mean25':>9s} {'mean26':>9s} {'sd25':>8s} {'SMD':>7s} {'null25':>7s} {'null26':>7s}  max |SMD| by airport")
    for c in FF.FLOW_COLS:
        x, z = a[c].drop_nulls().to_numpy(), b[c].drop_nulls().to_numpy()
        smd = (z.mean() - x.mean()) / x.std()
        worst = max(worst, abs(smd))
        per = {}
        for ap in sorted(a["apt"].unique().to_list()):
            xa = a.filter(pl.col("apt") == ap)[c].drop_nulls().to_numpy(); za = b.filter(pl.col("apt") == ap)[c].drop_nulls().to_numpy()
            if len(xa) > 100 and len(za) > 100 and xa.std() > 0:
                per[ap] = (za.mean() - xa.mean()) / xa.std()
        k = max(per, key=lambda q: abs(per[q]))
        print(f"{c:10s} {x.mean():9.2f} {z.mean():9.2f} {x.std():8.2f} {smd:+7.3f} {a[c].null_count() / a.height:7.3f} "
              f"{b[c].null_count() / b.height:7.3f}  {k} {per[k]:+.2f}")
    print(f"max pooled |SMD| {worst:.3f} -> condition (f) {'PASS' if worst <= 0.25 else 'FAIL'}")


if __name__ == "__main__":
    main()
