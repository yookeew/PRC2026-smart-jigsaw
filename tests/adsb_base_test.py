"""Stage A of reports/adsb_base_preregistration.md: LightGBM with ADS-B base features vs without.

Both arms: LightGBM `target="mixed"` holdout predictions (fit on the 10 training months),
LOBT window applied. Primary: body RMSE (NM-matched, y <= 1 h), pooled and per month.
Also: trimmed RMSE (guard), body at each airport, and the v32 chain's body for reference.

Run:  .venv/Scripts/python.exe tests/adsb_base_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "cache" / "eval"


def main() -> None:
    raw = (pl.scan_parquet(str(ROOT / "data" / "raw" / "training_*.parquet")).filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col("MVT_ID_mvt").cast(pl.Int64), nm=pl.col("FLIGHT_ID_mvt").is_null(),
                   lt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64)).collect())
    ev = lambda n, c: pl.read_parquet(EVAL / f"{n}_holdout_ev.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").alias(c))
    d = (ev("lgb_mixed_nb", "base").join(ev("lgb_mixed_nba", "new"), on="MVT_ID_mvt")
         .join(pl.read_parquet(EVAL / "v31_chain_holdout.parquet").select("MVT_ID_mvt", "ADEP_mvt", "taxi", "ym", "pred_v31"), on="MVT_ID_mvt")
         .join(raw, on="MVT_ID_mvt"))
    lt = d["lt"].to_numpy()
    win = lambda p: np.where(np.isnan(lt), p, np.clip(p, np.clip(lt - 3606, 0, None), lt + 3606))
    b, n, ch = win(d["base"].to_numpy()), win(d["new"].to_numpy()), win(d["pred_v31"].to_numpy())
    y, apt, ym = d["taxi"].to_numpy(), d["ADEP_mvt"].to_numpy(), d["ym"].to_numpy()
    body = (~d["nm"].to_numpy()) & (y <= 3600)
    r = lambda p, k: float(np.sqrt(np.mean((y - p)[k] ** 2)))
    print(f"rows {len(y):,}")
    print(f"{'slice':22s} {'lgb_nb':>8s} {'+ADS-B':>8s} {'delta':>7s} {'v32 chain':>10s}")
    rows = [("body pooled", body), ("body Jan", body & (ym == "2025-01")), ("body Jul", body & (ym == "2025-07")),
            ("trimmed (<=5h)", y <= 18000), ("full", np.ones_like(body))]
    rows += [(f"  body {a}", body & (apt == a)) for a in sorted(set(apt))]
    for name, k in rows:
        print(f"{name:22s} {r(b, k):8.2f} {r(n, k):8.2f} {r(n, k) - r(b, k):+7.2f} {r(ch, k):10.2f}")
    g1 = r(b, body) - r(n, body)
    gj = r(b, body & (ym == "2025-01")) - r(n, body & (ym == "2025-01"))
    gl = r(b, body & (ym == "2025-07")) - r(n, body & (ym == "2025-07"))
    ok = g1 >= 5 and gj > 0 and gl > 0 and r(n, y <= 18000) <= r(b, y <= 18000)
    print(f"\nSTAGE A: pooled body gain {g1:.2f} (Jan {gj:.2f}, Jul {gl:.2f}) -> {'PASS -> stage B' if ok else 'FAIL -> stop'}")


if __name__ == "__main__":
    main()
