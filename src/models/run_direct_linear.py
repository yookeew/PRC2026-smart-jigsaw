"""Screen of reports/base_linear_tree_preregistration.md (LightGBM direct regressor with linear leaves, holdout only).

  fit       PRC_LINEAR_TREE=1, target="direct" -> cache/eval/lgb_direct_linear_nb_holdout_ev.parquet
  compare   new direct regressor on non-LIRF, non-prior rows vs the cached lgb_mixed_nb base

Run:  .venv/Scripts/python.exe -u src/models/run_direct_linear.py fit
      .venv/Scripts/python.exe -u src/models/run_direct_linear.py compare
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
if len(sys.argv) > 1 and sys.argv[1] == "fit":
    os.environ["PRC_LINEAR_TREE"] = "1"   # read by models.fit at import time

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import models.run_direct_nolirf as R  # noqa: E402  (same comparison, different file)

R.EV = R.EVAL / "lgb_direct_linear_nb_holdout_ev.parquet"


def fit() -> None:
    from models.fit import LINEAR_TREE, run
    assert LINEAR_TREE
    if R.EV.exists():
        print(f"{R.EV.name} exists, nothing to do")
        return
    run(engine="lgb", target="direct", seed=42, feat_dir=R.FEAT, name="lgb_direct_linear_nb", submit=False, ev_out=R.EV)


def compare() -> None:
    R.compare()
    a = pl.read_parquet(R.REF).select("MVT_ID_mvt", "ADEP_mvt", "use_prior", "taxi", raw_ref="taxi_model_raw")
    b = pl.read_parquet(R.EV).select("MVT_ID_mvt", raw_new="taxi_model_raw")
    d = a.join(b, on="MVT_ID_mvt")
    sel = ((d["ADEP_mvt"] != "LIRF") & ~d["use_prior"]).to_numpy()
    out = lambda c: float(np.mean(((d[c].to_numpy() < 0) | (d[c].to_numpy() > 7200))[sel]))
    o_ref, o_new = out("raw_ref"), out("raw_new")
    print(f"\nextrapolation: raw outside [0, 7200] on changed rows: cached {o_ref:.4%}, new {o_new:.4%} "
          f"(min {d['raw_new'].to_numpy()[sel].min():.0f}, max {d['raw_new'].to_numpy()[sel].max():.0f}) "
          f"-> {'OK' if o_new <= 2 * max(o_ref, 1e-6) else 'FAILS rule 4'}")
    # outcome-conditioned buckets, information only
    from models.fit import CEIL, ECHO_WIDE_CEIL, FLOOR
    e = pl.read_parquet(R.REF).select("MVT_ID_mvt", "taxi", "echo_prob", off="sched_takeoff_offset", ref="pred").join(b, on="MVT_ID_mvt")
    y, ref = e["taxi"].to_numpy(), e["ref"].to_numpy().copy()
    p, off, raw = e["echo_prob"].to_numpy(), e["off"].to_numpy().astype(float), e["raw_new"].to_numpy()
    new = ref.copy()
    new[sel] = p[sel] * np.clip(off[sel], 0, ECHO_WIDE_CEIL) + (1 - p[sel]) * np.clip(raw[sel], FLOOR, CEIL)
    for lo, hi, nme in ((1200, 2400, "20-40 min"), (2400, 18000, "40 min-5 h")):
        k = sel & (y > lo) & (y <= hi)
        print(f"  label {nme:10s} (changed rows, info only): {np.sqrt(np.mean((y - ref)[k] ** 2)):.1f} -> {np.sqrt(np.mean((y - new)[k] ** 2)):.1f}")


if __name__ == "__main__":
    {"fit": fit, "compare": compare}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
