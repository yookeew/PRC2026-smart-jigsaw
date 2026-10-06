"""Screen of reports/base_direct_nolirf_preregistration.md (LightGBM, holdout only).

  fit       target="direct_nolirf" holdout fit -> cache/eval/lgb_direct_nolirf_nb_holdout_ev.parquet
  compare   new direct regressor on non-LIRF, non-prior rows vs the cached lgb_mixed_nb base

Run:  .venv/Scripts/python.exe -u src/models/run_direct_nolirf.py fit
      .venv/Scripts/python.exe -u src/models/run_direct_nolirf.py compare
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FEAT = ROOT / "cache" / "features_nb"
EVAL = ROOT / "cache" / "eval"
REF = EVAL / "lgb_mixed_nb_holdout_ev.parquet"
EV = EVAL / "lgb_direct_nolirf_nb_holdout_ev.parquet"


def fit() -> None:
    from models.fit import run
    if EV.exists():
        print(f"{EV.name} exists, nothing to do")
        return
    run(engine="lgb", target="direct_nolirf", seed=42, feat_dir=FEAT, name="lgb_direct_nolirf_nb", submit=False, ev_out=EV)


def compare() -> None:
    from models.fit import CEIL, ECHO_WIDE_CEIL, FLOOR
    a = pl.read_parquet(REF).select("MVT_ID_mvt", "ADEP_mvt", "taxi", "ym", "echo_prob", "use_prior",
                                    off="sched_takeoff_offset", ref="pred", raw_ref="taxi_model_raw")
    b = pl.read_parquet(EV).select("MVT_ID_mvt", raw_new="taxi_model_raw")
    d = a.join(b, on="MVT_ID_mvt", how="left")
    assert d.height == a.height and d["raw_new"].null_count() == 0
    nm = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{m}-01_2025-{n}-01.parquet"))
                    .filter(pl.col("PHASE_mvt") == "DEP")
                    .select(pl.col("MVT_ID_mvt").cast(pl.Float64), nm=pl.col("FLIGHT_ID_mvt").is_null())
                    for m, n in (("01", "02"), ("07", "08"))]).collect()
    d = d.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(nm, on="MVT_ID_mvt", how="left")
    y, ref, raw_ref, raw_new = (d[c].to_numpy().astype(float) for c in ("taxi", "ref", "raw_ref", "raw_new"))
    p, off = d["echo_prob"].to_numpy().astype(float), d["off"].to_numpy().astype(float)
    sel = ((d["ADEP_mvt"] != "LIRF") & ~d["use_prior"]).to_numpy()
    recon = lambda raw: p[sel] * np.clip(off[sel], 0, ECHO_WIDE_CEIL) + (1 - p[sel]) * np.clip(raw, FLOOR, CEIL)
    chk = np.abs(recon(raw_ref[sel]) - ref[sel]).max()
    print(f"rows changed: {sel.sum():,} of {len(y):,}; reconstruction check max |diff| {chk:.3g} s")
    assert chk < 1e-3
    q = ref.copy()
    q[sel] = recon(raw_new[sel])
    ym, apt = d["ym"].to_numpy(), d["ADEP_mvt"].to_numpy()
    body = (~d["nm"].to_numpy()) & (y <= 3600)
    r = lambda v, k: float(np.sqrt(np.mean((y - v)[k] ** 2)))
    rows = [("body pooled", body), ("body Jan", body & (ym == "2025-01")), ("body Jul", body & (ym == "2025-07")),
            ("trimmed (<=5h)", y <= 18000), ("full", np.ones_like(body))]
    rows += [(f"  body {x}", body & (apt == x)) for x in sorted(set(apt))]
    print(f"{'slice':18s} {'base':>9s} {'no LIRF':>9s} {'delta':>8s}")
    for nme, k in rows:
        print(f"{nme:18s} {r(ref, k):9.2f} {r(q, k):9.2f} {r(q, k) - r(ref, k):+8.2f}")
    g = {n: r(ref, k) - r(q, k) for n, k in (("pooled", body), ("Jan", body & (ym == "2025-01")),
                                             ("Jul", body & (ym == "2025-07")))}
    full_ok = r(q, np.ones_like(body)) <= r(ref, np.ones_like(body))
    ok = g["pooled"] >= 1.0 and g["Jan"] > 0 and g["Jul"] > 0 and full_ok
    print(f"\nSCREEN: body gain {g}, full not worse {full_ok} -> {'QUALIFIES for the full build' if ok else 'does not qualify'}")


if __name__ == "__main__":
    {"fit": fit, "compare": compare}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
