"""Base-regressor screen of reports/base_label_clip_preregistration.md (LightGBM, holdout only).

  c        PRC_LABEL_CLIP=1, L2     -> cache/eval/lgb_mixed_nb_clipl2_holdout_ev.parquet
  b        PRC_LABEL_CLIP=1, Huber  -> cache/eval/lgb_mixed_nb_cliphub_holdout_ev.parquet
  compare <c|b>   the arm vs the existing lgb_mixed_nb holdout predictions (arm a)

Run:  .venv/Scripts/python.exe src/models/run_label_clip.py c
      .venv/Scripts/python.exe src/models/run_label_clip.py compare c
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
if len(sys.argv) > 1 and sys.argv[1] in ("b", "c"):
    os.environ["PRC_LABEL_CLIP"] = "1"   # read by models.fit at import time

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FEAT = ROOT / "cache" / "features_nb"
EVAL = ROOT / "cache" / "eval"
REF = EVAL / "lgb_mixed_nb_holdout_ev.parquet"
ARMS = {"c": ("l2", "lgb_mixed_nb_clipl2"), "b": ("huber", "lgb_mixed_nb_cliphub")}


def fit_arm(arm: str) -> None:
    from models.fit import LABEL_CLIP, run
    assert LABEL_CLIP, "PRC_LABEL_CLIP must be on for arms b and c"
    loss, name = ARMS[arm]
    ev = EVAL / f"{name}_holdout_ev.parquet"
    if ev.exists():
        print(f"{ev.name} exists, nothing to do")
        return
    run(engine="lgb", target="mixed", seed=42, feat_dir=FEAT, name=name, loss=loss, submit=False, ev_out=ev)


def compare(arm: str) -> None:
    ev = EVAL / f"{ARMS[arm][1]}_holdout_ev.parquet"
    a = pl.read_parquet(REF).select("MVT_ID_mvt", "ADEP_mvt", "taxi", "ym", "has_aobt3", ref=pl.col("pred"))
    t = pl.read_parquet(ev).select("MVT_ID_mvt", new=pl.col("pred"))
    d = a.join(t, on="MVT_ID_mvt", how="inner")
    assert d.height == a.height, "arm and reference cover different rows"
    nm = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{m}-01_2025-{n}-01.parquet"))
                    .filter(pl.col("PHASE_mvt") == "DEP")
                    .select(pl.col("MVT_ID_mvt").cast(pl.Float64), nm=pl.col("FLIGHT_ID_mvt").is_null(),
                            day=pl.col("MVT_TIME_UTC_mvt").dt.date())
                    for m, n in (("01", "02"), ("07", "08"))]).collect()
    d = d.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(nm, on="MVT_ID_mvt", how="left")
    # disrupted days: top 5% of airport-days by mean true taxi, within each airport
    dm = d.group_by("ADEP_mvt", "day").agg(mt=pl.col("taxi").mean()).with_columns(
        q=pl.col("mt").quantile(0.95).over("ADEP_mvt"))
    d = d.join(dm, on=["ADEP_mvt", "day"], how="left").with_columns(disrupted=pl.col("mt") >= pl.col("q"))
    y, p0, p1 = (d[c].to_numpy().astype(float) for c in ("taxi", "ref", "new"))
    ym, apt = d["ym"].to_numpy(), d["ADEP_mvt"].to_numpy()
    body = (~d["nm"].to_numpy()) & (y <= 3600)
    r = lambda p, k: float(np.sqrt(np.mean((y - p)[k] ** 2)))
    rows = [("body pooled", body), ("body Jan", body & (ym == "2025-01")), ("body Jul", body & (ym == "2025-07")),
            ("trimmed (<=5h)", y <= 18000), ("full", np.ones_like(body)),
            ("disrupted days", d["disrupted"].to_numpy()), ("other days", ~d["disrupted"].to_numpy())]
    rows += [(f"  body {x}", body & (apt == x)) for x in sorted(set(apt))]
    print(f"arm {arm} ({ARMS[arm][1]}) vs arm a (lgb_mixed_nb)")
    print(f"{'slice':18s} {'a':>9s} {arm:>9s} {'delta':>8s}")
    for nme, k in rows:
        print(f"{nme:18s} {r(p0, k):9.2f} {r(p1, k):9.2f} {r(p1, k) - r(p0, k):+8.2f}")
    g = {n: r(p0, k) - r(p1, k) for n, k in (("pooled", body), ("Jan", body & (ym == "2025-01")),
                                             ("Jul", body & (ym == "2025-07")))}
    full_ok = r(p1, np.ones_like(body)) <= r(p0, np.ones_like(body))
    ok = g["pooled"] >= 1.0 and g["Jan"] > 0 and g["Jul"] > 0 and full_ok
    print(f"\nSCREEN: body gain {g}, full not worse {full_ok} -> {'QUALIFIES for rebuild' if ok else 'does not qualify'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ARMS:
        fit_arm(cmd)
    elif cmd == "compare" and len(sys.argv) > 2:
        compare(sys.argv[2])
    else:
        raise SystemExit(__doc__)
