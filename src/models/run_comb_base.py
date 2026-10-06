"""Base `lgb_comb_nb` of reports/base_comb_preregistration.md: no-LIRF direct + anchored members.

Stages (each resumable; finished pieces are skipped):
  fits      holdout, 10 OOF folds and the 12-month production refit of the two members
  combine   writes lgb_comb_nb in the lgb_mixed_nb formats:
              cache/eval/lgb_comb_nb_holdout_ev.parquet, cache/oof/lgb_comb_nb/fold=*.parquet,
              data/submissions/lgb_comb_nb.parquet
  compare   base-level slices, lgb_comb_nb vs lgb_mixed_nb on the holdout

Run:  .venv/Scripts/python.exe -u src/models/run_comb_base.py fits
      .venv/Scripts/python.exe -u src/models/run_comb_base.py combine
      .venv/Scripts/python.exe -u src/models/run_comb_base.py compare
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
EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"
REF = "lgb_mixed_nb"
NAME = "lgb_comb_nb"
MEMBERS = {"direct_nolirf": "lgb_direct_nolirf_nb", "anchored_nolirf": "lgb_anchored_nolirf_nb"}
W_ANCHOR = 0.5


def fits() -> None:
    from models.fit import run, run_oof
    for target, name in MEMBERS.items():
        ev = EVAL / f"{name}_holdout_ev.parquet"
        if not (ev.exists() and ev.with_suffix(".rounds.json").exists()):
            run(engine="lgb", target=target, seed=42, feat_dir=FEAT, name=name, submit=False, ev_out=ev)
        run_oof(engine="lgb", target=target, seed=42, feat_dir=FEAT, name=name, out_dir=OOF / name)
        rk = EVAL / f"{name}_rank.parquet"
        if not rk.exists():
            run(engine="lgb", target=target, seed=42, feat_dir=FEAT, name=name, submit=True, ev_out=ev,
                rank_out=rk, resume=True)


def _combine(ref: pl.DataFrame, d: pl.DataFrame, a: pl.DataFrame, apt: pl.DataFrame, has_a3: pl.DataFrame) -> pl.DataFrame:
    """ref with `pred` replaced on non-LIRF, non-prior rows (reports/base_comb_preregistration.md)."""
    from models.fit import CEIL, ECHO_WIDE_CEIL, FLOOR
    x = (ref.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id"))
         .join(d.select(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id"), raw_d="taxi_model_raw",
                        p_d="echo_prob", up_d="use_prior"), on="_id", how="left")
         .join(a.select(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id"), raw_a="taxi_model_raw"), on="_id", how="left")
         .join(apt, on="_id", how="left").join(has_a3, on="_id", how="left"))
    assert x.height == ref.height and x["raw_d"].null_count() == 0 and x["_apt"].null_count() == 0
    raw_d, raw_a = x["raw_d"].to_numpy().astype(float), x["raw_a"].fill_null(np.nan).to_numpy().astype(float)
    p, off = x["p_d"].to_numpy().astype(float), x["_off"].to_numpy().astype(float)
    sel = ((x["_apt"] != "LIRF") & ~x["up_d"]).to_numpy()
    both = sel & x["_a3"].to_numpy() & ~np.isnan(raw_a)
    raw = raw_d.copy()
    raw[both] = (1 - W_ANCHOR) * raw_d[both] + W_ANCHOR * raw_a[both]
    new = p * np.clip(off, 0, ECHO_WIDE_CEIL) + (1 - p) * np.clip(raw, FLOOR, CEIL)
    pred = x["pred"].to_numpy().astype(float) if "pred" in x.columns else x["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    pred[sel] = new[sel]
    col = "pred" if "pred" in ref.columns else "TAXITIME_SEC_mvt"
    out = pl.Series(col, pred)
    if ref.schema[col] in (pl.Int32, pl.Int64):
        out = out.round().cast(ref.schema[col])
    print(f"  {sel.sum():,} of {ref.height:,} rows replaced ({both.sum():,} with the anchored member)")
    return ref.with_columns(out)


def _meta(srcs: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    f = pl.concat([pl.scan_parquet(s).filter(pl.col("PHASE_mvt") == "DEP").select(
        _id=pl.col("MVT_ID_mvt").cast(pl.Float64), _apt=pl.col("ADEP_mvt"),
        _a3=pl.col("AOBT_3_flt").is_not_null(),
        _off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
        for s in srcs]).collect().unique("_id")
    return f.select("_id", "_apt"), f.select("_id", "_a3", "_off")


def combine() -> None:
    apt, a3 = _meta([str(ROOT / "data" / "raw" / "training_*.parquet"), str(ROOT / "data" / "ranking" / "ranking.parquet")])
    d_name, a_name = MEMBERS["direct_nolirf"], MEMBERS["anchored_nolirf"]
    print("holdout")
    ev = _combine(pl.read_parquet(EVAL / f"{REF}_holdout_ev.parquet"), pl.read_parquet(EVAL / f"{d_name}_holdout_ev.parquet"),
                  pl.read_parquet(EVAL / f"{a_name}_holdout_ev.parquet"), apt, a3)
    ev.write_parquet(EVAL / f"{NAME}_holdout_ev.parquet")
    (OOF / NAME).mkdir(parents=True, exist_ok=True)
    for f in sorted(p.name for p in (OOF / REF).glob("fold=*.parquet")):
        print(f"oof {f}")
        o = _combine(pl.read_parquet(OOF / REF / f), pl.read_parquet(OOF / d_name / f), pl.read_parquet(OOF / a_name / f), apt, a3)
        o.write_parquet(OOF / NAME / f)
    print("ranking")
    rk = _combine(pl.read_parquet(SUB / f"{REF}.parquet"), pl.read_parquet(EVAL / f"{d_name}_rank.parquet"),
                  pl.read_parquet(EVAL / f"{a_name}_rank.parquet"), apt, a3)
    rk.write_parquet(SUB / f"{NAME}.parquet")


def compare() -> None:
    a = pl.read_parquet(EVAL / f"{REF}_holdout_ev.parquet").select("MVT_ID_mvt", "ADEP_mvt", "taxi", "ym", ref="pred")
    b = pl.read_parquet(EVAL / f"{NAME}_holdout_ev.parquet").select("MVT_ID_mvt", new="pred")
    d = a.join(b, on="MVT_ID_mvt")
    nm = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{m}-01_2025-{n}-01.parquet"))
                    .filter(pl.col("PHASE_mvt") == "DEP")
                    .select(pl.col("MVT_ID_mvt").cast(pl.Float64), nm=pl.col("FLIGHT_ID_mvt").is_null())
                    for m, n in (("01", "02"), ("07", "08"))]).collect()
    d = d.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(nm, on="MVT_ID_mvt", how="left")
    y, p0, p1 = (d[c].to_numpy().astype(float) for c in ("taxi", "ref", "new"))
    ym, apt = d["ym"].to_numpy(), d["ADEP_mvt"].to_numpy()
    body = (~d["nm"].to_numpy()) & (y <= 3600)
    r = lambda v, k: float(np.sqrt(np.mean((y - v)[k] ** 2)))
    rows = [("body pooled", body), ("body Jan", body & (ym == "2025-01")), ("body Jul", body & (ym == "2025-07")),
            ("trimmed (<=5h)", y <= 18000), ("full", np.ones_like(body))] + [(f"  body {x}", body & (apt == x)) for x in sorted(set(apt))]
    print(f"{'slice':18s} {REF:>13s} {NAME:>12s} {'delta':>8s}")
    for nme, k in rows:
        print(f"{nme:18s} {r(p0, k):13.2f} {r(p1, k):12.2f} {r(p1, k) - r(p0, k):+8.2f}")


if __name__ == "__main__":
    {"fits": fits, "combine": combine, "compare": compare}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
