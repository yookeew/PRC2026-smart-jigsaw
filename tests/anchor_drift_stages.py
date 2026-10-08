"""Which stage moves LSZH (and others) in 2026? Offset of each stage's prediction vs the AOBT_3 anchor and the
ADS-B taxi (tiers dwell/pass), clipped mean, 2025 holdout vs 2026 ranking. Follow-up to anchor_drift_triangulate.py.

Run:  .venv/Scripts/python.exe -u tests/anchor_drift_stages.py [APT ...]
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_drift_triangulate as T  # noqa: E402

ROOT, EV, SUB = T.ROOT, T.ROOT / "cache" / "eval", T.ROOT / "data" / "submissions"
STAGES = {  # name: (2025 holdout file, column), 2026 file
    "base": ((EV / "lgb_mixed_nb_holdout_ev.parquet", "pred"), SUB / "lgb_mixed_nb.parquet"),
    "cat": ((EV / "cat_mixed_holdout_ev.parquet", "pred"), SUB / "cat_mixed.parquet"),
    "corrected": ((EV / "ctxc1sel_holdout_ev.parquet", "pred"), SUB / "ctxc1sel.parquet"),
    "v42": ((EV / "v42_chain_holdout.parquet", "v42"), SUB / "smart-jigsaw_v42.parquet"),
}


def main() -> None:
    apts = sys.argv[1:] or ["LSZH", "EDDF", "EDDM", "LEBL", "LFPG", "LTFM"]
    allm = pl.concat([T.frame(str(T.RAW / "training_*.parquet"), 2025),
                      T.frame(str(ROOT / "data" / "ranking" / "ranking.parquet"), 2026)]).collect()
    det = pl.concat([pl.read_parquet(p) for p in sorted(T.DET.glob("day=202[56]-0[17]*.parquet"))], how="diagonal")
    det = det.select(pl.col("MVT_ID_mvt").cast(pl.Int64), "adsb_tier", pb=pl.col("adsb_pushback_ts").cast(pl.Int64))
    dep = (allm.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("nm") & pl.col("a").is_not_null() & pl.col("apt").is_in(apts))
           .join(det, on="MVT_ID_mvt", how="left")
           .with_columns(at=(pl.col("t") - pl.col("pb")).cast(pl.Float64), good=pl.col("adsb_tier").is_in(T.GOOD).fill_null(False)))
    for n, ((f25, c25), f26) in STAGES.items():
        p = pl.concat([pl.read_parquet(f25).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col(c25).cast(pl.Float64).alias(n)),
                       pl.read_parquet(f26).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(n))])
        dep = dep.join(p, on="MVT_ID_mvt", how="left")
    cm = lambda e: e.clip(-T.CLIP, T.CLIP).mean()
    cols = ["y"] + list(STAGES)
    g = dep.group_by("apt", "mo", "yr").agg(
        *[cm(pl.col(c) - pl.col("a")).alias(f"{c}-a") for c in cols],
        *[cm((pl.col(c) - pl.col("at")).filter(pl.col("good"))).alias(f"{c}-ad") for c in cols],
        n_ad=pl.col("good").sum()).sort("apt", "mo", "yr")
    f = lambda v: f"{v:7.0f}" if v is not None else "      -"
    for suf, lab in (("-a", "vs AOBT_3 anchor"), ("-ad", "vs ADS-B taxi (dwell/pass)")):
        print(f"\nclipped-mean offset of each stage {lab}")
        print(f"{'apt':5s} {'mo':>2s} {'yr':>4s} " + " ".join(f"{c:>9s}" for c in cols) + ("    n_ad" if suf == "-ad" else ""))
        for r in g.iter_rows(named=True):
            print(f"{r['apt']:5s} {r['mo']:2d} {r['yr']:4d} " + " ".join(f"  {f(r[c + suf])}" for c in cols)
                  + (f" {r['n_ad']:7d}" if suf == "-ad" else ""))


if __name__ == "__main__":
    main()
