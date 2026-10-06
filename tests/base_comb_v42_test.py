"""Gate of reports/base_comb_preregistration.md, amendment A1: v42's recipe on the lgb_comb_nb base vs v42's chain.

Prerequisites (each its own process; run_comb_base.py fits + combine first):
  PRC_BASE=lgb_comb_nb ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1             .venv/Scripts/python.exe -u src/post/ctx_armc.py inputs
  PRC_BASE=lgb_comb_nb ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_ADSB=1  .venv/Scripts/python.exe -u src/post/ctx_armc.py inputs
  PRC_BASE=lgb_comb_nb ARMC_ARM=c1                                 .venv/Scripts/python.exe -u src/post/ctx_armc.py sel
Then this script: stage 2 on ctxc1sel_comb (as tests/v42_holdout_baseline.py), LIRF NM-unmatched rows from v41's chain,
compared with cache/eval/v42_chain_holdout.parquet; also the base-level gain and the base->chain transfer ratio.
Writes cache/eval/comb_v42_chain_holdout.parquet.

Run:  PRC_BASE=lgb_comb_nb PRC_LGB=ctxcorr2r .venv/Scripts/python.exe -u tests/base_comb_v42_test.py
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
EVAL, OOF = ROOT / "cache" / "eval", ROOT / "cache" / "oof"
SEL = "ctxc1sel_comb"


def main() -> None:
    assert os.environ.get("PRC_BASE") == "lgb_comb_nb" and os.environ.get("PRC_LGB") == "ctxcorr2r"
    import oof_context_corrector_stage2 as S2
    S2.EV, S2.OOFD = EVAL / f"{SEL}_holdout_ev.parquet", OOF / SEL
    S2.main()
    i64 = lambda f, **c: pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Int64), **c)  # noqa: E731
    d = (i64(EVAL / "ctxcorr_chain_holdout.parquet", chain="new")
         .join(i64(EVAL / "v41_chain_holdout.parquet", v41="new"), on="MVT_ID_mvt")
         .join(i64(EVAL / "v42_chain_holdout.parquet", v42="v42"), on="MVT_ID_mvt")
         .join(i64(EVAL / "lgb_mixed_nb_holdout_ev.parquet", b0="pred"), on="MVT_ID_mvt")
         .join(i64(EVAL / "lgb_comb_nb_holdout_ev.parquet", b1="pred"), on="MVT_ID_mvt"))
    raw = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet"))
                     .filter(pl.col("PHASE_mvt") == "DEP")
                     .select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                             nm=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), apt=pl.col("ADEP_mvt"))
                     for a, b in (("01", "02"), ("07", "08"))]).collect()
    n0 = d.height
    d = d.join(raw, on="MVT_ID_mvt")
    assert d.height == n0
    lirf_nmu = ((d["apt"] == "LIRF") & d["nm"]).to_numpy()
    new = np.where(lirf_nmu, d["v41"].to_numpy(), d["chain"].to_numpy())
    d = d.with_columns(new=pl.Series(new))
    d.select("MVT_ID_mvt", "new").write_parquet(EVAL / "comb_v42_chain_holdout.parquet")
    y, mo, apt = d["y"].to_numpy(), d["mo"].to_numpy(), d["apt"].to_numpy()
    body = (~d["nm"].to_numpy()) & (y <= 3600)
    r = lambda c, k: float(np.sqrt(np.mean((y - d[c].to_numpy())[k] ** 2)))  # noqa: E731
    sl = [("body pooled", body), ("body Jan", body & (mo == 1)), ("body Jul", body & (mo == 7)),
          ("trimmed", y <= 18000), ("full", np.ones_like(body))] + [(f"  body {x}", body & (apt == x)) for x in sorted(set(apt))]
    print(f"\nLIRF NM-unmatched rows from v41's chain: {lirf_nmu.sum()}")
    print(f"{'slice':16s} {'base b0':>8s} {'base b1':>8s} {'b1-b0':>7s} {'v42':>8s} {'comb':>8s} {'delta':>8s}")
    for nme, k in sl:
        g0, g1, c0, c1 = r("b0", k), r("b1", k), r("v42", k), r("new", k)
        print(f"{nme:16s} {g0:8.2f} {g1:8.2f} {g1 - g0:+7.2f} {c0:8.2f} {c1:8.2f} {c1 - c0:+8.2f}")
    gb = r("b0", body) - r("b1", body)
    gc = {n: r("v42", k) - r("new", k) for n, k in (("pooled", body), ("Jan", body & (mo == 1)), ("Jul", body & (mo == 7)))}
    print(f"\nbase-level body gain {gb:.2f}; chain-level body gain {gc['pooled']:.2f}; "
          f"transfer ratio {gc['pooled'] / gb if gb else float('nan'):.2f}")
    ok = gc["pooled"] > 0 and gc["Jan"] > 0 and gc["Jul"] > 0
    print(f"GATE (vs v42 chain, body pooled/Jan/Jul gain > 0): {'ADOPT' if ok else 'REJECT'}  {gc}")


if __name__ == "__main__":
    main()
