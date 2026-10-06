"""Gate of reports/base_comb_preregistration.md: v41's recipe on the lgb_comb_nb base vs v41's chain.

Steps (run in order; each is its own process because the corrector blocks are read from the environment
at import time):
  corr 0    ctxcorr2r_comb   (blocks 1-7)        holdout + OOF corrector outputs on the new base
  corr 1    ctxcorr2rb_comb  (blocks 1-7 + 9)
  chain     ctxsel_comb = block-9 corrector at EHAM/LSZH/EDDF/LEBL, the other elsewhere; v41's chain on it;
            compared with v41's saved chain (cache/eval/v41_chain_holdout.parquet), plus the base-level gain
            and the base->chain transfer ratio.

Run:  PRC_BASE=lgb_comb_nb .venv/Scripts/python.exe -u tests/base_comb_test.py corr 0
      PRC_BASE=lgb_comb_nb .venv/Scripts/python.exe -u tests/base_comb_test.py corr 1
      PRC_BASE=lgb_comb_nb PRC_LGB=ctxcorr2r .venv/Scripts/python.exe -u tests/base_comb_test.py chain
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ["CTX_ENS"] = "2"
os.environ["CTX_RWY"] = "1"
if len(sys.argv) > 2 and sys.argv[1] == "corr":
    os.environ["CTX_ADSB"] = sys.argv[2]
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL, OOF = ROOT / "cache" / "eval", ROOT / "cache" / "oof"
SEL = "ctxsel_comb"


def corr() -> None:
    assert os.environ.get("PRC_BASE") == "lgb_comb_nb"
    import oof_context_corrector_stage2 as S2
    print(f"building {S2.NAME}", flush=True)
    if not S2.EV.exists():
        S2.build_inputs()


def chain() -> None:
    assert os.environ.get("PRC_BASE") == "lgb_comb_nb" and os.environ.get("PRC_LGB") == "ctxcorr2r"
    import ctx_adsb_stable_test as T
    apt = T._airports()
    a, b = "ctxcorr2r_comb", "ctxcorr2rb_comb"
    ev, n = T._select(pl.read_parquet(EVAL / f"{a}_holdout_ev.parquet"), pl.read_parquet(EVAL / f"{b}_holdout_ev.parquet"), "pred", apt)
    ev.write_parquet(EVAL / f"{SEL}_holdout_ev.parquet")
    (OOF / SEL).mkdir(parents=True, exist_ok=True)
    for f in sorted(p.name for p in (OOF / a).glob("fold=*.parquet")):
        o, _ = T._select(pl.read_parquet(OOF / a / f), pl.read_parquet(OOF / b / f), "pred", apt)
        o.write_parquet(OOF / SEL / f)
    print(f"{SEL}: holdout {n:,} rows from {b}", flush=True)
    import oof_context_corrector_stage2 as S2
    S2.EV, S2.OOFD = EVAL / f"{SEL}_holdout_ev.parquet", OOF / SEL
    S2.main()                                   # chain on the new corrector; writes ctxcorr_chain_holdout.parquet
    new = pl.read_parquet(EVAL / "ctxcorr_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), v42="new")
    v41 = pl.read_parquet(EVAL / "v41_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), v41="new")
    base = (pl.read_parquet(EVAL / "lgb_mixed_nb_holdout_ev.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), b0="pred")
            .join(pl.read_parquet(EVAL / "lgb_comb_nb_holdout_ev.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), b1="pred"), on="MVT_ID_mvt"))
    raw = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{m}-01_2025-{n}-01.parquet"))
                     .filter(pl.col("PHASE_mvt") == "DEP")
                     .select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                             nm=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), apt=pl.col("ADEP_mvt"))
                     for m, n in (("01", "02"), ("07", "08"))]).collect()
    d = new.join(v41, on="MVT_ID_mvt").join(base, on="MVT_ID_mvt").join(raw, on="MVT_ID_mvt")
    assert d.height == new.height
    y, mo, apt = d["y"].to_numpy(), d["mo"].to_numpy(), d["apt"].to_numpy()
    body = (~d["nm"].to_numpy()) & (y <= 3600)
    r = lambda c, k: float(np.sqrt(np.mean((y - d[c].to_numpy())[k] ** 2)))
    sl = [("body pooled", body), ("body Jan", body & (mo == 1)), ("body Jul", body & (mo == 7)),
          ("trimmed", y <= 18000), ("full", np.ones_like(body))] + [(f"  body {x}", body & (apt == x)) for x in sorted(set(apt))]
    print(f"\n{'slice':16s} {'base b0':>8s} {'base b1':>8s} {'b1-b0':>7s} {'v41':>8s} {'v42':>8s} {'v42-v41':>8s}")
    for nme, k in sl:
        g0, g1, c0, c1 = r("b0", k), r("b1", k), r("v41", k), r("v42", k)
        print(f"{nme:16s} {g0:8.2f} {g1:8.2f} {g1 - g0:+7.2f} {c0:8.2f} {c1:8.2f} {c1 - c0:+8.2f}")
    gb = r("b0", body) - r("b1", body)
    gc = {n: r("v41", k) - r("v42", k) for n, k in (("pooled", body), ("Jan", body & (mo == 1)), ("Jul", body & (mo == 7)))}
    print(f"\nbase-level body gain {gb:.2f}; chain-level body gain {gc['pooled']:.2f}; "
          f"transfer ratio {gc['pooled'] / gb if gb else float('nan'):.2f}")
    ok = gc["pooled"] > 0 and gc["Jan"] > 0 and gc["Jul"] > 0
    print(f"GATE (vs v41 chain, pooled/Jan/Jul > 0): {'ADOPT' if ok else 'REJECT'}")


if __name__ == "__main__":
    {"corr": corr, "chain": chain}[sys.argv[1]]()
