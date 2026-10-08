"""v42's chain on the Jan+Jul 2025 holdout, as a baseline for later gates (PROGRESS.md §97).

The Arm C corrector (ctxc1sel, teammate's build: cache/eval/ctxc1sel_holdout_ev.parquet, cache/oof/ctxc1sel/)
through the stage-2 harness of tests/oof_context_corrector_stage2.py (stack, ADS-B blend, combiner, neighbour
state, LOBT window, NM lines fitted on the 10 training months). As in v42's build (src/post/armc_build.py),
LIRF NM-unmatched rows take v41's chain value (cache/eval/v41_chain_holdout.parquet).
Writes cache/eval/v42_chain_holdout.parquet (MVT_ID_mvt, v42).

Run:  PRC_LGB=ctxcorr2r .venv/Scripts/python.exe -u tests/v42_holdout_baseline.py
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


def main() -> None:
    assert os.environ.get("PRC_LGB") == "ctxcorr2r"
    import oof_context_corrector_stage2 as S2
    S2.EV, S2.OOFD = EVAL / "ctxc1sel_holdout_ev.parquet", OOF / "ctxc1sel"
    S2.main()
    new = pl.read_parquet(EVAL / "ctxcorr_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), chain="new")
    v41 = pl.read_parquet(EVAL / "v41_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), v41="new")
    raw = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet"))
                     .filter(pl.col("PHASE_mvt") == "DEP")
                     .select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                             nm=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                             apt=pl.col("ADEP_mvt"))
                     for a, b in (("01", "02"), ("07", "08"))]).collect()
    d = new.join(v41, on="MVT_ID_mvt").join(raw, on="MVT_ID_mvt")
    assert d.height == new.height
    lirf_nmu = ((d["apt"] == "LIRF") & d["nm"]).to_numpy()
    v42 = np.where(lirf_nmu, d["v41"].to_numpy(), d["chain"].to_numpy())
    d = d.with_columns(v42=pl.Series(v42))
    d.select("MVT_ID_mvt", "v42").write_parquet(EVAL / "v42_chain_holdout.parquet")
    y, nm, apt, mo = d["y"].to_numpy(), d["nm"].to_numpy(), d["apt"].to_numpy(), d["mo"].to_numpy()
    body = (~nm) & (y <= 3600)
    r = lambda c, k: float(np.sqrt(np.mean((y - d[c].to_numpy())[k] ** 2)))
    print(f"\nLIRF NM-unmatched rows copied from v41: {lirf_nmu.sum()}")
    print(f"{'slice':28s} {'v41':>8s} {'v42':>8s} {'delta':>8s}")
    nmu_nl = nm & (apt != "LIRF")
    for n, k in [("body pooled", body), ("body Jan", body & (mo == 1)), ("body Jul", body & (mo == 7)),
                 ("trimmed", y <= 18000), ("full", np.ones_like(body)),
                 ("NM-unmatched non-LIRF", nmu_nl), ("NM-unmatched non-LIRF, y<=5h", nmu_nl & (y <= 18000))]:
        print(f"{n:28s} {r('v41', k):8.2f} {r('v42', k):8.2f} {r('v42', k) - r('v41', k):+8.2f}")
    print("wrote v42_chain_holdout.parquet")


if __name__ == "__main__":
    main()
