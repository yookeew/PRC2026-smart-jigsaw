"""reports/ctx_member_avg_preregistration.md: equal-weight mean of the v37 (ctxcorr2r) and v40 (ctxcorr2rb)
correctors, through v37's chain, vs v37's chain.

Builds  cache/eval/ctxavg_holdout_ev.parquet, cache/oof/ctxavg/fold=*.parquet, data/submissions/ctxavg.parquet
(each the mean of the two members), then runs the stage-2 comparison of oof_context_corrector_stage2.py.

Run:  PRC_LGB=ctxcorr2r .venv/Scripts/python.exe tests/ctx_member_avg_test.py
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

MEMBERS = ("ctxcorr2r", "ctxcorr2rb")
NAME = "ctxavg"
EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"


def _mean(frames: list[pl.DataFrame], col: str) -> pl.DataFrame:
    """First frame with `col` replaced by the mean over all frames (joined on MVT_ID_mvt)."""
    base = frames[0].with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
    vals = base.select("MVT_ID_mvt")
    for i, f in enumerate(frames):
        vals = vals.join(f.select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(col).cast(pl.Float64).alias(f"m{i}")),
                         on="MVT_ID_mvt", how="left")
    assert vals.height == base.height and all(vals[f"m{i}"].null_count() == 0 for i in range(len(frames)))
    m = vals.select(pl.mean_horizontal([f"m{i}" for i in range(len(frames))]))[:, 0]
    rms = float(np.sqrt(np.mean((vals["m0"] - vals["m1"]).to_numpy() ** 2)))
    return base.with_columns(m.cast(base.schema[col]).alias(col) if base.schema[col] != pl.Int32
                             else m.round().cast(pl.Int32).alias(col)), rms


def build() -> None:
    ev, rms = _mean([pl.read_parquet(EVAL / f"{n}_holdout_ev.parquet") for n in MEMBERS], "pred")
    ev.write_parquet(EVAL / f"{NAME}_holdout_ev.parquet")
    print(f"holdout: members differ by {rms:.1f} s RMS")
    (OOF / NAME).mkdir(parents=True, exist_ok=True)
    folds = sorted(p.name for p in (OOF / MEMBERS[0]).glob("fold=*.parquet"))
    assert folds == sorted(p.name for p in (OOF / MEMBERS[1]).glob("fold=*.parquet")) and len(folds) == 5
    for f in folds:
        o, rms = _mean([pl.read_parquet(OOF / n / f) for n in MEMBERS], "pred")
        o.write_parquet(OOF / NAME / f)
        print(f"oof {f}: members differ by {rms:.1f} s RMS")
    rk, rms = _mean([pl.read_parquet(SUB / f"{n}.parquet") for n in MEMBERS], "TAXITIME_SEC_mvt")
    rk.write_parquet(SUB / f"{NAME}.parquet")
    print(f"ranking: members differ by {rms:.1f} s RMS")


def main() -> None:
    assert os.environ.get("PRC_LGB") == "ctxcorr2r", "the base must be v37's chain: PRC_LGB=ctxcorr2r"
    build()
    import oof_context_corrector_stage2 as S2
    S2.EV, S2.OOFD = EVAL / f"{NAME}_holdout_ev.parquet", OOF / NAME
    S2.main()
    both_members()


def both_members() -> None:
    """Amended gate: the average must beat v37's chain AND v40's chain on holdout body."""
    avg = pl.read_parquet(EVAL / "ctxcorr_chain_holdout.parquet").rename({"base": "v37", "new": "avg"})
    v40 = pl.read_parquet(EVAL / "v40_chain_holdout.parquet").select("MVT_ID_mvt", v40="new",
                                                                     v37_check="base")
    raw = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / f"training_2025-{a}-01_2025-{b}-01.parquet"))
                     .filter(pl.col("PHASE_mvt") == "DEP")
                     .select(pl.col("MVT_ID_mvt").cast(pl.Int64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
                             nm=pl.col("FLIGHT_ID_mvt").is_null(), m=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                             apt=pl.col("ADEP_mvt"))
                     for a, b in (("01", "02"), ("07", "08"))]).collect()
    d = (avg.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
         .join(v40.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)), on="MVT_ID_mvt")
         .join(raw, on="MVT_ID_mvt"))
    assert d.height == avg.height
    assert np.allclose(d["v37"].to_numpy(), d["v37_check"].to_numpy()), "v37 chain differs between the two runs"
    y, m, apt = d["y"].to_numpy(), d["m"].to_numpy(), d["apt"].to_numpy()
    body = (~d["nm"].to_numpy()) & (y <= 3600)
    r = lambda c, k: float(np.sqrt(np.mean((y - d[c].to_numpy())[k] ** 2)))
    sl = [("body pooled", body), ("body Jan", body & (m == 1)), ("body Jul", body & (m == 7)),
          ("trimmed", y <= 18000), ("full", np.ones_like(body))] + [(f"  body {a}", body & (apt == a)) for a in sorted(set(apt))]
    print(f"\n{'slice':16s} {'v37':>8s} {'v40':>8s} {'avg':>8s} {'avg-v37':>8s} {'avg-v40':>8s}")
    for n, k in sl:
        a37, a40, av = r("v37", k), r("v40", k), r("avg", k)
        print(f"{n:16s} {a37:8.2f} {a40:8.2f} {av:8.2f} {av - a37:+8.2f} {av - a40:+8.2f}")
    ok = all(r(ref, k) - r("avg", k) > 0 for ref in ("v37", "v40") for k in (body, body & (m == 1), body & (m == 7)))
    print(f"\nAMENDED GATE (beats both members, pooled/Jan/Jul): {'ADOPT' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
