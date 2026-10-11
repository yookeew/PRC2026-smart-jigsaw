"""Gate for the hour-window corrector block (reports/ctx_hour_preregistration.md).

1. Holdout chain (tests/v50_chain_gate.py ARMS=all): cache/v64/gate_roh/chain_all vs cache/v64/gate_ro/chain_all.
2. OOF months, corrector level: cache/oof/ctxc1sel_<TAG> vs cache/oof/ctxc1sel_ro (TAG env, default roh).

Run:  .venv/Scripts/python.exe tests/ctx_hour_gate.py
"""
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TRIM = 18000
TAG = os.environ.get("TAG", "roh")
CTL = os.environ.get("CTL", "ro")                      # control corrector tag (v65 gate: rohb)
GDIR = os.environ.get("GATE_DIR", "cache/v64")           # where gate_<TAG>/chain_all lives (control always cache/v64)
CTL_DIR = os.environ.get("CTL_DIR", "cache/v64")         # where gate_<CTL>/chain_all lives (v66 gate: cache/v65)
BOARD = float(os.environ.get("BOARD", "238.3173"))       # current best, for the 2026 projection


def labels() -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt"]
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))],
                    how="vertical_relaxed")
    return raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("TAXITIME_SEC_mvt").is_not_null()).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), apt="ADEP_mvt", y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
        nm=pl.col("FLIGHT_ID_mvt").is_null(), mo=pl.col("MVT_TIME_UTC_mvt").dt.month())


def rd(path, name):
    return pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("pred").cast(pl.Float64).alias(name))


def main() -> None:
    lab = labels()
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = dict(rk.group_by("ADEP_mvt").agg(n=pl.len()).iter_rows())
    rm = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")  # noqa: E731

    # 1. holdout chain
    x = (rd(ROOT / f"{CTL_DIR}/gate_{CTL}/chain_all.parquet", "c").join(rd(ROOT / f"{GDIR}/gate_{TAG}/chain_all.parquet", "t"), on="MVT_ID_mvt")
         .join(lab, on="MVT_ID_mvt", how="inner"))
    y, c, t = x["y"].to_numpy(), x["c"].to_numpy(), x["t"].to_numpy()
    apt, mo, nm = x["apt"].to_numpy(), x["mo"].to_numpy(), x["nm"].to_numpy()
    trim, body = y <= TRIM, (~nm) & (y <= 3600)
    d = lambda m: rm((y - t)[m]) - rm((y - c)[m])  # noqa: E731
    print(f"holdout chain rows {len(y):,}; RMS change {rm(t - c):.2f} s, mean {np.mean(t - c):+.2f}")
    res = {}
    for nmx, m in [("trimmed", trim), ("full", np.ones_like(trim)), ("body", body),
                   ("trimmed Jan", trim & (mo == 1)), ("trimmed Jul", trim & (mo == 7)),
                   ("body Jan", body & (mo == 1)), ("body Jul", body & (mo == 7)),
                   ("trimmed non-LIRF", trim & (apt != "LIRF")), ("full non-LIRF", apt != "LIRF"), ("body non-LIRF", body & (apt != "LIRF"))]:
        res[nmx] = d(m)
        print(f"  {nmx:18s} control {rm((y - c)[m]):8.2f}  treatment {rm((y - t)[m]):8.2f}  delta {res[nmx]:+.3f}")
    print("  per airport trimmed:", {a: round(d(trim & (apt == a)), 2) for a in sorted(set(apt))})
    print("  per airport body:   ", {a: round(d(body & (apt == a)), 2) for a in sorted(set(apt))})
    tot = sum(n26.values())
    for lab_, keep in [("all airports", lambda a: True), ("LIRF frozen", lambda a: a != "LIRF"), ("production rows (no LIRF, no LSZH)", lambda a: a not in ("LIRF", "LSZH"))]:
        dm = sum(n26[a] * np.mean((y - t)[apt == a] ** 2 - (y - c)[apt == a] ** 2) for a in set(apt) if keep(a) and a in n26) / tot
        print(f"  2026 projection ({lab_}): {np.sqrt(BOARD ** 2 + dm) - BOARD:+.3f} s")

    # 2. OOF months, corrector level
    o = (pl.concat([rd(f, "c") for f in sorted((ROOT / f"cache/oof/ctxc1sel_{CTL}").glob("fold=*.parquet"))])
         .join(pl.concat([rd(f, "t") for f in sorted((ROOT / f"cache/oof/ctxc1sel_{TAG}").glob("fold=*.parquet"))]), on="MVT_ID_mvt")
         .join(lab, on="MVT_ID_mvt", how="inner"))
    yo, co, to, moo, apo = (o[k].to_numpy() for k in ("y", "c", "t", "mo", "apt"))
    tr = yo <= TRIM
    om = {m: rm((yo - to)[tr & (moo == m)]) - rm((yo - co)[tr & (moo == m)]) for m in sorted(set(moo))}
    print(f"OOF rows {len(yo):,}; trimmed pooled delta {rm((yo - to)[tr]) - rm((yo - co)[tr]):+.3f}")
    print("  trimmed by month:", {m: round(v, 2) for m, v in om.items()})
    print("  per airport trimmed:", {a: round(rm((yo - to)[tr & (apo == a)]) - rm((yo - co)[tr & (apo == a)]), 2) for a in sorted(set(apo))})
    better = sum(v < 0 for v in om.values())

    ok = {"a trimmed <= -0.10": res["trimmed"] <= -0.10, "b Jan and Jul trimmed better": res["trimmed Jan"] < 0 and res["trimmed Jul"] < 0,
          "c full <= 0": res["full"] <= 0, "d body <= +0.10": res["body"] <= 0.10,
          "e OOF Feb, Jun better, >= 7/10 months": om[2] < 0 and om[6] < 0 and better >= 7}
    print("conditions:", ok)
    print("DECISION:", "PASS" if all(ok.values()) else "FAIL")


if __name__ == "__main__":
    sys.exit(main())
