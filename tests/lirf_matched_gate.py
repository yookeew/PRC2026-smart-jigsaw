"""Gate for LIRF NM-matched rows from the hour-block chain (reports/lirf_matched_unfreeze_preregistration.md).

Run:  .venv/Scripts/python.exe tests/lirf_matched_gate.py
"""
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import ctx_hour_gate as G  # noqa: E402

TRIM = 18000


def main() -> None:
    lab = G.labels()
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
    ctl = pl.read_parquet(ROOT / "cache/eval/v42_chain_holdout.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), c=pl.col("v42").cast(pl.Float64))
    x = ctl.join(G.rd(ROOT / "cache/v64/gate_roh/chain_all.parquet", "t"), on="MVT_ID_mvt").join(lab, on="MVT_ID_mvt", how="inner")
    k = ((x["apt"] == "LIRF") & ~x["nm"]).to_numpy()
    y, c, t0, mo, nm = (x[q].to_numpy() for q in ("y", "c", "t", "mo", "nm"))
    t = np.where(k, t0, c)                      # only LIRF NM-matched rows change
    trim, body = y <= TRIM, (~nm) & (y <= 3600)
    d = lambda m: rm((y - t)[m]) - rm((y - c)[m])  # noqa: E731
    res = {n: d(m) for n, m in [("trimmed", trim), ("full", np.ones_like(trim)), ("body", body),
                                ("trimmed Jan", trim & (mo == 1)), ("trimmed Jul", trim & (mo == 7))]}
    print(f"LIRF NM-matched holdout rows {k.sum():,}; RMS change {rm((t0 - c)[k]):.1f}, mean {np.mean((t0 - c)[k]):+.1f}")
    print(f"  on LIRF NM-matched rows: trimmed {rm((y - c)[k & trim]):.2f} -> {rm((y - t0)[k & trim]):.2f}; body "
          f"{rm((y - c)[k & body]):.2f} -> {rm((y - t0)[k & body]):.2f}; full {rm((y - c)[k]):.2f} -> {rm((y - t0)[k]):.2f}")
    for n, v in res.items():
        print(f"  pooled {n:12s} {v:+.3f}")
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "FLIGHT_ID_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    n26 = rk.height
    n26_k = rk.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).height  # placeholder total; LIRF count below
    rkl = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt"]).filter(
        (pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_not_null()).height
    dm = rkl * np.mean((y - t0)[k] ** 2 - (y - c)[k] ** 2) / n26
    print(f"  2026 projection (LIRF NM-matched share {rkl / n26:.3f}): {np.sqrt(238.3173 ** 2 + dm) - 238.3173:+.3f} s")
    del n26_k

    o = (pl.concat([G.rd(f, "c") for f in sorted(glob.glob(str(ROOT / "cache/oof/ctxc1sel/*.parquet")))])
         .join(pl.concat([G.rd(f, "t") for f in sorted(glob.glob(str(ROOT / "cache/oof/ctxc1sel_roh/*.parquet")))]), on="MVT_ID_mvt")
         .join(lab, on="MVT_ID_mvt", how="inner").filter((pl.col("apt") == "LIRF") & ~pl.col("nm") & (pl.col("y") <= TRIM)))
    om = {int(m): rm((g["y"] - g["t"]).to_numpy()) - rm((g["y"] - g["c"]).to_numpy()) for (m,), g in o.group_by("mo")}
    om = dict(sorted(om.items()))
    print("OOF LIRF NM-matched trimmed by month (ctxc1sel_roh - ctxc1sel):", {m: round(v, 2) for m, v in om.items()})
    ok = {"a": res["trimmed"] <= -0.10, "b": res["trimmed Jan"] < 0 and res["trimmed Jul"] < 0, "c": res["full"] <= 0,
          "d": res["body"] <= 0, "e": om.get(2, 1) < 0 and om.get(6, 1) < 0 and sum(v < 0 for v in om.values()) >= 7}
    print("conditions:", ok, "->", "PASS" if all(ok.values()) else "FAIL")


if __name__ == "__main__":
    main()
