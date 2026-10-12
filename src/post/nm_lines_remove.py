"""Remove the NM-line override at non-LIRF airports (reports/lirf_count_dates_preregistration.md, amendment A1).

src/post/nm_lines.py sets NM-unmatched departures > 6 h late to a per-airport line in d = T - S. Outside LIRF the line
is far worse than the chain it overrides (2025: SSE 3.21e8 vs 8.51e7 on 144 rows). For those rows:
  new = input + (chain before the line - chain after the line)
so only the line's own contribution is removed; every other adjustment the row carries is kept. LIRF rows untouched.

Run:  .venv/Scripts/python.exe src/post/nm_lines_remove.py IN.parquet OUT.parquet CHAIN_WIN.parquet CHAIN_FINAL.parquet
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scope  # noqa: E402

LINE_MIN_D = 6 * 3600


def main(in_f, out_f, win_f, fin_f):
    tpl = pl.read_parquet(scope.TEMPLATE)
    rk = (pl.read_parquet(scope.RANKING).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col("MVT_ID_mvt").cast(pl.Float64), "ADEP_mvt", "FLIGHT_mvt", nmu=pl.col("FLIGHT_ID_mvt").is_null(),
                  d=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64)))
    rd = lambda f, c: pl.read_parquet(f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(c))  # noqa: E731
    x = (tpl.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(rk, on="MVT_ID_mvt", how="left")
         .join(rd(in_f, "p"), on="MVT_ID_mvt", how="left").join(rd(win_f, "win"), on="MVT_ID_mvt", how="left")
         .join(rd(fin_f, "fin"), on="MVT_ID_mvt", how="left"))
    assert x.height == tpl.height and all(x[c].null_count() == 0 for c in ("p", "win", "fin", "ADEP_mvt"))
    k = ((x["ADEP_mvt"] != "LIRF") & x["nmu"] & (x["d"] > LINE_MIN_D)).to_numpy()
    p, win, fin = x["p"].to_numpy(), x["win"].to_numpy(), x["fin"].to_numpy()
    other = ~k & (np.abs(fin - win) > 0.5) & (x["ADEP_mvt"] != "LIRF").to_numpy()
    print(f"rows in scope {k.sum()}; non-LIRF rows outside scope where the line stage moved the chain: {other.sum()}")
    q = np.where(k, p + (win - fin), p)
    d = q - p
    assert np.all((q[k] >= 200) & (q[k] <= 4000)), "stop rule: new values outside [200, 4000]"
    apt = x["ADEP_mvt"].to_numpy()
    for a in sorted(set(apt[k])):
        m = k & (apt == a)
        print(f"  {a}: rows {m.sum():3d}, mean before {p[m].mean():7.1f}, after {q[m].mean():7.1f}, mean change {d[m].mean():+8.1f}")
    print(f"rows changed {(np.abs(d) > 0.5).sum()}, sum d^2 {np.sum(d ** 2):.3g}, RMS over all rows {np.sqrt(np.mean(d ** 2)):.2f}")
    out = pl.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"], "TAXITIME_SEC_mvt": np.round(q).astype(np.int32)})
    assert out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(*sys.argv[1:5])
