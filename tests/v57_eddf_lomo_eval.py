"""v57 part A gate read-out (information only: the EDDF build was stopped by witness W3, amendment W-A2).

Reads cache/v57/lomo_2025-{09..12}.parquet (ctl / trt). Primary population P: EDDF rows on OOF ADS-B days >= 2025-09-16
whose airport-day OPDI ground-event share >= 0.2. LSZH positive control on the same months (the trt arm was run for
Sep-Dec only, not all 10 months as the pre-registration's condition 5 says).

Run:  python tests/v57_eddf_lomo_eval.py
"""
import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def main():
    lo = pl.concat([pl.read_parquet(ROOT / f"cache/v57/lomo_2025-{m}.parquet") for m in ("09", "10", "11", "12")])
    of = pl.read_parquet(ROOT / "cache/opdi_feats/opdi_feats.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "o_lead_first")
    lo = lo.join(of, on="MVT_ID_mvt", how="left").with_columns(cov=(pl.col("o_lead_first").fill_null(0) >= 60))
    lo = lo.with_columns(share=pl.col("cov").mean().over("ADEP_mvt", "day"))
    y = lambda d: d["taxi"].to_numpy()  # noqa: E731
    r = lambda d, c: float(np.sqrt(np.mean((y(d) - d[c].to_numpy()) ** 2))) if d.height else np.nan  # noqa: E731
    for apt in ("EDDF", "LSZH"):
        P = lo.filter((pl.col("ADEP_mvt") == apt) & (pl.col("day") >= pl.date(2025, 9, 16)) & (pl.col("share") >= 0.2))
        body = P.filter(~pl.col("nmu") & (pl.col("taxi") <= 3600))
        print(f"\n{apt}: P rows {P.height:,} on {P['day'].n_unique()} days, covered share {P['cov'].mean():.3f}")
        for lab, d in (("full", P), ("trimmed", P.filter(pl.col("taxi") <= 18000)), ("body", body),
                       ("body covered", body.filter(pl.col("cov"))), ("body uncovered", body.filter(~pl.col("cov"))),
                       ("body combiner", body.filter(pl.col("served") == "comb")), ("body stage b", body.filter(pl.col("served") == "b"))):
            print(f"  {lab:15s} n {d.height:6,}  ctl {r(d, 'ctl'):7.2f}  trt {r(d, 'trt'):7.2f}  delta {r(d, 'trt') - r(d, 'ctl'):+6.2f}")
        print("  body delta by month:", {m: round(r(g, 'trt') - r(g, 'ctl'), 2) for (m,), g in sorted(body.group_by("ym"))})
        days = body["day"].to_numpy()
        ud, di = np.unique(days, return_inverse=True)
        e0, e1 = (y(body) - body["ctl"].to_numpy()) ** 2, (y(body) - body["trt"].to_numpy()) ** 2
        s0, s1, n = np.bincount(di, e0), np.bincount(di, e1), np.bincount(di)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            s = rng.integers(0, len(ud), len(ud))
            bs.append(np.sqrt(s1[s].sum() / n[s].sum()) - np.sqrt(s0[s].sum() / n[s].sum()))
        print(f"  body delta day-cluster bootstrap 95% CI [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
        g = (e0 - e1)
        top = np.sort(g)[::-1][:20].sum() / max(g.sum(), 1e-9)
        print(f"  top-20 rows share of the body SSE gain {top:.2f}")


if __name__ == "__main__":
    main()
