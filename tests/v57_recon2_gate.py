"""v57 part B gate (reports/v57_pru_recon2_preregistration.md): V* vs R0 on the Jan + Jul 2025 holdout chain.

Base: cache/v50/chain_all.parquet (v50's holdout chain), non-LIRF rows. eta calibration cross-month (Jan from Jul's
chain residuals and vice versa), tracked / untracked. V* from env: VSTAR_KIND (uniform | mult | trk | trkmult),
VSTAR_A (shrink), VSTAR_W (tracked weight).

Run:  VSTAR_KIND=mult VSTAR_A=0.75 VSTAR_W=1 python tests/v57_recon2_gate.py
"""
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from post import pru_recon2 as R  # noqa: E402
import v57_recon2_dev as DV  # noqa: E402

KIND = os.environ.get("VSTAR_KIND", "mult")
A = float(os.environ.get("VSTAR_A", "0.75"))
W = float(os.environ.get("VSTAR_W", "1"))


def holdout_frame() -> pl.DataFrame:
    ch = pl.read_parquet(ROOT / "cache/v50/chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("pred"))
    det = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier"]) for p in
                     sorted((ROOT / "cache/adsb_pushback_v4").glob("day=2025-0[17]-*.parquet"))], how="diagonal")
    ch = ch.join(det.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)), on="MVT_ID_mvt", how="left").with_columns(
        trk=pl.col("adsb_tier").is_in(list(R.TRACKED)).fill_null(False)).drop("adsb_tier")
    x = DV.frame(ch)
    return x.filter(pl.col("ym").is_in(["2025-01", "2025-07"]))


def main():
    x = holdout_frame()
    print(f"holdout non-LIRF rows {x.height:,}; tracked share {x['trk'].mean():.3f}; V* = {KIND}, a {A}, w {W}")
    y, p = x["y"].to_numpy(), x["p"].to_numpy().astype(float)
    q0, qv = p.copy(), p.copy()
    lszh = {}
    for m in ("2025-01", "2025-07"):
        mk = (x["ym"] == m).to_numpy()
        xm, idx = x.filter(pl.Series(mk)), np.flatnonzero(mk)
        dists = DV.dists_from(x.filter(~pl.Series(mk)), split=True)
        q0[idx] = R.apply(xm, R.solve_days(xm, None), 0.75)
        qv[idx] = R.apply(xm, R.solve_days(xm, dists, KIND, W), A)
        ql = R.apply(xm, R.solve_days(xm, dists, KIND, W, skip=("LIRF",)), A)        # information: LSZH under V*
        k = (xm["apt"] == "LSZH").to_numpy()
        lszh[m] = round(float(np.mean(ql[k] - xm["p"].to_numpy()[k])), 1)
    nmu, ym, apt = x["nmu"].to_numpy(), x["ym"].to_numpy(), x["apt"].to_numpy()
    body, trim, allr = ~nmu & (y <= 3600), y <= 18000, np.ones(len(y), bool)
    r = lambda q, m: float(np.sqrt(np.mean((y[m] - q[m]) ** 2)))  # noqa: E731
    print(f"base full {r(p, allr):.3f}, body {r(p, body):.3f}")
    rows = []
    for lab, m in (("full", allr), ("trimmed", trim), ("body", body), ("body Jan", body & (ym == "2025-01")),
                   ("body Jul", body & (ym == "2025-07")), ("full Jan", ym == "2025-01"), ("full Jul", ym == "2025-07")):
        rows.append({"metric": lab, "R0 - base": r(q0, m) - r(p, m), "V* - base": r(qv, m) - r(p, m), "V* - R0": r(qv, m) - r(q0, m)})
    with pl.Config(tbl_rows=20, float_precision=3):
        print(pl.DataFrame(rows))
    print("per airport body V* - R0:", {a: round(r(qv, body & (apt == a)) - r(q0, body & (apt == a)), 2) for a in sorted(set(apt))})
    print("per airport full V* - R0:", {a: round(r(qv, apt == a) - r(q0, apt == a), 2) for a in sorted(set(apt))})
    # airport-day cluster bootstrap on full (V* vs R0)
    key = (x["apt"] + "_" + x["MVT_TIME_UTC_mvt"].dt.date().cast(pl.Utf8)).to_numpy()
    ud, di = np.unique(key, return_inverse=True)
    s0, sv, nd = np.bincount(di, (y - q0) ** 2), np.bincount(di, (y - qv) ** 2), np.bincount(di)
    sb0, sbv = np.bincount(di, ((y - q0) ** 2) * body), np.bincount(di, ((y - qv) ** 2) * body)
    nb = np.bincount(di, body.astype(float))
    rng = np.random.default_rng(0)
    bs, bb = [], []
    for _ in range(2000):
        s = rng.integers(0, len(ud), len(ud))
        bs.append(np.sqrt(sv[s].sum() / nd[s].sum()) - np.sqrt(s0[s].sum() / nd[s].sum()))
        bb.append(np.sqrt(sbv[s].sum() / nb[s].sum()) - np.sqrt(sb0[s].sum() / nb[s].sum()))
    bs, bb = np.array(bs), np.array(bb)
    print(f"bootstrap (airport-day, 2000): full V*-R0 95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], "
          f"P(worse) {np.mean(bs > 0):.3f}; body 95% CI [{np.percentile(bb, 2.5):+.3f}, {np.percentile(bb, 97.5):+.3f}]")
    d = {x["metric"]: x["V* - R0"] for x in rows}
    gate = [d["body"] <= -0.10, d["body Jan"] <= 0 and d["body Jul"] <= 0, d["full"] <= 0, float(np.mean(bs > 0)) < 0.2]
    print(f"GATE (body<=-0.10, Jan&Jul<=0, full<=0, P(worse)<0.2): {gate} -> {'PASS' if all(gate) else 'FAIL'}")
    print(f"information: LSZH mean taxi change under V* (if not skipped): {lszh}")
    # 2026 projection: per (airport, month) holdout dMSE per row (V* - R0) x 2026 row counts, onto v56's board score
    rk = pl.read_parquet(ROOT / "data/ranking/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP").group_by(apt=pl.col("ADEP_mvt"), mo=pl.col("MVT_TIME_UTC_mvt").dt.month()).agg(n=pl.len())
    n26 = {(r["apt"], r["mo"]): r["n"] for r in rk.iter_rows(named=True)}
    N = sum(n26.values())
    mo = np.where(ym == "2025-01", 1, 7)
    dmse = sum(n26.get((a, m), 0) * np.mean((y[(apt == a) & (mo == m)] - qv[(apt == a) & (mo == m)]) ** 2
                                             - (y[(apt == a) & (mo == m)] - q0[(apt == a) & (mo == m)]) ** 2)
               for a in sorted(set(apt)) for m in (1, 7)) / N
    print(f"2026 projection (V* - R0, airport x month cells): board {np.sqrt(240.7425 ** 2 + dmse) - 240.7425:+.3f} s "
          f"(dMSE {dmse:+.1f} s^2 over {N:,} rows)")


if __name__ == "__main__":
    main()
