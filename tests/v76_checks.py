"""Checks behind v76 and the final blind file (reports/twin_build_average_preregistration.md, PROGRESS.md §163).

v76 = v75 averaged with the four-month build's Jan / Jul rows (src/post/twin_average.py); the final blind file
(final4m/smart-jigsaw_final.parquet = blind_v8) = Jan / Jul from v76, Feb / Jun from f4m_v75. Prints:
  A  input / output md5s and lineage (v75 is the board-scored file; f4m_v75 is the build behind blind_v7's Feb / Jun);
  B  the same MVT_ID is the same movement in ranking.parquet and final_ranking.parquet (identity columns; T dither);
  C  an independent pandas recompute of v76;
  D  format of v76 and of the final file against their templates;
  E  label-free proxies on the scope rows (NM-matched, non-LIRF), mid frame (T averaged over the two releases):
     T - AOBT_3 (all / ADS-B tracked with identical detections / untracked / detections differ; airport-day clustered
     standard error and bootstrap) and T - ADS-B pushback (identical detections only);
  F  the slope of (four-month - live) on (T_final - T_live), i.e. the release frame effect.

Run:  .venv/Scripts/python.exe tests/v76_checks.py   (about 2 min; log in logs/v76/v76_checks.log)
"""
import glob
import hashlib
import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
S, RK = ROOT / "data" / "submissions", ROOT / "data" / "ranking"
F64 = pl.col("MVT_ID_mvt").cast(pl.Float64)
GOOD = ["dwell", "appear"]
EXPECT = {"smart-jigsaw_v75.parquet": "027b245e7a54c99dc753040fb8317f4c", "smart-jigsaw_v76.parquet": "9b818fb96911e546ebc591a81830bec8",
          "final4m/f4m_v75.parquet": "af5dcb99e817a08f1834fae4996545b2", "final4m/blind_v7.parquet": "6ce6152ca956b5b271235de937496969",
          "final4m/blind_v8.parquet": "08adc51280d01f1521e27e72f06ce21c", "final4m/smart-jigsaw_final.parquet": "08adc51280d01f1521e27e72f06ce21c"}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def rd(f, n):
    return pl.read_parquet(f).select(F64, pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(n))


def tsec(src, n):
    return (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
            .select(F64, pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms").cast(pl.Float64).truediv(1000).alias(n)).collect())


def dets(pat, sfx):
    d = pl.concat([pl.read_parquet(f).select(F64, "adsb_tier", "adsb_pushback_ts") for f in sorted(glob.glob(str(pat)))]).unique("MVT_ID_mvt")
    ts = d["adsb_pushback_ts"]
    pb = ts.dt.epoch("ms").cast(pl.Float64) / 1000 if str(ts.dtype).startswith("Datetime") else ts.cast(pl.Float64)
    return d.with_columns(pb=pb).select("MVT_ID_mvt", pl.col("adsb_tier").alias("tier" + sfx), pl.col("pb").alias("pb" + sfx))


def gain(g, z):
    """mean and row-level se of MSE(v75) - MSE(average) and MSE(v75) - MSE(four-month) against proxy z, 60-3,600 s."""
    p, q, a, zz = (g[c].to_numpy() for c in ("p", "q", "a", z))
    k = (zz >= 60) & (zz <= 3600)
    ga = (p[k] - zz[k]) ** 2 - (a[k] - zz[k]) ** 2
    gq = (p[k] - zz[k]) ** 2 - (q[k] - zz[k]) ** 2
    return (f"n {k.sum():7,d}  MSE(v75)-MSE(avg) {ga.mean():7.1f} +- {ga.std() / np.sqrt(k.sum()):5.1f}   "
            f"MSE(v75)-MSE(4m) {gq.mean():7.1f} +- {gq.std() / np.sqrt(k.sum()):5.1f}")


def main():
    print("A. md5 and lineage")
    for f, h in EXPECT.items():
        got = md5(S / f)
        print(f"   {got} {'ok' if got == h else 'MISMATCH (expected ' + h + ')'}  {f}")
        assert got == h, f
    b7, b8 = rd(S / "final4m/blind_v7.parquet", "b7"), rd(S / "final4m/smart-jigsaw_final.parquet", "b8")
    f4, v75, v76 = rd(S / "final4m/f4m_v75.parquet", "q"), rd(S / "smart-jigsaw_v75.parquet", "p"), rd(S / "smart-jigsaw_v76.parquet", "a")
    m = b7.join(b8, on="MVT_ID_mvt").join(f4, on="MVT_ID_mvt").join(v75, on="MVT_ID_mvt", how="left").join(v76, on="MVT_ID_mvt", how="left")
    jj = m["p"].is_not_null()
    print(f"   blind_v7: Feb/Jun == f4m_v75 {bool((m.filter(~jj)['b7'] == m.filter(~jj)['q']).all())}, Jan/Jul == v75 {bool((m.filter(jj)['b7'] == m.filter(jj)['p']).all())}")
    print(f"   final:    Feb/Jun == f4m_v75 {bool((m.filter(~jj)['b8'] == m.filter(~jj)['q']).all())}, Jan/Jul == v76 {bool((m.filter(jj)['b8'] == m.filter(jj)['a']).all())}"
          f"  (rows Jan/Jul {int(jj.sum()):,}, Feb/Jun {int((~jj).sum()):,})")

    print("B. same MVT_ID = same movement in both releases (live departures)")
    cols = ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
            "LOBT_flt", "AOBT_3_flt", "CALLSIGN_flt"]
    a_ = pl.scan_parquet(RK / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(F64, *cols).collect()
    b_ = pl.scan_parquet(RK / "final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(F64, *cols).collect()
    j = a_.join(b_, on="MVT_ID_mvt", how="left", suffix="_f")
    print(f"   live departures {a_.height:,}, found in the final release {j['ADEP_mvt_f'].is_not_null().sum():,}")
    for c in cols:
        same = (j[c] == j[c + "_f"]).fill_null(j[c].is_null() & j[c + "_f"].is_null())
        print(f"   {c:20s} identical on {same.sum():,} of {j.height:,}")

    print("C. independent recompute of v76 (pandas)")
    rk = pd.read_parquet(RK / "ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "LOBT_flt", "MVT_TIME_UTC_mvt"])
    rk = rk[rk.PHASE_mvt == "DEP"].astype({"MVT_ID_mvt": "float64"})
    lv = pd.read_parquet(S / "smart-jigsaw_v75.parquet").astype({"MVT_ID_mvt": "float64"})
    bq = pd.read_parquet(S / "final4m/f4m_v75.parquet").astype({"MVT_ID_mvt": "float64"}).rename(columns={"TAXITIME_SEC_mvt": "q"})
    z = lv.merge(rk, on="MVT_ID_mvt", how="left").merge(bq, on="MVT_ID_mvt", how="left")
    sc = (z.ADEP_mvt != "LIRF") & z.FLIGHT_ID_mvt.notna()
    lt = (z.MVT_TIME_UTC_mvt - z.LOBT_flt).dt.total_seconds()
    lo, hi = np.clip(lt - 3606, 0, None).fillna(-np.inf), (lt + 3606).fillna(np.inf)
    mine = np.where(sc, np.rint(np.clip((z.TAXITIME_SEC_mvt.astype(float) + z.q.astype(float)) / 2, lo, hi)), z.TAXITIME_SEC_mvt.astype(float))
    got = pd.read_parquet(S / "smart-jigsaw_v76.parquet")
    print(f"   recompute == v76: {np.array_equal(mine, got.TAXITIME_SEC_mvt.to_numpy().astype(float))}; scope rows {int(sc.sum()):,}; "
          f"changed vs v75 {int((got.TAXITIME_SEC_mvt != lv.TAXITIME_SEC_mvt).sum()):,} (all in scope: {bool(((got.TAXITIME_SEC_mvt != lv.TAXITIME_SEC_mvt) <= sc).all())})")

    print("D. format")
    for f, t in (("smart-jigsaw_v76.parquet", "submitting.parquet"), ("final4m/smart-jigsaw_final.parquet", "final_submitting.parquet")):
        x, tp = pd.read_parquet(S / f), pd.read_parquet(RK / t)
        print(f"   {f}: columns {list(x.columns) == list(tp.columns)}, row order == {t} {bool((x.MVT_ID_mvt.to_numpy() == tp.MVT_ID_mvt.to_numpy()).all())}, "
              f"rows {len(x):,}, nulls {int(x.TAXITIME_SEC_mvt.isna().sum())}, dups {int(x.MVT_ID_mvt.duplicated().sum())}, "
              f"dtype {x.TAXITIME_SEC_mvt.dtype}, range [{x.TAXITIME_SEC_mvt.min()}, {x.TAXITIME_SEC_mvt.max()}]")

    print("E. label-free proxies, NM-matched non-LIRF rows, mid frame")
    base = (pl.scan_parquet(RK / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
            .select(F64, apt="ADEP_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
                    nmu=pl.col("FLIGHT_ID_mvt").is_null(), a3=pl.col("AOBT_3_flt").dt.epoch("ms").cast(pl.Float64).truediv(1000)).collect())
    x = (base.join(tsec(RK / "ranking.parquet", "TL"), on="MVT_ID_mvt").join(tsec(RK / "final_ranking.parquet", "TF"), on="MVT_ID_mvt")
         .join(dets(ROOT / "cache/adsb_pushback_v4/day=2026-*.parquet", "L"), on="MVT_ID_mvt", how="left")
         .join(dets(ROOT / "cache/adsb_pushback_v4/final4m/day=2026-0[17]-*.parquet", "F"), on="MVT_ID_mvt", how="left")
         .join(v75, on="MVT_ID_mvt").join(f4, on="MVT_ID_mvt").join(v76, on="MVT_ID_mvt")
         .filter((pl.col("apt") != "LIRF") & ~pl.col("nmu")))
    same = (pl.col("tierL").is_in(GOOD) & pl.col("tierF").is_in(GOOD) & (pl.col("tierL") == pl.col("tierF"))
            & ((pl.col("pbL") - pl.col("pbF")).abs() < 1)).fill_null(False)
    anyg = (pl.col("tierL").is_in(GOOD).fill_null(False) | pl.col("tierF").is_in(GOOD).fill_null(False))
    x = x.with_columns(trk=same, dif=anyg & ~same, z3=(pl.col("TL") + pl.col("TF")) / 2 - pl.col("a3"),
                       zb=(pl.col("TL") + pl.col("TF")) / 2 - pl.col("pbL"))
    print("   T - AOBT_3:")
    for lab, g in (("all", x), ("tracked, identical detections", x.filter(pl.col("trk"))),
                   ("untracked", x.filter(~pl.col("trk") & ~pl.col("dif"))), ("detections differ", x.filter(pl.col("dif")))):
        print(f"     {lab:30s} {gain(g.filter(pl.col('z3').is_not_null()), 'z3')}")
    y = x.filter(pl.col("z3").is_between(60, 3600)).with_columns(g=(pl.col("p") - pl.col("z3")) ** 2 - (pl.col("a") - pl.col("z3")) ** 2)
    c = y.group_by("apt", "day").agg(sg=pl.col("g").sum(), n=pl.len())
    sg, nn, G = c["sg"].to_numpy(), c["n"].to_numpy(), c.height
    mean = sg.sum() / nn.sum()
    se = np.sqrt(G / (G - 1) * np.sum((sg - mean * nn) ** 2)) / nn.sum()
    rng = np.random.default_rng(0)
    bs = np.array([sg[i].sum() / nn[i].sum() for i in (rng.integers(0, G, G) for _ in range(2000))])
    print(f"     airport-day clustered: {mean:.1f} +- {se:.1f} s^2/row ({G} clusters); bootstrap 95% CI [{np.quantile(bs, 0.025):.1f}, "
          f"{np.quantile(bs, 0.975):.1f}], P(average worse) {np.mean(bs < 0):.4f}")
    cell = y.group_by("apt", "mo").agg(g=pl.col("g").mean())
    print(f"     airport x month cells where the average is better: {(cell['g'] > 0).sum()} of {cell.height}")
    print("   T - ADS-B pushback (identical detections):")
    print(f"     {'tracked':30s} {gain(x.filter(pl.col('trk')), 'zb')}")

    print("F. release frame")
    e, d = (x["TF"] - x["TL"]).to_numpy(), (x["q"] - x["p"]).to_numpy()
    print(f"   sd(T_final - T_live) {e.std():.2f} s; slope of (four-month - live) on it {np.cov(d, e)[0, 1] / np.var(e):.3f}; "
          f"share of the difference variance {np.corrcoef(d, e)[0, 1] ** 2:.3f}; E[d^2]/4 {np.mean(d ** 2) / 4:.1f} s^2/row")


if __name__ == "__main__":
    main()
