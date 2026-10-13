"""LIRF NM-unmatched hedge repair and shifted block times (reports/lirf_hedge_repair_preregistration.md, PROGRESS.md §155).

L1  N1  v48-scope rows (1 h <= off < 6 h, non-twin): final floor at 600 s
    T1  twin rows whose echo the join window excludes: min(E[y | 2025 twin non-echo, y <= up], up)
    T2  forced twin rows (block before LOBT_twin - 3,606) with the echo possible: P_E offc + (1 - P_E) (lo + 900)
L2  shifted block times, rows 0 < off < 1 h (non-twin):
    key    key-failure flight numbers (null / doubled prefix / odd):  PE_KEY off + (1 - PE_KEY) M_N
    shift  well-formed flight numbers of airlines with >= 2 shifted 2025 rows:
           psh (tzo + M_SH) + (1 - psh) (PE_TM off + (1 - PE_TM) M_N),  psh = (s + 2 P0T) / (n + 2)
    twin   forced twin rows with off < lo, well-formed flight number: +1 h / +2 h mixture (>= lo)
L3  (LHR_L3=1, default off) v48-scope rows: non-echo branch capped at 0.25 P_ne + 0.75 C_LATE (only lowers)

Guard: a row changes only if its input value equals (+-1 s) the replayed hedge chain P0 -> v45 -> v47 -> v48
(lirf_signals, lirf_twin_bounds, lirf_hedge2), so no row a later stage touched is changed.

Run:  .venv/Scripts/python.exe -u src/post/lirf_hedge_repair.py IN.parquet OUT.parquet [P0.parquet]
      P0 default: data/submissions/smart-jigsaw_v42.parquet (live board) / data/submissions/final4m/f4m_a.parquet (final4m)
      PRC_SCOPE=final4m ... (four-month template; src/scope.py)
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
import post.lirf_signals as L  # noqa: E402
from post.lirf_hedge2 import add_rate, airline_late, design2, fit2, frame25, rate_table, treat, with_loo_rate  # noqa: E402
from post.lirf_twin_bounds import FORCED, WIN, bounds, twin_frame  # noqa: E402

F_FLOOR, ALT_T2, S_MIN, K_PRIOR = 600.0, 900.0, 2, 2.0
A_PNE, C_LATE = 0.25, 1097.0                       # L3 only (off by default)
L3_ON = os.environ.get("LHR_L3", "0") == "1"
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
EXPECT_LIVE = {"N1": 29, "T1": 22, "T2": 13, "L2key": 16, "L2shift": 3, "L2twin": 2}
FL = pl.col("FLIGHT_mvt")
FMT = (pl.when(FL.is_null()).then(pl.lit("key"))
       .when((FL.str.slice(1, 2) == FL.str.slice(3, 2)) | (FL.str.slice(2, 1) == FL.str.slice(3, 1))).then(pl.lit("key"))
       .when(FL.str.contains(r"^[A-Z]{3}[0-9]{1,4}[A-Z]{0,2}$")).then(pl.lit("time")).otherwise(pl.lit("key")))
LT = pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome")
TZO = (LT.dt.base_utc_offset().dt.total_seconds() + LT.dt.dst_offset().dt.total_seconds()).cast(pl.Float64)


def norm_key(c: pl.Expr) -> pl.Expr:
    """Flight number with a doubled prefix removed: ITYTY672 -> ITY672, AEZZ2023 -> AEZ2023."""
    a = c.str.slice(0, 3)
    return (pl.when(c.str.slice(1, 2) == c.str.slice(3, 2)).then(a + c.str.slice(5))
            .when(c.str.slice(2, 1) == c.str.slice(3, 1)).then(a + c.str.slice(4)).otherwise(c))


def twins_norm(src) -> pl.DataFrame:
    """LIRF NM-unmatched departures with an NM-matched arrival twin (flight number exact or with a doubled prefix removed)."""
    ts = lambda c: pl.col(c).dt.epoch("s")  # noqa: E731
    m = (pl.scan_parquet(src).select(L.I64, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", T=ts("MVT_TIME_UTC_mvt"),
                                     S=ts("SCHED_TIME_UTC_mvt"), La=ts("LOBT_flt")).collect().with_columns(key=norm_key(FL)))
    dep = m.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()
                   & pl.col("ADES_mvt").is_in(APT) & (pl.col("ADES_mvt") != "LIRF") & FL.is_not_null())
    arr = (m.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("La").is_not_null())
           .select("ADEP_mvt", "ADES_mvt", "key", Lt="La", Ta="T"))
    j = (dep.join(arr, on=["ADEP_mvt", "ADES_mvt", "key"]).filter((pl.col("Ta") > pl.col("T")) & (pl.col("Ta") < pl.col("T") + 8 * 3600))
         .sort("Ta").group_by("MVT_ID_mvt", maintain_order=True).first())
    return j.select("MVT_ID_mvt", tl=(pl.col("T") - pl.col("Lt")).cast(pl.Float64))


def stats_2025() -> dict:
    """Every constant the rules use, from all of 2025."""
    src = str(L.RAW / "training_*.parquet")
    tw = twin_frame(src)
    yl = np.sort(tw.filter((pl.col("tl") >= FORCED) & (pl.col("off") > pl.col("tl") - WIN))["y"].to_numpy())
    fe = tw.filter((pl.col("tl") < FORCED) & (pl.col("off") >= pl.col("tl") + WIN))
    p_e = (float(((fe["y"] - fe["off"]).abs() < 60).sum()) + 1) / (fe.height + 2)
    twn = set(twins_norm(src)["MVT_ID_mvt"].to_list())
    y, off = pl.col("y"), pl.col("off")
    z = (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
         .select(L.I64, "FLIGHT_mvt", off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
                 y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).collect()
         .filter((off > 0) & (off < 3600) & ~pl.col("MVT_ID_mvt").is_in(list(twn)))
         .with_columns(fmt=FMT, al=FL.str.slice(0, 3))
         .with_columns(cls=pl.when((y - off).abs() < 60).then(pl.lit("echo"))
                       .when((y - 3600).is_between(240, 2700) | (y - 7200).is_between(240, 2700)).then(pl.lit("sh"))
                       .when(y <= 2700).then(pl.lit("normal")).otherwise(pl.lit("other"))))
    key = z.filter((pl.col("fmt") == "key") & (pl.col("cls") != "other"))
    tim = z.filter((pl.col("fmt") == "time") & (pl.col("cls") != "other"))
    sh = z.filter(pl.col("cls") == "sh")
    return dict(
        yl=yl, P_E=p_e, n_forced=fe.height,
        PE_KEY=float((key["cls"] == "echo").mean()), P0T=float((tim["cls"] == "sh").mean()),
        PE_TM=float((tim.filter(pl.col("cls") != "sh")["cls"] == "echo").mean()),
        M_N=float(z.filter(pl.col("cls") == "normal")["y"].mean()),
        M_SH=float(np.median((sh["y"] - np.where(sh["y"] > 6000, 7200, 3600)).to_numpy())),
        rec=tim.group_by("al").agg(n=pl.len(), s=(pl.col("cls") == "sh").sum()),
        table=z.group_by("fmt", "cls").len().sort("fmt", "cls"))


def main(in_f, out_f, p0_f) -> None:
    rk = str(L.RANKING)
    st = stats_2025()
    print(f"2025 constants: twin non-echo labels {len(st['yl'])} (mean {st['yl'].mean():.0f}); P_E {st['P_E']:.3f} (forced, echo possible: "
          f"{st['n_forced']}); PE_KEY {st['PE_KEY']:.3f}; P0T {st['P0T']:.3f}; PE_TM {st['PE_TM']:.3f}; M_N {st['M_N']:.0f}; M_SH {st['M_SH']:.0f}")
    print(st["table"].pivot(on="cls", index="fmt", values="len").fill_null(0))
    assert 0.80 <= st["P_E"] <= 0.95 and 0.70 <= st["PE_KEY"] <= 0.85 and 0.55 <= st["P0T"] <= 0.80 and 700 <= st["M_SH"] <= 1000

    # production classifiers of v45 / v48, fitted on 2025 exactly as lirf_hedge2.main does
    hist = airline_late(str(L.RAW / "training_*.parquet"))
    d25 = with_loo_rate(frame25(), hist)
    clf45, clf2 = L.fit(d25), fit2(d25)

    tpl = pl.read_parquet(L.TEMPLATE)
    sub = pl.read_parquet(in_f)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    inp = sub.select(L.I64, v=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    P0 = pl.read_parquet(p0_f).select(L.I64, P0=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    a = (pl.read_parquet(rk).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
         .select(L.I64, "FLIGHT_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), tzo=TZO,
                 off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
         .with_columns(fmt=FMT, al=FL.str.slice(0, 3)).join(inp, on="MVT_ID_mvt").join(P0, on="MVT_ID_mvt"))

    # replay the hedge chain P0 -> v45 -> v47 -> v48
    d = (L.frame(rk).join(L.p_2026(), on="MVT_ID_mvt", how="left").join(airline_late(rk).select("MVT_ID_mvt", "al"), on="MVT_ID_mvt", how="left")
         .join(P0, on="MVT_ID_mvt"))
    d = add_rate(d, rate_table(hist))
    tw_ex = twin_frame(rk).join(L.p_2026(), on="MVT_ID_mvt", how="left")
    d = d.with_columns(twin_ex=pl.col("MVT_ID_mvt").is_in(tw_ex["MVT_ID_mvt"].to_list()))
    v45, _, _ = L.apply(d, d["P0"].to_numpy(), clf45)
    v48, k48 = treat(d, v45, clf45, clf2, d["twin_ex"].to_numpy())
    q2 = np.full(d.height, np.nan); q45 = np.full(d.height, np.nan)
    if k48.any():
        zk = d.filter(pl.Series(k48))
        q45[k48] = clf45.predict_proba(L.design(zk))[:, 1]
        q2[k48] = clf2.predict_proba(design2(zk))[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pne = (v45 - q45 * np.clip(d["off"].to_numpy(), 0, L.WIDE)) / (1 - q45)
    rep = d.select("MVT_ID_mvt").with_columns(v45=pl.Series(v45), v48=pl.Series(v48), k48=pl.Series(k48), q2=pl.Series(q2), pne=pl.Series(pne))
    a = a.join(rep, on="MVT_ID_mvt", how="left").with_columns(pl.col("k48").fill_null(False), v45=pl.coalesce("v45", "P0"))
    tw_ex = tw_ex.join(a.select("MVT_ID_mvt", "v45"), on="MVT_ID_mvt", how="left")
    q47 = bounds(tw_ex["v45"].to_numpy(), tw_ex["p"].to_numpy(), tw_ex["off"].to_numpy(), tw_ex["tl"].to_numpy())
    a = (a.join(tw_ex.select("MVT_ID_mvt", v47=pl.Series(q47)), on="MVT_ID_mvt", how="left")
         .join(twins_norm(rk), on="MVT_ID_mvt", how="left")
         .with_columns(chain=pl.when(pl.col("k48")).then(pl.col("v48")).when(pl.col("v47").is_not_null()).then(pl.col("v47")).otherwise(pl.col("v45"))))
    v, chain = a["v"].to_numpy(), a["chain"].round().to_numpy()
    guard = np.abs(v - chain) <= 1.0
    off = a["off"].to_numpy(); offc = np.clip(off, 0, L.WIDE); tzo = a["tzo"].to_numpy(); fmt = a["fmt"].to_numpy()
    tl = a["tl"].fill_null(np.nan).to_numpy(); istw = ~np.isnan(tl); up, lo = tl - WIN, tl + WIN
    k48a = a["k48"].to_numpy()
    print(f"LIRF NM-unmatched rows {a.height}; v48 scope {k48a.sum()}; twins (exact + normalised) {istw.sum()}; "
          f"rows equal to the replayed chain {guard.sum()} ({(~guard).sum()} touched by later stages, left alone)")
    out, tag = v.copy(), np.array([""] * a.height, dtype=object)

    def put(mask, val, name):
        nonlocal out
        m = mask & guard & (tag == "")
        out = np.where(m, val, out); tag[m] = name

    with np.errstate(invalid="ignore"):
        noecho = istw & (tl >= FORCED) & (off > up)
        forced = istw & (tl < FORCED)
    scope48 = k48a & ~istw
    # L3 (off by default): cap the non-echo branch, then N1 floors
    if L3_ON:
        q2a, pnea = a["q2"].to_numpy(), a["pne"].to_numpy()
        with np.errstate(invalid="ignore"):
            cap = q2a * offc + (1 - q2a) * np.minimum(np.maximum(A_PNE * pnea + (1 - A_PNE) * C_LATE, 0), np.maximum(off - WIN, 0))
        l3 = scope48 & guard & (np.nan_to_num(cap, nan=1e12) < v - 0.5)
        out = np.where(l3, cap, out); tag[l3] = "L3"
    n1 = scope48 & guard & (out < F_FLOOR)
    out = np.where(n1, F_FLOOR, out); tag[n1] = np.where(tag[n1] == "L3", "L3+N1", "N1")
    # L1 T1 / T2
    yl = st["yl"]
    t1v = np.array([min((yl[yl <= u].mean() if (yl <= u).sum() >= 10 else min(yl[:10].mean(), u)), u) if ok else np.nan for u, ok in zip(up, noecho)])
    put(noecho & (np.abs(np.nan_to_num(t1v) - out) > 0.5), t1v, "T1")
    t2v = st["P_E"] * offc + (1 - st["P_E"]) * (lo + ALT_T2)
    with np.errstate(invalid="ignore"):
        put(forced & (off >= lo) & (np.abs(np.nan_to_num(t2v) - out) > 0.5), t2v, "T2")
    # L2
    rec = a.select("al").join(st["rec"], on="al", how="left").with_columns(pl.col("n", "s").fill_null(0))
    n_, s_ = rec["n"].to_numpy(), rec["s"].to_numpy()
    small = (off > 0) & (off < 3600) & ~istw
    key = fmt == "key"
    put(small & key, st["PE_KEY"] * off + (1 - st["PE_KEY"]) * st["M_N"], "L2key")
    psh = (s_ + K_PRIOR * st["P0T"]) / (n_ + K_PRIOR)
    hsh = psh * (tzo + st["M_SH"]) + (1 - psh) * (st["PE_TM"] * off + (1 - st["PE_TM"]) * st["M_N"])
    put(small & ~key & (s_ >= S_MIN), hsh, "L2shift")
    h1, h2 = np.maximum(lo, 3600 + st["M_SH"]), np.maximum(lo, 7200 + st["M_SH"])
    with np.errstate(invalid="ignore"):
        put(forced & (off < lo) & ~key, np.where(tzo > 3600, 0.5 * h1 + 0.5 * h2, h1), "L2twin")

    res = a.select("MVT_ID_mvt", "FLIGHT_mvt", "mo", "off").with_columns(v=pl.Series(v), new=pl.Series(np.round(out)), tag=pl.Series(tag.astype(str)))
    res = res.with_columns(d=pl.col("new") - pl.col("v"))
    ch = res.filter((pl.col("tag") != "") & (pl.col("d").abs() > 0.5))
    pl.Config.set_tbl_rows(60)
    print(ch.group_by("tag").agg(n=pl.len(), mean=pl.col("d").mean().round(0), rms=(pl.col("d") ** 2).mean().sqrt().round(0),
                                 energy=(pl.col("d") ** 2).sum()).sort("tag"))
    print(ch.filter(pl.col("tag").is_in(["L2shift", "L2twin"])).select("FLIGHT_mvt", "mo", "off", "v", "new", "tag"))

    # stop rules
    tg = ch["tag"].to_numpy(); ii = {m: i for i, m in enumerate(a["MVT_ID_mvt"].to_list())}
    idx = np.array([ii[m] for m in ch["MVT_ID_mvt"].to_list()], dtype=int)
    nv = ch["new"].to_numpy()
    assert guard[idx].all(), "stop rule 1: a changed row fails the guard"
    if not scope.FINAL and not L3_ON:
        got = {k: int((tg == k).sum()) for k in EXPECT_LIVE}
        assert got == EXPECT_LIVE, f"stop rule 2: rows per tag {got} != {EXPECT_LIVE}"
    chk = {"N1": np.all(nv[tg == "N1"] == F_FLOOR),
           "T1": np.all(nv[tg == "T1"] <= np.round(up[idx][tg == "T1"]) + 1),
           "T2": np.all((nv[tg == "T2"] >= np.minimum(lo, offc)[idx][tg == "T2"] - 1) & (nv[tg == "T2"] <= np.maximum(lo + ALT_T2, offc)[idx][tg == "T2"] + 1)),
           "L2key": np.all((nv[tg == "L2key"] >= np.minimum(off, st["M_N"])[idx][tg == "L2key"] - 1) & (nv[tg == "L2key"] <= np.maximum(off, st["M_N"])[idx][tg == "L2key"] + 1)),
           "L2shift": np.all((nv[tg == "L2shift"] > v[idx][tg == "L2shift"]) & (nv[tg == "L2shift"] <= tzo[idx][tg == "L2shift"] + 3600)),
           "L2twin": np.all(nv[tg == "L2twin"] >= np.round(lo[idx][tg == "L2twin"]) - 1)}
    assert all(chk.values()), f"stop rule 3: value checks {chk}"
    energy = float((ch["d"] ** 2).sum())
    e_max = 1.5e8 * tpl.height / 344841          # live-board bound, per row: the four-month template is ~1.95x the rows
    assert L3_ON or energy <= e_max, f"stop rule 4: sum d^2 {energy:.3g} > {e_max:.3g}"
    print(f"stop rules 1-4 pass; rows changed {ch.height}, sum d^2 {energy:.3g}, RMS over the file {np.sqrt(energy / tpl.height):.2f} s")

    upd = ch.select(pl.col("MVT_ID_mvt").alias("_id"), "new")
    o = (sub.with_columns(_id=L.I64).join(upd, on="_id", how="left")
         .with_columns(TAXITIME_SEC_mvt=pl.coalesce(pl.col("new"), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).round().cast(dtype))
         .select(sub.columns))
    assert o.height == tpl.height and o["TAXITIME_SEC_mvt"].null_count() == 0 and float(o["TAXITIME_SEC_mvt"].min()) >= 0
    assert set(o["MVT_ID_mvt"].to_list()) == set(tpl["MVT_ID_mvt"].to_list())
    n_diff = int((o["TAXITIME_SEC_mvt"].cast(pl.Float64) != sub["TAXITIME_SEC_mvt"].cast(pl.Float64)).sum())
    assert n_diff == ch.height, f"rows differing {n_diff} != changed {ch.height}"
    o.write_parquet(out_f)
    print(f"wrote {out_f}")


if __name__ == "__main__":
    default_p0 = ROOT / "data" / "submissions" / ("final4m/f4m_a.parquet" if scope.FINAL else "smart-jigsaw_v42.parquet")
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else str(default_p0))
