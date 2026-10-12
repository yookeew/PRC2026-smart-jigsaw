"""LIRF NM-unmatched departures 1-6 h late: the airline's own 2025 echo record (PROGRESS.md §157, §158).

On these rows (v48 scope: NM-unmatched, 1 h <= T - S < 6 h, stack p < 0.99, no arrival twin) the label is either the echo
(block = SCHED, y = T - S) or a late pushback with a stale LOBT (y = a normal taxi <= T - S - 3,606). Which one is an airline /
handler practice, and it is persistent: in 2025 ETH 21/21, TKJ 20/20, ISR 56/56, LAV 22/22, TWB 19/19 are echoes, ITY 3/61,
KLM 0/17, EWG 0/14, ELY 0/12, EZY 1/24 almost never. v48's echo probability q2 prices it only partly.

For airlines with a decisive 2025 record (n >= N_MIN scope rows and (e + 1)/(n + 2) <= R_LO or >= 1 - R_LO):
    pi    = sigmoid(b0 + b1 logit(q2) + b2 logit((e + 1)/(n + 2)))     logistic fitted on 2025, records and q2 out of month
    value = max(pi * offc + (1 - pi) * m(ub), 600),  m(ub) = mean of the 2025 late-branch labels <= ub, ub = T - S - 3,606
Rows change only if
  - the input value equals the replayed hedge chain P0 -> v45 -> v47 -> v48 (+-1 s) or v71's 600 s floor of it;
  - the row has no arrival twin (exact or with a doubled prefix removed);
  - for low-echo airlines, the flight number is well formed (null / doubled prefix / odd / ADES = LIRF are key failures,
    where the airline's time-failure record does not apply), and a row >= 4 h late (beyond every 2025 low-echo record row)
    changes only if q2 <= 0.2 agrees with the record;
  - persistence guard (label-free): the airline is skipped when its LIRF early-arrival echo rate in the scope's 2026 months
    moved from its rate in the same 2025 months by more than 0.15 and more than 3 standard errors (>= 20 such arrivals in 2026).
    On the live board KMM fails (0.79 -> 0.33).

Run:  .venv/Scripts/python.exe -u src/post/lirf_airline_record.py IN.parquet OUT.parquet [P0.parquet]
      P0 default: data/submissions/smart-jigsaw_v42.parquet (live board) / data/submissions/final4m/f4m_a.parquet (final4m)
      PRC_SCOPE=final4m ... (four-month template; src/scope.py)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scope  # noqa: E402
import post.lirf_signals as L  # noqa: E402
from post.lirf_hedge2 import add_rate, airline_late, design2, fit2, frame25, rate_table, treat, with_loo_rate  # noqa: E402
from post.lirf_twin_bounds import WIN, bounds, twin_frame  # noqa: E402
from post.lirf_hedge_repair import twins_norm  # noqa: E402

N_MIN, R_LO, N_COMBO, F_FLOOR, Z_ARR, D_ARR, N_ARR = 12, 0.1, 5, 600.0, 3.0, 0.15, 20
OFF_SUPPORT, Q2_AGREE = 4 * 3600, 0.2               # 2025 low-echo records end below 4 h late; beyond, act only if q2 <= 0.2 agrees
EXPECT_LIVE = 54                                    # rows changed on the live board (scratch analysis, §158)
FL = pl.col("FLIGHT_mvt")
FMT = (pl.when(FL.is_null() | (pl.col("ADES_mvt") == "LIRF")).then(pl.lit("key"))
       .when((FL.str.slice(1, 2) == FL.str.slice(3, 2)) | (FL.str.slice(2, 1) == FL.str.slice(3, 1))).then(pl.lit("key"))
       .when(FL.str.contains(r"^[A-Z]{3}[0-9]{1,4}[A-Z]{0,2}$")).then(pl.lit("time")).otherwise(pl.lit("key")))


def lg(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def record_2025() -> dict:
    """2025 v48-scope rows with out-of-month q2 and airline records; the full-year record; the combo; the late-branch pool."""
    src = str(L.RAW / "training_*.parquet")
    hist = airline_late(src)
    d = with_loo_rate(frame25(), hist)
    tw = set(twin_frame(src)["MVT_ID_mvt"].to_list())
    d = d.with_columns(twin=pl.col("MVT_ID_mvt").is_in(list(tw)))
    parts = []
    for m in sorted(d["mt"].unique().to_list()):                       # q2 as v48 computes it, fitted on the other 11 months
        tr, te = d.filter(pl.col("mt") != m), d.filter(pl.col("mt") == m)
        c45, c2 = L.fit(tr), fit2(tr)
        base, _, _ = L.apply(te, te["Pb"].to_numpy(), c45)
        _, k = treat(te, base, c45, c2, te["twin"].to_numpy())
        if k.any():
            zk = te.filter(pl.Series(k) & pl.col("echo").is_not_null())
            parts.append(zk.select("MVT_ID_mvt", "mt", "al", "off", "y", "echo").with_columns(q2=pl.Series(c2.predict_proba(design2(zk))[:, 1])))
    s = pl.concat(parts)
    rec = lambda x: x.group_by("al").agg(n=pl.len(), e=pl.col("echo").cast(pl.Int64).sum())  # noqa: E731
    lomo = pl.concat([s.filter(pl.col("mt") == m).join(rec(s.filter(pl.col("mt") != m)), on="al", how="left")
                      for m in sorted(s["mt"].unique().to_list())]).with_columns(pl.col("n", "e").fill_null(0))
    t = lomo.filter(pl.col("n") >= N_COMBO)
    clf = LogisticRegression(C=10).fit(np.column_stack([lg(t["q2"].to_numpy()), lg((t["e"].to_numpy() + 1) / (t["n"].to_numpy() + 2))]),
                                       t["echo"].to_numpy().astype(int))
    late = np.sort(s.filter(~pl.col("echo") & (pl.col("y") <= pl.col("off") - WIN + 60))["y"].to_numpy())
    return dict(rows=s, rec=rec(s), clf=clf, late=late, hist=hist, d25=d)


def arrival_echo(src, months) -> pl.DataFrame:
    """LIRF early arrivals (landing > 10 min before schedule) per airline: count and echo share (in-block == scheduled)."""
    return (pl.scan_parquet(src).filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")
                                        & pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months)
                                        & ((pl.col("SCHED_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds() > 600))
            .select(al=FL.str.slice(0, 3), echo=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() < 30)
            .group_by("al").agg(na=pl.len(), ea=pl.col("echo").mean()).collect())


def main(in_f, out_f, p0_f) -> None:
    rk = str(L.RANKING)
    st = record_2025()
    clf, late = st["clf"], st["late"]
    print(f"2025 v48-scope rows {st['rows'].height} (echo {int(st['rows']['echo'].sum())}); combo: logit(q2) {clf.coef_[0][0]:.3f}, "
          f"logit(record) {clf.coef_[0][1]:.3f}, intercept {clf.intercept_[0]:.3f}; late-branch labels {len(late)} (mean {late.mean():.0f})")
    assert 0.6 <= clf.coef_[0][0] <= 1.2 and 0.5 <= clf.coef_[0][1] <= 1.0 and 1000 <= late.mean() <= 1300
    rec = st["rec"].with_columns(r=(pl.col("e") + 1) / (pl.col("n") + 2))
    dec = rec.filter((pl.col("n") >= N_MIN) & ((pl.col("r") <= R_LO) | (pl.col("r") >= 1 - R_LO)))
    # persistence guard on the arrival echo
    months = sorted(pl.scan_parquet(rk).select(pl.col("MVT_TIME_UTC_mvt").dt.month().unique()).collect().to_series().to_list())
    arr = (arrival_echo(str(L.RAW / "training_*.parquet"), months).rename({"na": "na25", "ea": "ea25"})
           .join(arrival_echo(rk, months).rename({"na": "na26", "ea": "ea26"}), on="al", how="left"))
    dec = dec.join(arr, on="al", how="left").with_columns(
        z=((pl.col("ea26") - pl.col("ea25")).abs() / (pl.col("ea25").clip(0.02, 0.98) * (1 - pl.col("ea25").clip(0.02, 0.98)) / pl.col("na26")).sqrt()))
    dec = dec.with_columns(skip=(pl.col("na26") >= N_ARR) & (pl.col("z") > Z_ARR) & ((pl.col("ea26") - pl.col("ea25")).abs() > D_ARR)
                           ).with_columns(pl.col("skip").fill_null(False))
    pl.Config.set_tbl_rows(40)
    print("decisive airlines (2025 record, arrival-echo persistence check on months", months, "):")
    print(dec.select("al", "n", "e", "r", "na25", "ea25", "na26", "ea26", "z", "skip").sort("al"))
    lo_al = dec.filter(~pl.col("skip") & (pl.col("r") <= R_LO))["al"].to_list()
    hi_al = dec.filter(~pl.col("skip") & (pl.col("r") >= 1 - R_LO))["al"].to_list()

    # replay the hedge chain exactly as src/post/lirf_hedge_repair.py does
    hist, d25 = st["hist"], st["d25"].drop("twin")
    clf45, clf2 = L.fit(d25), fit2(d25)
    tpl = pl.read_parquet(L.TEMPLATE)
    sub = pl.read_parquet(in_f)
    dtype = sub["TAXITIME_SEC_mvt"].dtype
    inp = sub.select(L.I64, v=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    P0 = pl.read_parquet(p0_f).select(L.I64, P0=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    a = (pl.read_parquet(rk).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
         .select(L.I64, "FLIGHT_mvt", "ADES_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                 off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
         .with_columns(fmt=FMT, al=FL.str.slice(0, 3)).join(inp, on="MVT_ID_mvt").join(P0, on="MVT_ID_mvt"))
    d = (L.frame(rk).join(L.p_2026(), on="MVT_ID_mvt", how="left").join(airline_late(rk).select("MVT_ID_mvt", "al"), on="MVT_ID_mvt", how="left")
         .join(P0, on="MVT_ID_mvt"))
    d = add_rate(d, rate_table(hist))
    tw_ex = twin_frame(rk).join(L.p_2026(), on="MVT_ID_mvt", how="left")
    d = d.with_columns(twin_ex=pl.col("MVT_ID_mvt").is_in(tw_ex["MVT_ID_mvt"].to_list()))
    v45, _, _ = L.apply(d, d["P0"].to_numpy(), clf45)
    v48, k48 = treat(d, v45, clf45, clf2, d["twin_ex"].to_numpy())
    q2 = np.full(d.height, np.nan)
    if k48.any():
        q2[k48] = clf2.predict_proba(design2(d.filter(pl.Series(k48))))[:, 1]
    rep = d.select("MVT_ID_mvt").with_columns(v45=pl.Series(v45), v48=pl.Series(v48), k48=pl.Series(k48), q2=pl.Series(q2))
    a = a.join(rep, on="MVT_ID_mvt", how="left").with_columns(pl.col("k48").fill_null(False), v45=pl.coalesce("v45", "P0"))
    tw_ex = tw_ex.join(a.select("MVT_ID_mvt", "v45"), on="MVT_ID_mvt", how="left")
    q47 = bounds(tw_ex["v45"].to_numpy(), tw_ex["p"].to_numpy(), tw_ex["off"].to_numpy(), tw_ex["tl"].to_numpy())
    a = (a.join(tw_ex.select("MVT_ID_mvt", v47=pl.Series(q47)), on="MVT_ID_mvt", how="left")
         .join(twins_norm(rk), on="MVT_ID_mvt", how="left")
         .with_columns(chain=pl.when(pl.col("k48")).then(pl.col("v48")).when(pl.col("v47").is_not_null()).then(pl.col("v47")).otherwise(pl.col("v45"))))
    v, chain = a["v"].to_numpy(), a["chain"].round().to_numpy()
    guard = (np.abs(v - chain) <= 1.0) | ((chain < F_FLOOR) & (np.abs(v - F_FLOOR) <= 0.5))
    istw = a["tl"].is_not_null().to_numpy()
    scope48 = a["k48"].to_numpy() & ~istw
    print(f"LIRF NM-unmatched rows {a.height}; v48 scope (no twin) {scope48.sum()}, of which equal to the replayed chain or its floor {(scope48 & guard).sum()}")

    al, fmt = a["al"].to_numpy(), a["fmt"].to_numpy()
    rr = a.select("al").join(rec, on="al", how="left")
    n_, e_ = rr["n"].fill_null(0).to_numpy(), rr["e"].fill_null(0).to_numpy()
    q2a = a["q2"].to_numpy()
    off = a["off"].to_numpy(); offc = np.clip(off, 0, L.WIDE); ub = np.maximum(off - WIN, 0)
    with np.errstate(invalid="ignore"):
        lo_ok = np.isin(al, lo_al) & (fmt == "time") & ((off < OFF_SUPPORT) | (q2a <= Q2_AGREE))
    act = scope48 & guard & (lo_ok | np.isin(al, hi_al))
    pi = np.full(a.height, np.nan)
    pi[act] = clf.predict_proba(np.column_stack([lg(q2a[act]), lg((e_[act] + 1) / (n_[act] + 2))]))[:, 1]
    m_ub = np.array([late[late <= u].mean() if (late <= u).sum() >= 20 else late[:20].mean() for u in ub])
    new = np.where(act, np.maximum(pi * offc + (1 - pi) * m_ub, F_FLOOR), v)
    res = (a.select("MVT_ID_mvt", "FLIGHT_mvt", "mo", "off", "fmt").with_columns(q2=pl.Series(q2a), pi=pl.Series(pi), v=pl.Series(v), new=pl.Series(np.round(new)),
                                                                                 act=pl.Series(act))
           .with_columns(d=pl.col("new") - pl.col("v")))
    ch = res.filter(pl.col("act") & (pl.col("d").abs() > 0.5))
    pl.Config.set_tbl_rows(80)
    print(ch.sort("d").select("FLIGHT_mvt", "mo", "fmt", "off", "q2", "pi", "v", "new", "d"))
    # stop rules
    energy = float((ch["d"] ** 2).sum())
    print(f"rows active {int(act.sum())}, changed {ch.height} (Jan {ch.filter(pl.col('mo') == 1).height}, Jul {ch.filter(pl.col('mo') == 7).height}, "
          f"other months {ch.filter(~pl.col('mo').is_in([1, 7])).height}); sum d^2 {energy:.3g}; RMS over the file {np.sqrt(energy / tpl.height):.2f} s")
    nv, ov = ch["new"].to_numpy(), ch["off"].to_numpy()
    assert np.all(nv >= F_FLOOR) and np.all(nv <= np.maximum(np.clip(ov, 0, L.WIDE), F_FLOOR) + 1), "stop rule: value outside [600, offc]"
    if not scope.FINAL:
        assert ch.height == EXPECT_LIVE, f"stop rule: rows changed {ch.height} != {EXPECT_LIVE}"
        assert energy <= 2.0e8, f"stop rule: sum d^2 {energy:.3g}"
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
