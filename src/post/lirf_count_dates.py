"""LIRF off-block dates from the published daily departure counts (reports/lirf_count_dates_preregistration.md).

FLT_DEP_IFR_2 (all_pre_departure_delays_2026.csv) counts departures per UTC off-block date. For delayed LIRF departures
whose echo block (S) and real block (T - taxi) fall on different dates, the counts decide which date the block is on.
Rows changed: NM-unmatched, 1 h <= T - S < 12 h, decided date reachable only by an echo / "+24 h" block, prediction
below the echo value -> echo value T - S + e_T. Every other row keeps the input value.

The bookkeeping always uses the four-month file (final_ranking.parquet), so month boundaries inside it are complete.

Run:  .venv/Scripts/python.exe src/post/lirf_count_dates.py IN.parquet OUT.parquet
      PRC_SCOPE=final4m ... (four-month template)
"""
import datetime as dt
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "post"))
import scope  # noqa: E402
from post import pru_daily_recon as P  # noqa: E402

APT, NLO, NHI, WIN, MAX_COMP = "LIRF", 180, 3600, 3606, 16
D0 = dt.date(1970, 1, 1)
DAYS_OK = [(dt.date(2026, 1, 1), dt.date(2026, 1, 31)), (dt.date(2026, 2, 1), dt.date(2026, 2, 27)),
           (dt.date(2026, 6, 1), dt.date(2026, 6, 30)), (dt.date(2026, 7, 1), dt.date(2026, 7, 30))]


def candidates(T, S, allow_echo, nmu):
    """Candidate UTC block days (days since epoch) per row: real T - N, echo S +- 6 s, "+24 h" one day before real."""
    dS, dS2 = (S - 6) // 86400, (S + 6) // 86400
    r_lo, r_hi = (T - NHI) // 86400, (T - NLO) // 86400
    off = T - S
    out = []
    for i in range(len(T)):
        c = set(range(int(r_lo[i]), int(r_hi[i]) + 1))
        if allow_echo[i]:
            c |= {int(dS[i]), int(dS2[i])}
        if nmu[i] and off[i] >= 10 * 3600:
            c |= {d - 1 for d in range(int(r_lo[i]), int(r_hi[i]) + 1)}
        out.append(c)
    return out


def decide(T, S, allow_echo, nmu, pub_days: dict):
    """Decided block day per row (-1 = undecided / unambiguous)."""
    cand = candidates(T, S, allow_echo, nmu)
    certain, amb = {}, []
    for i, c in enumerate(cand):
        if len(c) == 1:
            d = next(iter(c)); certain[d] = certain.get(d, 0) + 1
        else:
            amb.append(i)
    k = {d: pub_days[d] - certain.get(d, 0) for d in pub_days}
    parent = list(range(len(amb)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    by_day = {}
    for j, i in enumerate(amb):
        for d in cand[i]:
            by_day.setdefault(d, []).append(j)
    for js in by_day.values():
        for j in js[1:]:
            ra, rb = find(js[0]), find(j)
            if ra != rb:
                parent[ra] = rb
    comps = {}
    for j in range(len(amb)):
        comps.setdefault(find(j), []).append(j)
    dec = np.full(len(T), -1)
    for js in comps.values():
        rows = [amb[j] for j in js]
        if len(rows) > MAX_COMP:
            continue
        known = [d for d in sorted(set().union(*[cand[i] for i in rows])) if d in k]
        best, keep = None, []
        for combo in itertools.product(*[sorted(cand[i]) for i in rows]):
            cnt = {}
            for d in combo:
                cnt[d] = cnt.get(d, 0) + 1
            viol = frozenset(d for d in known if cnt.get(d, 0) != k[d])
            if best is None or len(viol) < best:
                best, keep = len(viol), [(combo, viol)]
            elif len(viol) == best:
                keep.append((combo, viol))
        for pos, i in enumerate(rows):
            vals = {c[pos] for c, _ in keep}
            if len(vals) == 1 and all(not (viol & cand[i]) for _, viol in keep):
                dec[i] = next(iter(vals))
    return dec


def pub_days_2026() -> dict:
    p = P.pub(2026).filter(pl.col("apt") == APT)
    ok = {(a - D0).days + j for a, b in DAYS_OK for j in range((b - a).days + 1)}
    return {(d - D0).days: n for d, n in zip(p["day"].to_list(), p["nD"].to_list()) if (d - D0).days in ok}


def lirf_frame() -> pl.DataFrame:
    z = (pl.read_parquet(scope._RANK_DIR / "final_ranking.parquet")
         .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == APT))
         .select(pl.col("MVT_ID_mvt").cast(pl.Float64), "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
                 T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"), S=pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s"),
                 L=pl.col("LOBT_flt").dt.epoch("s"), nmu=pl.col("FLIGHT_ID_mvt").is_null()))
    z = z.with_columns(allow=pl.col("nmu") | ((pl.col("S") - pl.col("L")).abs() <= WIN).fill_null(False))
    T, S = z["T"].to_numpy(), z["S"].to_numpy()
    dec = decide(T, S, z["allow"].to_numpy(), z["nmu"].to_numpy(), pub_days_2026())
    eT = T % 60
    eT = np.where(eT > 30, eT - 60, eT)
    dS = (S - 6) // 86400
    real_ok = (dec >= (T - NHI) // 86400) & (dec <= (T - NLO) // 86400)
    echo_only = (dec >= 0) & (dec == dS) & ~real_ok
    real_only = (dec >= 0) & real_ok & (dec != dS)
    strad = dS != (T - 900) // 86400
    return z.with_columns(off=pl.Series((T - S).astype(float)), echo_val=pl.Series((T - S + eT).astype(float)),
                          echo_only=pl.Series(echo_only), real_only=pl.Series(real_only), strad=pl.Series(strad))


def main(in_f, out_f):
    z = lirf_frame()
    tpl = pl.read_parquet(scope.TEMPLATE)
    sub = pl.read_parquet(in_f).select(pl.col("MVT_ID_mvt").cast(pl.Float64), p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    x = tpl.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(sub, on="MVT_ID_mvt", how="left").join(z, on="MVT_ID_mvt", how="left")
    assert x.height == tpl.height and x["p"].null_count() == 0
    scope_rows = (x["nmu"].fill_null(False) & (x["off"] >= 3600) & (x["off"] < 12 * 3600)).to_numpy()
    st = scope_rows & x["strad"].fill_null(False).to_numpy()
    eo = x["echo_only"].fill_null(False).to_numpy()
    ro = x["real_only"].fill_null(False).to_numpy()
    p, ev = x["p"].to_numpy(), x["echo_val"].to_numpy()
    chg = scope_rows & eo & (p < ev)
    print(f"in-file NM-unmatched straddlers 1-12 h: {st.sum()}; decided echo-only {(st & eo).sum()}, real-only {(st & ro).sum()}, "
          f"undecided {(st & ~eo & ~ro).sum()}")
    dec_n = (st & (eo | ro)).sum()
    print(f"stop rule 3: echo-only share of decided {((st & eo).sum() / max(dec_n, 1)):.2f} (>= 0.80 required)")
    q = np.where(chg, ev, p)
    d = q - p
    assert np.all(d >= 0) and np.all(d <= 6000), "stop rule 2"
    print(x.filter(pl.Series(chg)).select("FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "off", "p", "echo_val")
          .with_columns(move=pl.col("echo_val") - pl.col("p")))
    print(f"rows changed {chg.sum()}, sum d^2 {np.sum(d ** 2):.3g}, RMS over all rows {np.sqrt(np.mean(d ** 2)):.2f}")
    out = pl.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"], "TAXITIME_SEC_mvt": np.round(q).astype(np.int32)})
    assert out["TAXITIME_SEC_mvt"].null_count() == 0 and (out["TAXITIME_SEC_mvt"] >= 0).all()
    base = pl.read_parquet(in_f)
    same = out.join(base.select("MVT_ID_mvt", b="TAXITIME_SEC_mvt"), on="MVT_ID_mvt")
    assert same.height == tpl.height and int((same["TAXITIME_SEC_mvt"] != same["b"]).sum()) == int(chg.sum())
    out.write_parquet(out_f)
    print("wrote", out_f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
