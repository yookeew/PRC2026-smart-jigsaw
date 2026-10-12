"""Scoping (PROGRESS §141), step 3: flight-number history on top of v48 with PAST-ONLY history, as 2026 has it.
Row in month mi gets history from late LIRF departures of the same FLIGHT_mvt in months <= mi - GAP (never the test month).
GAP 1 stands in for Jan 2026 (nearest history Dec 2025), GAP 7 for Jul 2026 (nearest Dec 2025); test months are those
with at least one month of eligible past (GAP 1: Feb-Dec; GAP 7: Aug-Dec)."""
import sys; sys.path[:0] = ["src", "tests"]
import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
import post.lirf_signals as L
import post.lirf_hedge2 as H
from post.lirf_twin_bounds import twin_frame
import lirf_flightno_echo as FE

src = str(L.RAW / "training_*.parquet")
hist_all = FE.deps(src).with_columns(mt=pl.format("2025-{}", pl.col("mi").cast(pl.Utf8).str.zfill(2)))
d = H.with_loo_rate(H.frame25(), H.airline_late(src))
d = d.join(hist_all.select("MVT_ID_mvt", "fn"), on="MVT_ID_mvt", how="left").with_columns(mi=pl.col("mt").str.slice(5, 2).cast(pl.Int64))
tw = set(twin_frame(src)["MVT_ID_mvt"].to_list()); d = d.with_columns(twin=pl.col("MVT_ID_mvt").is_in(list(tw)))
lo = lambda q: np.log(np.clip(q, .02, .98) / (1 - np.clip(q, .02, .98)))  # noqa: E731


def past_hist(z, hist, gap):
    h = hist.filter(pl.col("fn").is_not_null() & pl.col("has_blk")).select("fn", "nmu", hm=pl.col("mi").cast(pl.Int64), he=pl.col("echo").cast(pl.Float64))
    t = z.select("fn", "mi").with_row_index("_r")
    j = t.join(h, on="fn", how="inner").filter(pl.col("hm") <= pl.col("mi") - gap)
    g = j.group_by("_r", "nmu").agg(n=pl.len(), e=pl.col("he").sum())
    out = np.zeros((t.height, 4)); out[:, 1] = out[:, 3] = FE.PRIOR
    for nmu, c in ((True, 0), (False, 2)):
        s = g.filter(pl.col("nmu") == nmu); r = s["_r"].to_numpy()
        out[r, c] = s["n"].to_numpy(); out[r, c + 1] = (s["e"].to_numpy() + FE.W * FE.PRIOR) / (s["n"].to_numpy() + FE.W)
    f = np.column_stack([lo(out[:, 1]) * (out[:, 0] > 0), np.log1p(out[:, 0]), lo(out[:, 3]) * (out[:, 2] > 0), np.log1p(out[:, 2])])
    return f, out


for gap, months in ((1, range(2, 13)), (7, range(8, 13))):
    tot_b = tot_n = 0.0; better = 0; per = []; reach = nk = 0
    for mm in months:
        m = f"2025-{mm:02d}"
        tr, te = d.filter(pl.col("mt") != m), d.filter(pl.col("mt") == m)
        hist = hist_all.filter(pl.col("mt") != m)
        c45, c2 = L.fit(tr), H.fit2(tr)
        b45, _, _ = L.apply(te, te["Pb"].to_numpy(), c45)
        b48, k = H.treat(te, b45, c45, c2, te["twin"].to_numpy())
        p = te["p"].fill_null(1.0).to_numpy(); off = te["off"].to_numpy(); offc = np.clip(off, 0, L.WIDE)
        t = L.train_rows(tr)
        ft, _ = past_hist(t, hist, gap)
        c4 = LogisticRegression(C=1.0, max_iter=4000).fit(np.column_stack([H.design2(t), ft]), t["echo"].to_numpy().astype(int))
        q45 = p.copy(); q4 = p.copy()
        if k.any():
            zk = te.filter(pl.Series(k))
            fk, Hk = past_hist(zk, hist, gap)
            q45[k] = c45.predict_proba(L.design(zk))[:, 1]
            q4[k] = c4.predict_proba(np.column_stack([H.design2(zk), fk]))[:, 1]
            reach += int(((Hk[:, 0] + Hk[:, 2]) > 0).sum()); nk += int(k.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            pne = (b45 - q45 * offc) / (1 - q45)
        n4 = np.where(k, np.maximum(q4 * offc + (1 - q4) * np.minimum(pne, np.maximum(off - 3606, 0)), 0), b45)
        y = te["y"].to_numpy(); sb = np.sum(((y - b48) ** 2)[k]); sn = np.sum(((y - n4) ** 2)[k])
        tot_b += sb; tot_n += sn; better += sn < sb; per.append(f"{mm}:{sn / sb:.2f}" if sb > 0 else f"{mm}:-")
    print(f"past-only gap >= {gap} (test months {months.start}-{months.stop - 1}): in-scope SSE {tot_b:.4g} -> {tot_n:.4g} "
          f"({tot_n / tot_b:.3f}); months better {better}/{len(months)}; rows with a history {reach}/{nk}; {' '.join(per)}", flush=True)

# 2026 reach: v48-scope-like rows (LIRF NM-unmatched, 1-6 h late) with any 2025 history of the same flight number
rk = (pl.scan_parquet(str(L.RANKING)).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
      .select(fn="FLIGHT_mvt", mo=pl.col("MVT_TIME_UTC_mvt").dt.month(), off=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
      .filter((pl.col("off") >= 3600) & (pl.col("off") < 6 * 3600)).collect())
fns = hist_all.filter(pl.col("has_blk")).group_by("fn").agg(n=pl.len(), last=pl.col("mi").max())
r = rk.join(fns, on="fn", how="left")
print("2026 LIRF NM-unmatched 1-6 h late rows:", r.group_by("mo").agg(rows=pl.len(), with_hist=pl.col("n").is_not_null().sum(),
      hist_from_nov_dec=(pl.col("last") >= 11).sum()).sort("mo"))
