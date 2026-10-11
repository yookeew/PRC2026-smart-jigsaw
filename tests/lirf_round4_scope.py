"""Scoping (PROGRESS §141): flight-number echo history (§111) on top of v48's LIRF NM-unmatched hedge.

Same harness as tests/lirf_round3_scope.py: per test month m (2025), v45 / v48 refitted on the other months; round 4
replaces v48's q2 by q4 = logistic on [design2, flight-number history] with the same late-branch bound. History per row:
late (>= 1 h) LIRF departures of the same FLIGHT_mvt from months != m and != the row's own month, at least GAP months
away (tests/lirf_flightno_echo.py::history; NM-unmatched and NM-matched counted separately). Scope = v48's treated rows.
"""
import sys; sys.path[:0] = ["src", "tests"]
import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
import post.lirf_signals as L
import post.lirf_hedge2 as H
from post.lirf_twin_bounds import twin_frame
import lirf_flightno_echo as FE

src = str(L.RAW / "training_*.parquet")
hist_all = FE.deps(src)                                   # late LIRF departures 2025 with fn, nmu, mi, echo
hist_all = hist_all.with_columns(mt=pl.format("2025-{}", pl.col("mi").cast(pl.Utf8).str.zfill(2)))
d = H.with_loo_rate(H.frame25(), H.airline_late(src))
d = d.join(hist_all.select("MVT_ID_mvt", "fn", "mi"), on="MVT_ID_mvt", how="left")
tw = set(twin_frame(src)["MVT_ID_mvt"].to_list()); d = d.with_columns(twin=pl.col("MVT_ID_mvt").is_in(list(tw)))
d = d.with_columns(mi=pl.col("mi").fill_null(pl.col("mt").str.slice(5, 2).cast(pl.Int8)))


def hfeat(z: pl.DataFrame, hist: pl.DataFrame, gap: int) -> np.ndarray:
    Hh = FE.history(z.select("MVT_ID_mvt", "fn", "mi"), hist, gap)
    lo = lambda q: np.log(np.clip(q, .02, .98) / (1 - np.clip(q, .02, .98)))  # noqa: E731
    return np.column_stack([lo(Hh[:, 1]) * (Hh[:, 0] > 0), np.log1p(Hh[:, 0]), lo(Hh[:, 3]) * (Hh[:, 2] > 0), np.log1p(Hh[:, 2])]), Hh


for gap in (None, 1, 3, 6):
    tot_b = tot_n = 0.0; better = 0; per = []; reach = 0; nk = 0
    for m in sorted(d["mt"].unique().to_list()):
        tr, te = d.filter(pl.col("mt") != m), d.filter(pl.col("mt") == m)
        hist = hist_all.filter(pl.col("mt") != m)
        c45, c2 = L.fit(tr), H.fit2(tr)
        b45, _, _ = L.apply(te, te["Pb"].to_numpy(), c45)
        b48, k = H.treat(te, b45, c45, c2, te["twin"].to_numpy())
        p = te["p"].fill_null(1.0).to_numpy(); off = te["off"].to_numpy(); offc = np.clip(off, 0, L.WIDE)
        q45 = p.copy(); q4 = p.copy()
        if gap is None:      # control: v48 refit through this code path (must equal b48)
            n4 = b48
        else:
            t = L.train_rows(tr)
            ft, _ = hfeat(t, hist, gap)
            c4 = LogisticRegression(C=1.0, max_iter=4000).fit(np.column_stack([H.design2(t), ft]), t["echo"].to_numpy().astype(int))
            if k.any():
                zk = te.filter(pl.Series(k))
                fk, Hk = hfeat(zk, hist, gap)
                q45[k] = c45.predict_proba(L.design(zk))[:, 1]
                q4[k] = c4.predict_proba(np.column_stack([H.design2(zk), fk]))[:, 1]
                reach += int(((Hk[:, 0] + Hk[:, 2]) > 0).sum()); nk += int(k.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                pne = (b45 - q45 * offc) / (1 - q45)
            n4 = np.where(k, np.maximum(q4 * offc + (1 - q4) * np.minimum(pne, np.maximum(off - 3606, 0)), 0), b45)
        y = te["y"].to_numpy(); sb = np.sum(((y - b48) ** 2)[k]); sn = np.sum(((y - n4) ** 2)[k])
        tot_b += sb; tot_n += sn; better += sn < sb; per.append(sn / sb if sb > 0 else 1)
    lab = "control (v48 refit)" if gap is None else f"history gap >= {gap} mo"
    print(f"{lab:22s}: in-scope SSE v48 {tot_b:.4g} -> {tot_n:.4g} ({tot_n / tot_b:.3f}); months better {better}/12; "
          f"rows with a history {reach}/{nk}; per month {' '.join(f'{x:.2f}' for x in per)}", flush=True)
