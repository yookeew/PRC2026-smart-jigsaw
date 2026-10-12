"""Candidate corrector block 15 (PROGRESS.md §140; not wired into the corrector yet): takeoff-order and runway-configuration
state. Label-free: movement data of the frame only (takeoff, runway, LOBT, AOBT_3 of the other departures).

  dep: MVT_ID_mvt, apt, rwy, t, lo, a3 (epoch seconds: takeoff, LOBT, AOBT_3)

LOBT order (`lq_*`): departures on the same runway with |LOBT_j - LOBT| <= 20 min: count, overtakers (later LOBT, earlier
takeoff), overtaken (earlier LOBT, later takeoff), median takeoff gap, net share. AOBT_3 order (`aq_*`): the same at airport
level with AOBT_3. Configuration (`cf_*`): dominant departure runway per 15-min bin of the airport; time since the last
change and until the next (both capped at 3 h), whether the flight's runway was dominant 20 / 40 min before its takeoff,
and its runway's share of the takeoff bin. Same definitions as the screen (tests/flow_screen/dir_screen.py, families L, C).
"""
import numpy as np
import polars as pl

from post.flow_feats import _win_join

ORDER_COLS = ([f"lq_{c}" for c in ("n", "over", "under", "mdT", "net")] + [f"aq_{c}" for c in ("n", "over", "under", "mdT", "net")]
              + ["cf_since", "cf_until", "cf_dom20", "cf_dom40", "cf_share_now"])
W_ORD, BIN, CAP = 1200.0, 900, 10800.0


def _order(dep: pl.DataFrame, keys: list[str], col: str, tag: str) -> pl.DataFrame:
    a = dep.select("MVT_ID_mvt", *keys, "t", s=col).filter(pl.col("s").is_not_null())
    b = a.select(*keys, id_j="MVT_ID_mvt", tj="t", sj="s")
    out = []
    for ap in a["apt"].unique().sort().to_list():
        x = _win_join(a.filter(pl.col("apt") == ap), b.filter(pl.col("apt") == ap), keys, "s", "sj", W_ORD)
        out.append(x.group_by("MVT_ID_mvt").agg(**{
            f"{tag}_n": pl.len(), f"{tag}_over": ((pl.col("sj") > pl.col("s")) & (pl.col("tj") < pl.col("t"))).sum(),
            f"{tag}_under": ((pl.col("sj") < pl.col("s")) & (pl.col("tj") > pl.col("t"))).sum(),
            f"{tag}_mdT": (pl.col("tj") - pl.col("t")).median()}))
    return pl.concat(out).with_columns(**{f"{tag}_net": (pl.col(f"{tag}_over").cast(pl.Float64)
                                                          - pl.col(f"{tag}_under").cast(pl.Float64)) / pl.col(f"{tag}_n").cast(pl.Float64)})


def _config(dep: pl.DataFrame) -> pl.DataFrame:
    T, apt, rw = dep["t"].to_numpy(), dep["apt"].to_numpy(), dep["rwy"].to_numpy().astype(str)
    cs = {k: np.full(dep.height, np.nan) for k in ("cf_since", "cf_until", "cf_dom20", "cf_dom40", "cf_share_now")}
    for ap in np.unique(apt):
        ka = np.flatnonzero(apt == ap)
        bins = (T[ka] // BIN).astype(np.int64)
        t = pl.DataFrame({"b": bins, "r": rw[ka]}).group_by("b", "r").agg(c=pl.len()).sort(["b", "c", "r"], descending=[False, True, False])
        dom = t.unique("b", keep="first", maintain_order=True).sort("b")
        db, dr = dom["b"].to_numpy(), dom["r"].to_numpy().astype(str)
        cb = db[np.flatnonzero(dr[1:] != dr[:-1]) + 1] * BIN          # start of each new dominant-runway regime
        i = np.searchsorted(cb, T[ka], "right")
        cs["cf_since"][ka] = np.where(i > 0, T[ka] - cb[np.maximum(i - 1, 0)], np.nan)
        cs["cf_until"][ka] = np.where(i < len(cb), cb[np.minimum(i, len(cb) - 1)] - T[ka], np.nan)
        m = dict(zip(db.tolist(), dr.tolist()))
        for lag, k in ((1200, "cf_dom20"), (2400, "cf_dom40")):
            lb = ((T[ka] - lag) // BIN).astype(np.int64)
            cs[k][ka] = [float(m.get(x) == r_) if m.get(x) is not None else np.nan for x, r_ in zip(lb.tolist(), rw[ka].tolist())]
        sh = t.join(t.group_by("b").agg(n=pl.col("c").sum()), on="b").with_columns(s=pl.col("c") / pl.col("n"))
        shm = {(b_, r_): s_ for b_, r_, s_ in sh.select("b", "r", "s").iter_rows()}
        cs["cf_share_now"][ka] = [shm.get((x, r_), np.nan) for x, r_ in zip(bins.tolist(), rw[ka].tolist())]
    for k in ("cf_since", "cf_until"):
        cs[k] = np.minimum(cs[k], CAP)
    return pl.DataFrame(cs)


def build(dep: pl.DataFrame) -> pl.DataFrame:
    """One row per dep row (same order): MVT_ID_mvt + ORDER_COLS (Float64)."""
    dep = dep.select("MVT_ID_mvt", "apt", pl.col("rwy").fill_null("?"), "t", "lo", "a3")
    out = (dep.select("MVT_ID_mvt").with_row_index("_i")
           .join(_order(dep, ["apt", "rwy"], "lo", "lq"), on="MVT_ID_mvt", how="left")
           .join(_order(dep, ["apt"], "a3", "aq"), on="MVT_ID_mvt", how="left").sort("_i").drop("_i"))
    out = pl.concat([out, _config(dep)], how="horizontal")
    assert out.height == dep.height and out["MVT_ID_mvt"].n_unique() == dep.height
    return out.select("MVT_ID_mvt", *[pl.col(c).cast(pl.Float64) for c in ORDER_COLS])
