"""Flow-state inputs, block 13 of the context corrector (reports/ctx_flow_preregistration.md; PROGRESS.md §137).

Label-free: movement data only (takeoff, SOBT, LOBT, ADES of the other departures in the same frame), so the same code
serves 2025, the Jan/Jul 2026 board and the four-month blind board. Every frame holds all ten origins.

  dep: MVT_ID_mvt, apt, ades, t, s, lo (epoch seconds: takeoff, SOBT, LOBT)

Destination flow (other origins +-60 min, same origin +-90 min), region flow (same two-letter ICAO prefix of ADES, a
different ADES, +-60 min, same / other origins) and schedule-order displacement at the airport (SOBT within +-20 min).
Rows without neighbours get null; a null ADES never joins.
"""
import polars as pl

STATS = ["n", "moff", "mdl", "f30"]
FLOW_COLS = ([f"xo_{c}" for c in STATS] + [f"so_{c}" for c in STATS] + [f"rso_{c}" for c in STATS]
             + [f"rxo_{c}" for c in STATS] + ["sq_n", "sq_over", "sq_under", "sq_mdT", "sq_net"])
W_XO, W_SO, W_REG, W_SQ = 3600.0, 5400.0, 3600.0, 1200.0


def _win_join(a: pl.DataFrame, b: pl.DataFrame, keys: list[str], ta: str, tb: str, w: float) -> pl.DataFrame:
    """Pairs (a row i, b row j != i) with equal keys and |tb_j - ta_i| <= w, via w-wide time buckets."""
    a = a.with_columns(_bk=(pl.col(ta) // w).cast(pl.Int64))
    b = b.with_columns(_bk0=(pl.col(tb) // w).cast(pl.Int64))
    x = pl.concat([a.join(b.with_columns(_bk=pl.col("_bk0") - k), on=keys + ["_bk"], how="inner") for k in (-1, 0, 1)])
    return x.filter((pl.col("id_j") != pl.col("MVT_ID_mvt")) & ((pl.col(tb) - pl.col(ta)).abs() <= w))


def _stats(x: pl.DataFrame, tag: str) -> pl.DataFrame:
    return x.group_by("MVT_ID_mvt").agg(**{f"{tag}_n": pl.len(), f"{tag}_moff": pl.col("off_j").median(),
                                           f"{tag}_mdl": pl.col("dl_j").median(),
                                           f"{tag}_f30": (pl.col("off_j") > 1800).mean()})


def build(dep: pl.DataFrame) -> pl.DataFrame:
    dep = dep.select("MVT_ID_mvt", "apt", "ades", "t", "s", "lo").with_columns(
        reg=pl.col("ades").str.slice(0, 2), off=pl.col("t") - pl.col("s"), dl=pl.col("t") - pl.col("lo"))
    b = dep.select(id_j="MVT_ID_mvt", apt_j="apt", ades_j="ades", reg="reg", ades="ades", tj="t", sj="s",
                   off_j="off", dl_j="dl")
    parts = []
    for a_ in dep["apt"].unique().sort().to_list():   # one origin at a time keeps the pair frames small
        a = dep.filter(pl.col("apt") == a_).select("MVT_ID_mvt", "apt", "ades", "reg", "t", "s")
        x = _win_join(a, b.drop("reg"), ["ades"], "t", "tj", W_SO)
        xo = _stats(x.filter((pl.col("apt_j") != pl.col("apt")) & ((pl.col("tj") - pl.col("t")).abs() <= W_XO)), "xo")
        so = _stats(x.filter(pl.col("apt_j") == pl.col("apt")), "so")
        x = _win_join(a.rename({"ades": "ades_i"}), b.drop("ades"), ["reg"], "t", "tj", W_REG).filter(
            pl.col("ades_j") != pl.col("ades_i"))
        rso = _stats(x.filter(pl.col("apt_j") == pl.col("apt")), "rso")
        rxo = _stats(x.filter(pl.col("apt_j") != pl.col("apt")), "rxo")
        x = _win_join(a.select("MVT_ID_mvt", "apt", "t", "s"), b.filter(pl.col("apt_j") == a_).select("id_j", "tj", "sj"),
                      [], "s", "sj", W_SQ)
        sq = x.group_by("MVT_ID_mvt").agg(
            sq_n=pl.len(), sq_over=((pl.col("sj") > pl.col("s")) & (pl.col("tj") < pl.col("t"))).sum(),
            sq_under=((pl.col("sj") < pl.col("s")) & (pl.col("tj") > pl.col("t"))).sum(),
            sq_mdT=(pl.col("tj") - pl.col("t")).median())
        o = a.select("MVT_ID_mvt")
        for g in (xo, so, rso, rxo, sq):
            o = o.join(g, on="MVT_ID_mvt", how="left")
        parts.append(o)
    out = pl.concat(parts, how="diagonal_relaxed").with_columns(
        sq_net=(pl.col("sq_over").cast(pl.Float64) - pl.col("sq_under").cast(pl.Float64)) / pl.col("sq_n").cast(pl.Float64))
    assert out.height == dep.height and out["MVT_ID_mvt"].n_unique() == dep.height
    return out.select("MVT_ID_mvt", *[pl.col(c).cast(pl.Float64) for c in FLOW_COLS])
