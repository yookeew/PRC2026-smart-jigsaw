"""Context corrector Arm C (reports/ctx_armc_preregistration.md).

The BASE corrector recipe (tests/context_corrector_test.py, ensemble2, L2) with:
  C1: NM-unmatched rows added to training (nm & ~line_rows & stack s <= 7,200 s -- selected on the
      prediction, never on the label) plus an `nm_unmatched` input;
  C2: C1 plus the operator echo block (ec_rate, ec_n): per (airport, AIRCRAFT_OPERATOR_flt), among
      labelled departures with T - SCHED >= 30 min, the share with |BLOCK - SCHED| < 30 s; out of fold by month.

The member (v37's `r` or v40's `rb`) comes from the teammate's env flags, read when context_corrector_test
is imported, so each member runs in its own process:
  ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1             -> ctxc1r
  ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_ADSB=1  -> ctxc1rb

Commands:
  inputs : holdout (fit on the 10 OOF months) + five two-month OOF blocks, as oof_context_corrector_stage2.build_inputs
           -> cache/eval/<name>_holdout_ev.parquet (with the chain columns), cache/oof/<name>/fold=*.parquet
  prod   : 12-month production fit, as src/post/ctx_corrector.py -> data/submissions/<name>.parquet,
           plus cache/eval/armc_rank_<name>.parquet (stack, corrected value, null-pattern flags) for diagnostics
  c0 is the BASE recipe unchanged (same code path and machine as the arms: the gate's base).
  sel    : the ctxsel combination (rb at EHAM/LSZH/EDDF/LEBL, r elsewhere) of an arm's two members -> ctx<arm>sel

Run:  ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 [CTX_ADSB=1] .venv/Scripts/python.exe src/post/ctx_armc.py inputs|prod
      ARMC_ARM=c1 .venv/Scripts/python.exe src/post/ctx_armc.py sel
"""

from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

ARM = os.environ.get("ARMC_ARM", "")
assert ARM in ("c0", "c1", "c2"), "ARMC_ARM must be c0 (the BASE recipe, same machine), c1 or c2"
MEMBER = "rb" if os.environ.get("CTX_ADSB") == "1" else "r"
BASE = os.environ.get("PRC_BASE", "lgb_mixed_nb")   # reports/base_comb_preregistration.md; read by O.prepare() too
SUFFIX = ("" if BASE == "lgb_mixed_nb" else "_" + BASE.removeprefix("lgb_").removesuffix("_nb")) + os.environ.get("PRC_ARMC_TAG", "")
# PRC_CELN_SCALE=1 (PROGRESS §105): training rows' cell tables leave their own month out (11 months), the ranking
# rows' table uses all 12, so 2026 cel_n runs ~12/11 above anything the corrector saw; scale it back by 11/12.
CELN_SCALE = os.environ.get("PRC_CELN_SCALE") == "1"
# PRC_CTX_SEEDS=42,43,44 (reports/ctx_seedavg_preregistration.md): the corrector is fitted once per LightGBM seed and the
# corrections averaged. Default (unset): one fit with the recipe's seed 42, unchanged. Outputs get "_s<K>".
SEEDS = [int(x) for x in os.environ.get("PRC_CTX_SEEDS", "").split(",") if x]
if SEEDS:
    SUFFIX += f"_s{len(SEEDS)}"


def fit_ens(*a):
    """CC.fit_apply_ens, averaged over SEEDS when set."""
    import context_corrector_test as CC
    if not SEEDS:
        return CC.fit_apply_ens(*a)
    p0, pa0 = dict(CC.PARAMS), dict(CC.PARAMS_APT)
    cs, bs = [], []
    try:
        for sd in SEEDS:
            CC.PARAMS, CC.PARAMS_APT = {**p0, "seed": sd}, {**pa0, "seed": sd}
            c, b = CC.fit_apply_ens(*a)
            cs.append(c); bs.append(b)
    finally:
        CC.PARAMS, CC.PARAMS_APT = p0, pa0
    print(f"  seeds {SEEDS}: best_iter {bs}; RMS between seed fits {np.sqrt(np.mean((cs[0] - cs[-1]) ** 2)):.2f}", flush=True)
    return np.mean(cs, axis=0), bs[0]
NAME = f"ctx{ARM}{MEMBER}{SUFFIX}"
SEL = f"ctx{ARM}sel{SUFFIX}"
EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"
RANKING = ROOT / "data" / "ranking" / "ranking.parquet"
TEMPLATE = ROOT / "data" / "ranking" / "submitting.parquet"
NMU_PRED_MAX = 7200.0      # NM-unmatched training rows: stack prediction <= 2 h
ECHO_LATE, ECHO_TOL = 1800.0, 30.0
HOLDOUT_MONTHS = [1, 7]
BLOCKS = [(2, 3), (4, 5), (6, 8), (9, 10), (11, 12)]


# ---------------------------------------------------------------- arm features and training mask

def echo_table(dep: pl.DataFrame, excl: list[int]) -> pl.DataFrame:
    """(apt, op, month) sums of echo and delayed-departure counts over labelled rows, months in excl left out."""
    return (dep.filter(pl.col("y").is_not_null() & (pl.col("op") != "?") & ~pl.col("m").is_in(excl)
                       & ((pl.col("T") - pl.col("t_SCHED_TIME_UTC_mvt")) >= ECHO_LATE))
            .with_columns(e=((pl.col("T") - pl.col("t_SCHED_TIME_UTC_mvt") - pl.col("y")).abs() < ECHO_TOL).cast(pl.Float64))
            .group_by("apt", "op", "m").agg(se=pl.col("e").sum(), n=pl.len().cast(pl.Float64)))


def echo_block(dep: pl.DataFrame, g: pl.DataFrame, lomo: bool) -> pl.DataFrame:
    """ec_rate / ec_n per row; lomo removes the row's own month (training rows only, never 2026 rows)."""
    tot = g.group_by("apt", "op").agg(SE=pl.col("se").sum(), N=pl.col("n").sum())
    x = dep.select("apt", "op", "m").with_row_index("_i").join(tot, on=["apt", "op"], how="left")
    if lomo:
        x = x.join(g, on=["apt", "op", "m"], how="left").with_columns(
            SE=pl.col("SE") - pl.col("se").fill_null(0), N=pl.col("N") - pl.col("n").fill_null(0))
    x = x.sort("_i")
    assert x.height == dep.height
    known = pl.col("op") != "?"
    return x.select(ec_rate=pl.when(known & (pl.col("N").fill_null(0) > 0)).then(pl.col("SE") / pl.col("N")),
                    ec_n=pl.when(known).then(pl.col("N").fill_null(0)))


def add_arm(X: pl.DataFrame, dep: pl.DataFrame, echo: pl.DataFrame | None) -> pl.DataFrame:
    if ARM == "c0":
        return X
    X = X.with_columns(nm_unmatched=dep["nm"].cast(pl.Float64))
    if ARM == "c2":
        assert echo is not None and echo.height == X.height
        X = pl.concat([X, echo], how="horizontal")
    return X


def arm_fit(base_fit: np.ndarray, nm: np.ndarray, line_rows: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    if ARM == "c0":
        return base_fit
    return base_fit | (nm & ~line_rows & (s <= NMU_PRED_MAX) & ~np.isnan(y))


def null_patterns(X: np.ndarray) -> np.ndarray:
    """One hashable key per row: its pattern of null inputs."""
    b = np.packbits(np.isnan(X), axis=1)
    return np.ascontiguousarray(b).view(np.dtype((np.void, b.shape[1]))).ravel()


# ---------------------------------------------------------------- commands

def inputs() -> None:
    import context_corrector_test as CC
    import oof_context_corrector_test as O
    P = O.prepare()
    dep, s, y, m, nm, lr, is_ho, inner = (P[k] for k in ("dep", "s", "y", "m", "nm", "line_rows", "is_ho", "inner"))
    echo = echo_block(dep, echo_table(dep, HOLDOUT_MONTHS), lomo=True) if ARM == "c2" else None
    Xdf = add_arm(P["X_full"], dep, echo)
    cols = Xdf.columns
    X = Xdf.to_numpy().astype(np.float64)
    cats = [cols.index(c) for c in P["catn"]]
    base_fit = (~nm) & (y <= O.TRIM) & ~lr
    trainable = (~is_ho) & arm_fit(base_fit, nm, lr, s, y)
    print(f"{NAME}: features {len(cols)}; trainable rows {trainable.sum():,} "
          f"(BASE {((~is_ho) & base_fit).sum():,}, NM-unmatched added {(trainable & nm).sum():,})", flush=True)
    if ARM == "c2":
        e = Xdf["ec_rate"].to_numpy()
        print(f"  ec_rate null share {np.isnan(e).mean():.3f}; mean {np.nanmean(e):.3f}", flush=True)
    fa = lambda *a: fit_ens(*a, P["apt"])  # noqa: E731
    out = s.copy()
    app = is_ho & ~lr
    c, best = fa(X, y - s, trainable, inner, app, cats)
    out[app] = s[app] + c
    print(f"holdout corrector: best_iter {best}", flush=True)
    for blk in BLOCKS:
        inb = np.isin(m, blk) & ~is_ho
        c, best = fa(X, y - s, trainable & ~inb, inner, inb & ~lr, cats)
        out[inb & ~lr] = s[inb & ~lr] + c
        print(f"OOF block {blk}: best_iter {best}", flush=True)
    out = P["win"](np.clip(out, 0, None))
    ids = dep["MVT_ID_mvt"]
    ev = pl.DataFrame({"MVT_ID_mvt": ids.filter(pl.Series(is_ho)).cast(pl.Float64), "pred": out[is_ho]})
    chain = pl.read_parquet(EVAL / f"{BASE}_holdout_ev.parquet").drop("pred")
    ev = chain.join(ev, on="MVT_ID_mvt", how="inner")   # as ctx_corrector.py's "rewrote ... with the chain columns"
    assert ev.height == int(is_ho.sum()), (ev.height, int(is_ho.sum()))
    ev.write_parquet(EVAL / f"{NAME}_holdout_ev.parquet")
    d = OOF / NAME
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.parquet"):
        f.unlink()
    for blk in BLOCKS:
        k = np.isin(m, blk) & ~is_ho
        pl.DataFrame({"MVT_ID_mvt": ids.filter(pl.Series(k)), "pred": out[k]}).write_parquet(d / f"fold={blk[0]:02d}-{blk[1]:02d}.parquet")
    print(f"wrote {NAME}_holdout_ev.parquet and {d.relative_to(ROOT)}")


def prod() -> None:
    """src/post/ctx_corrector.py (CTX_ENS=2) with the arm's training rows and inputs."""
    import context_corrector_test as CC
    from post import nm_lines as NL
    rd = lambda path, c, col="pred": pl.read_parquet(path).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col(col).cast(pl.Float64).alias(c))  # noqa: E731
    oof = rd(OOF / BASE / "*.parquet", "lgb").join(rd(OOF / "cat_mixed" / "*.parquet", "cat"), on="MVT_ID_mvt")
    ho = rd(EVAL / f"{BASE}_holdout_ev.parquet", "lgb").join(rd(EVAL / "cat_mixed_holdout_ev.parquet", "cat"), on="MVT_ID_mvt")
    lab = NL.frame(str(CC.RAW / "training_*.parquet")).select("MVT_ID_mvt", "y")
    oy = oof.join(lab, on="MVT_ID_mvt")
    w, _ = nnls(oy.select("lgb", "cat").to_numpy(), oy["y"].to_numpy())
    print("stack weights (OOF months):", np.round(w, 3))
    tr = pl.concat([oof, ho])
    dep_all, arr_all = CC.movements(str(CC.RAW / "training_*.parquet"))
    dep = tr.select("MVT_ID_mvt").join(dep_all, on="MVT_ID_mvt", how="inner")
    tr = dep.select("MVT_ID_mvt").join(tr, on="MVT_ID_mvt", how="left")
    lt = (dep["T"] - dep["t_LOBT_flt"]).to_numpy()
    win = lambda q, lt=lt: np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))  # noqa: E731
    s = win(tr.select("lgb", "cat").to_numpy() @ w)
    arr = arr_all.filter(pl.col("L").is_not_null())
    X, _, catn = CC.build(dep, arr, s, CC.cell_table(dep_all))
    m = dep["m"].to_numpy()
    cel_cols = ["cel_p50", "cel_p90", "cel_sd", "cel_n"]
    cel = X.select(cel_cols).to_numpy().copy()
    for mm in sorted(set(m)):
        k = m == mm
        cel[k] = CC.apply_cells(dep.filter(pl.Series(k)), CC.cell_table(dep_all.filter(pl.col("m") != mm))).to_numpy()
    X = X.with_columns([pl.Series(c, cel[:, i]) for i, c in enumerate(cel_cols)]).with_columns(cel_rel=pl.Series(s) - pl.col("cel_p50"))
    base_cols = X.columns
    g_echo = echo_table(dep_all, []) if ARM == "c2" else None
    X = add_arm(X, dep, echo_block(dep, g_echo, lomo=True) if ARM == "c2" else None)
    cols = X.columns
    y = dep["y"].to_numpy(); nm = dep["nm"].to_numpy()
    d = (dep["T"] - dep["t_SCHED_TIME_UTC_mvt"]).to_numpy()
    line_rows = nm & (np.nan_to_num(d, nan=-np.inf) > NL.LINE_MIN_D)
    base_fit = (~nm) & (y <= CC.TRIM) & ~line_rows
    fit = arm_fit(base_fit, nm, line_rows, s, y)
    inner = (dep["T"].to_numpy() // 86400).astype(int) % 5 == 0
    print(f"training rows {fit.sum():,} of {len(y):,} (BASE {base_fit.sum():,}; NM-unmatched added {(fit & nm).sum():,}); "
          f"features {len(cols)}", flush=True)

    # ranking rows
    rk_l = pl.read_parquet(SUB / f"{BASE}.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    rk_c = pl.read_parquet(SUB / "cat_mixed.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), cat=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    dep_r, arr_r = CC.movements(str(RANKING))
    rk = dep_r.select("MVT_ID_mvt").join(rk_l, on="MVT_ID_mvt", how="left").join(rk_c, on="MVT_ID_mvt", how="left")
    assert rk["lgb"].null_count() == 0 and rk["cat"].null_count() == 0
    lt_r = (dep_r["T"] - dep_r["t_LOBT_flt"]).to_numpy()
    s_r = win(rk.select("lgb", "cat").to_numpy() @ w, lt_r)
    Xr, _, _ = CC.build(dep_r, arr_r.filter(pl.col("L").is_not_null()), s_r, CC.cell_table(dep_all))
    Xr = Xr.with_columns(cel_rel=pl.Series(s_r) - pl.col("cel_p50")).select(base_cols)
    if CELN_SCALE:
        Xr = Xr.with_columns(cel_n=pl.col("cel_n") * (11 / 12))
        print("cel_n on ranking rows scaled by 11/12 (PRC_CELN_SCALE)")
    Xr = add_arm(Xr, dep_r, echo_block(dep_r, g_echo, lomo=False) if ARM == "c2" else None).select(cols)
    nm_r = dep_r["nm"].to_numpy(); d_r = (dep_r["T"] - dep_r["t_SCHED_TIME_UTC_mvt"]).to_numpy()
    app = ~(nm_r & (np.nan_to_num(d_r, nan=-np.inf) > NL.LINE_MIN_D))

    Xa = X.to_numpy().astype(np.float64)
    Xra = Xr.to_numpy().astype(np.float64)
    # null-pattern support on NM-unmatched ranking rows: BASE inputs + BASE rows vs arm inputs + arm rows
    nb = len(base_cols)
    seen_b = set(null_patterns(Xa[base_fit][:, :nb]).tolist())
    seen_a = set(null_patterns(Xa[fit]).tolist())
    pr_b, pr_a = null_patterns(Xra[:, :nb]), null_patterns(Xra)
    unseen_b = np.array([p not in seen_b for p in pr_b.tolist()])
    unseen_a = np.array([p not in seen_a for p in pr_a.tolist()])
    k = nm_r & app
    print(f"NM-unmatched ranking rows served: {k.sum():,}; unseen null pattern: BASE {unseen_b[k].mean():.3f} -> {NAME} {unseen_a[k].mean():.3f}")

    cats = [cols.index(c) for c in catn]
    Xb = np.vstack([Xa, Xra])
    n_tr, n_r = len(y), Xr.height
    z = np.zeros(n_r, bool)
    apt = np.concatenate([dep["apt"].to_numpy(), dep_r["apt"].to_numpy()])
    corr, best = fit_ens(Xb, np.concatenate([y - s, np.zeros(n_r)]), np.concatenate([fit, z]),
                                  np.concatenate([inner, z]), np.concatenate([np.zeros(n_tr, bool), app]), cats, apt)
    print(f"corrector (ensemble2): global best_iter {best}")
    out = s_r.copy()
    out[app] = s_r[app] + corr
    out = win(np.clip(out, 0, None), lt_r)
    print(f"ranking: RMS change vs stack {np.sqrt(np.mean((out - s_r) ** 2)):.1f} s; mean {np.mean(out - s_r):+.1f} s")
    sub = pl.DataFrame({"MVT_ID_mvt": dep_r["MVT_ID_mvt"], "TAXITIME_SEC_mvt": np.round(out).astype(np.int32)})
    tpl = pl.read_parquet(TEMPLATE)
    sub = tpl.select("MVT_ID_mvt").join(sub.with_columns(pl.col("MVT_ID_mvt").cast(tpl["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
    assert sub.height == tpl.height and sub["TAXITIME_SEC_mvt"].null_count() == 0
    sub.write_parquet(SUB / f"{NAME}.parquet")
    pl.DataFrame({"MVT_ID_mvt": dep_r["MVT_ID_mvt"], "apt": dep_r["apt"], "nm": nm_r, "app": app, "s": s_r, "out": out,
                  "unseen_base": unseen_b, "unseen_arm": unseen_a}).write_parquet(EVAL / f"armc_rank_{NAME}.parquet")
    print(f"wrote {NAME}.parquet and armc_rank_{NAME}.parquet")


def sel() -> None:
    """ctxsel's per-airport combination (tests/ctx_adsb_stable_test.py) of this arm's members."""
    import ctx_adsb_stable_test as ST
    r, rb = f"ctx{ARM}r{SUFFIX}", f"ctx{ARM}rb{SUFFIX}"
    apt = ST._airports()
    ev, n = ST._select(pl.read_parquet(EVAL / f"{r}_holdout_ev.parquet"), pl.read_parquet(EVAL / f"{rb}_holdout_ev.parquet"), "pred", apt)
    ev.write_parquet(EVAL / f"{SEL}_holdout_ev.parquet")
    print(f"holdout: {n:,} of {ev.height:,} rows from {rb}")
    (OOF / SEL).mkdir(parents=True, exist_ok=True)
    folds = sorted(p.name for p in (OOF / r).glob("fold=*.parquet"))
    assert len(folds) == 5
    for f in folds:
        o, n = ST._select(pl.read_parquet(OOF / r / f), pl.read_parquet(OOF / rb / f), "pred", apt)
        o.write_parquet(OOF / SEL / f)
    if (SUB / f"{r}.parquet").exists() and (SUB / f"{rb}.parquet").exists():
        rk, n = ST._select(pl.read_parquet(SUB / f"{r}.parquet"), pl.read_parquet(SUB / f"{rb}.parquet"), "TAXITIME_SEC_mvt", apt)
        rk.write_parquet(SUB / f"{SEL}.parquet")
        print(f"ranking: {n:,} of {rk.height:,} rows from {rb} -> {SEL}.parquet")


if __name__ == "__main__":
    {"inputs": inputs, "prod": prod, "sel": sel}[sys.argv[1]]()
