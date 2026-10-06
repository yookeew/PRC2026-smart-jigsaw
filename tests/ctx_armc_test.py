"""Gate for reports/ctx_armc_preregistration.md (context corrector Arm C) on v41's chain.

  gate : v41's chain (v30 method cross-fit Jan<->Jul, then LOBT window + nm_lines 10-month fit) with the base
         corrector inputs (ctxc0sel, same machine as the arms) vs ctx<arm>sel; LIRF NM-unmatched rows copied
         from the base after the chain. Every number of the pre-registration, plus C2 vs C1.
  oos  : out-of-sample check on the 10 training months (five OOF-block fits): per airport, months in which the
         corrector reduces SSE on NM-unmatched non-line rows vs the uncorrected stack.
  diag : 2026 ranking diagnostics from src/post/ctx_armc.py prod outputs (null-pattern support, shift per airport).

Run:  .venv/Scripts/python.exe tests/ctx_armc_test.py gate|oos|diag
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ["PRC_LGB"] = os.environ.get("ARMC_BASE", "ctxc0sel")

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import cluster_bootstrap  # noqa: E402
from post import nm_lines as NL  # noqa: E402

EVAL, OOF, SUB = ROOT / "cache" / "eval", ROOT / "cache" / "oof", ROOT / "data" / "submissions"
BASE = os.environ["PRC_LGB"]
ARMS = [a for a in os.environ.get("ARMC_ARMS", "c1,c2").split(",") if a]
STABLE = ["EHAM", "LSZH", "EDDF", "LEBL"]
TRIM, BODY = 18000, 3600
N_RES, SEED = 2000, 0


def swap(df: pl.DataFrame, ex: pl.DataFrame, name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """oof_context_corrector_stage2.main's input swap: holdout + OOF-block corrector predictions as the LightGBM slot."""
    ev = pl.read_parquet(EVAL / f"{name}_holdout_ev.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    df2 = df.join(ev, on="MVT_ID_mvt", how="left"); assert df2["lgb_new"].null_count() == 0
    df2 = df2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    folds = sorted((OOF / name).glob("fold=*.parquet")); assert len(folds) == 5
    oof = pl.concat([pl.read_parquet(p) for p in folds]).select(pl.col("MVT_ID_mvt").cast(pl.Int64), lgb_new=pl.col("pred"))
    ex2 = ex.join(oof, on="MVT_ID_mvt", how="left"); assert ex2["lgb_new"].null_count() == 0
    ex2 = ex2.with_columns(lgb=pl.col("lgb_new")).drop("lgb_new")
    w, _ = nnls(ex2.select("lgb", "cat").to_numpy(), ex2["taxi"].to_numpy())
    print(f"{name}: OOF stack weights {np.round(w, 3)}", flush=True)
    return df2, ex2.with_columns(s=pl.Series(ex2.select("lgb", "cat").to_numpy() @ w))


def boot(y, b, t, k, apt, day) -> tuple[float, float, float, float]:
    """(rmse base, rmse treatment, delta, P(worse)) on rows k, (airport, day) cluster bootstrap."""
    d = pl.DataFrame({"a": apt[k], "d": day[k], "sb": (b[k] - y[k]) ** 2, "st": (t[k] - y[k]) ** 2})
    cl = d.group_by("a", "d").agg(pl.col("sb").sum(), pl.col("st").sum(), n=pl.len())
    pt, _, _, pw = cluster_bootstrap(cl["sb"].to_numpy(), cl["st"].to_numpy(), cl["n"].to_numpy(), N_RES, SEED)
    return float(np.sqrt(d["sb"].mean())), float(np.sqrt(d["st"].mean())), pt, pw


def report(label: str, y, b, t, ctx: dict) -> dict:
    apt, day, ym, nm = ctx["apt"], ctx["day"], ctx["ym"], ctx["nm"]
    jan, jul = ym == "2025-01", ym == "2025-07"
    allr = np.ones(len(y), bool)
    print(f"\n=== {label} ===\n{'slice':30s} {'base':>9s} {'treat':>9s} {'delta':>7s} {'P(worse)':>9s}")
    res = {}
    for sl, k0 in (("body", (~nm) & (y <= BODY)), ("trimmed", y <= TRIM), ("full", allr)):
        for mo, km in (("pooled", allr), ("Jan", jan), ("Jul", jul)):
            r = boot(y, b, t, k0 & km, apt, day)
            res[(sl, mo)] = r
            print(f"{sl + ' ' + mo:30s} {r[0]:9.2f} {r[1]:9.2f} {r[2]:+7.2f} {r[3]:9.3f}")
    # days with the largest contribution to the full gain
    dd = pl.DataFrame({"d": day, "c": (t - y) ** 2 - (b - y) ** 2}).group_by("d").agg(pl.col("c").sum()).sort("c")
    top5 = dd["d"][:5].to_list()
    keep = ~np.isin(day, np.array(top5, dtype=day.dtype))
    rb, rt = np.sqrt(np.mean((b - y)[keep] ** 2)), np.sqrt(np.mean((t - y)[keep] ** 2))
    res["drop5"] = rt - rb
    print(f"{'full without top-5 gain days':30s} {rb:9.2f} {rt:9.2f} {rt - rb:+7.2f}   days {[str(x) for x in top5]}")
    # NM-unmatched, LIRF excluded
    nmu = nm & (apt != "LIRF")
    r = boot(y, b, t, nmu, apt, day)
    res["nmu"] = r
    print(f"{'NM-unmatched non-LIRF full':30s} {r[0]:9.2f} {r[1]:9.2f} {r[2]:+7.2f} {r[3]:9.3f}   n={nmu.sum():,}")
    for a in sorted(set(apt[nmu])):
        k = nmu & (apt == a)
        rb_, rt_ = np.sqrt(np.mean((b - y)[k] ** 2)), np.sqrt(np.mean((t - y)[k] ** 2))
        print(f"   {a:27s} {rb_:9.2f} {rt_:9.2f} {rt_ - rb_:+7.2f}   n={k.sum():,}")
    k = nm & (apt == "LIRF")
    print(f"{'LIRF NM-unmatched (copied)':30s} max |t - b| = {np.max(np.abs(t - b)[k]) if k.any() else 0:.3g}   n={k.sum():,}")
    print("per airport, trimmed:")
    for a in sorted(set(apt)):
        k = (apt == a) & (y <= TRIM)
        print(f"   {a:27s} {np.sqrt(np.mean((b - y)[k] ** 2)):9.2f} {np.sqrt(np.mean((t - y)[k] ** 2)):9.2f} "
              f"{np.sqrt(np.mean((t - y)[k] ** 2)) - np.sqrt(np.mean((b - y)[k] ** 2)):+7.2f}")
    rules = {
        "1 pooled trimmed < 0, P(worse) < 0.05": res[("trimmed", "pooled")][2] < 0 and res[("trimmed", "pooled")][3] < 0.05,
        "2 pooled full < 0": res[("full", "pooled")][2] < 0,
        "3 no month worse in full by > 0.10": max(res[("full", "Jan")][2], res[("full", "Jul")][2]) <= 0.10,
        "4 full < 0 without top-5 gain days": res["drop5"] < 0,
        "5 NM-unmatched non-LIRF P(worse) < 0.9": res["nmu"][3] < 0.9,
    }
    for k_, v in rules.items():
        print(f"  rule {k_:42s} {'PASS' if v else 'FAIL'}")
    res["pass"] = all(rules.values())
    print(f"VERDICT {label}: {'PASS' if res['pass'] else 'FAIL'}")
    return res


def gate() -> None:
    import base_neighbour_chain_test as B
    import neighbour_state_test as N
    df = B.A.load()
    ex, lag = N.train_extra()
    paths = sorted(N.RAW.glob("training_*.parquet"))
    arr = N.arrivals(paths, paths)
    raw = NL.frame(str(ROOT / "data" / "raw" / "training_*.parquet"))
    prm = NL.fit(raw.filter(~pl.col("m").is_in([1, 7])))
    h = df.select("MVT_ID_mvt").join(raw, on="MVT_ID_mvt", how="left")
    assert h.height == df.height and h["nm"].null_count() == 0
    lt = h["lt"].to_numpy()
    fin = lambda q: NL.apply(h, np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606)), prm)[0]  # noqa: E731
    print(f"== base: {BASE}", flush=True)
    base = fin(B.v30_method(df, ex, lag, arr))
    y = df["taxi"].to_numpy().astype(np.float64)
    ctx = dict(apt=df["ADEP_mvt"].to_numpy(), day=df["day"].to_numpy(), ym=df["ym"].to_numpy(), nm=h["nm"].to_numpy())
    lirf_nmu = ctx["nm"] & (ctx["apt"] == "LIRF")
    r = lambda q, k: float(np.sqrt(np.mean((y - q)[k] ** 2)))  # noqa: E731
    print(f"base chain: body {r(base, (~ctx['nm']) & (y <= BODY)):.2f}  trimmed {r(base, y <= TRIM):.2f}  full {r(base, np.ones(len(y), bool)):.2f}"
          f"  (teammate's ctxsel gate: 194.13 / 246.06 / 304.77)", flush=True)
    outs = {"base": base}
    for arm in ARMS:
        name = f"ctx{arm}sel"
        print(f"== treatment: {name}", flush=True)
        df2, ex2 = swap(df, ex, name)
        t = fin(B.v30_method(df2, ex2, lag, arr))
        t[lirf_nmu] = base[lirf_nmu]          # team rule: LIRF NM-unmatched rows keep the base value
        outs[arm] = t
    pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], **{k: v for k, v in outs.items()}}).write_parquet(EVAL / "armc_gate_holdout.parquet")
    res = {arm: report(f"{arm.upper()} vs BASE ({BASE})", y, base, outs[arm], ctx) for arm in ARMS}
    if "c1" in outs and "c2" in outs:
        k = y <= TRIM
        rb, rt, pt, pw = boot(y, outs["c1"], outs["c2"], k, ctx["apt"], ctx["day"])
        print(f"\nC2 vs C1, pooled trimmed: {rb:.2f} -> {rt:.2f}  delta {pt:+.2f}  P(worse) {pw:.3f}")
        pick = None
        if res["c1"]["pass"] and res["c2"]["pass"]:
            pick = "c2" if (pt <= -0.2 and pw < 0.05) else "c1"
        elif res["c1"]["pass"] or res["c2"]["pass"]:
            pick = "c1" if res["c1"]["pass"] else "c2"
        print(f"DECISION: {('ADOPT ' + pick.upper()) if pick else 'REJECT both arms'}")


def _stack_oof() -> pl.DataFrame:
    """Stack s on OOF-month rows, as oof_context_corrector_test.prepare (no feature build)."""
    import oof_context_corrector_test as O
    raw = NL.frame(str(ROOT / "data" / "raw" / "training_*.parquet"))
    pr = O.preds().join(raw.select("MVT_ID_mvt", "ADEP_mvt", "lt", "d", "nm", "y", "m"), on="MVT_ID_mvt", how="inner")
    o = pr.filter(pl.col("split") == "oof")
    w, _ = nnls(o.select("lgb", "cat").to_numpy(), o["y"].to_numpy())
    lt = o["lt"].to_numpy()
    q = o.select("lgb", "cat").to_numpy() @ w
    s = np.where(np.isnan(lt), q, np.clip(q, np.clip(lt - 3606, 0, None), lt + 3606))
    line = o["nm"].to_numpy() & (np.nan_to_num(o["d"].to_numpy(), nan=-np.inf) > NL.LINE_MIN_D)
    return o.select("MVT_ID_mvt", "ADEP_mvt", "nm", "y", "m").with_columns(s=pl.Series(s), line=pl.Series(line))


def oos() -> None:
    o = _stack_oof().filter(pl.col("nm") & ~pl.col("line") & pl.col("y").is_not_null())
    print(f"NM-unmatched non-line OOF rows: {o.height:,}")
    names = [n for n in ("ctxsel", "ctxc0sel", *[f"ctx{a}sel" for a in ARMS]) if (OOF / n).exists()]
    for n in names:
        p = pl.concat([pl.read_parquet(f) for f in sorted((OOF / n).glob("fold=*.parquet"))]).select(
            pl.col("MVT_ID_mvt").cast(pl.Int64), c=pl.col("pred"))
        x = o.join(p, on="MVT_ID_mvt", how="inner")
        g = (x.with_columns(dsse=(pl.col("c") - pl.col("y")) ** 2 - (pl.col("s") - pl.col("y")) ** 2)
             .group_by("ADEP_mvt", "m").agg(pl.col("dsse").sum()))
        a = g.group_by("ADEP_mvt").agg(better=(pl.col("dsse") < 0).sum(), months=pl.len(),
                                       dsse_M=pl.col("dsse").sum() / 1e6).sort("ADEP_mvt")
        sh = (x["c"] - x["s"]).to_numpy()
        print(f"\n{n}: corrector vs stack on NM-unmatched rows; pooled dSSE {g['dsse'].sum() / 1e6:+.0f}M; "
              f"shift mean {sh.mean():+.1f} s RMS {np.sqrt(np.mean(sh ** 2)):.1f} s")
        for row in a.iter_rows(named=True):
            print(f"   {row['ADEP_mvt']}  better in {row['better']:2d}/{row['months']:2d} months  dSSE {row['dsse_M']:+9.1f}M"
                  f"{'   >= 7/10' if row['better'] >= 7 else ''}")


def diag() -> None:
    """2026 ranking: null-pattern support and corrector shift on NM-unmatched served rows, BASE vs arms."""
    rk = {}
    for arm in ARMS:
        parts = []
        for mem in ("r", "rb"):
            f = EVAL / f"armc_rank_ctx{arm}{mem}.parquet"
            if not f.exists():
                print(f"missing {f.name}; run src/post/ctx_armc.py prod first"); return
            d = pl.read_parquet(f)
            parts.append(d.filter(pl.col("apt").is_in(STABLE) if mem == "rb" else ~pl.col("apt").is_in(STABLE)))
        rk[arm] = pl.concat(parts)
    # BASE shift: v41's corrector files (ctxcorr2r = teammate's ranking file, ctxcorr2rb = rebuilt here) minus the stack
    old = pl.read_parquet(SUB / "ctxsel.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), old=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    for arm, d in rk.items():
        x = d.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(old, on="MVT_ID_mvt", how="left").filter(pl.col("nm") & pl.col("app"))
        print(f"\n{arm.upper()}: NM-unmatched served ranking rows {x.height:,}; unseen null pattern "
              f"BASE {x['unseen_base'].mean():.3f} -> {arm} {x['unseen_arm'].mean():.3f}")
        print(f"   {'apt':5s} {'n':>6s}  {'BASE mean':>9s} {'RMS':>7s}  {arm + ' mean':>9s} {'RMS':>7s}  unseen BASE->arm")
        g = x.with_columns(so=pl.col("old") - pl.col("s"), sn=pl.col("out") - pl.col("s")).group_by("apt").agg(
            n=pl.len(), om=pl.col("so").mean(), orms=(pl.col("so") ** 2).mean().sqrt(),
            nmn=pl.col("sn").mean(), nrms=(pl.col("sn") ** 2).mean().sqrt(),
            ub=pl.col("unseen_base").mean(), ua=pl.col("unseen_arm").mean()).sort("apt")
        for r in g.iter_rows(named=True):
            print(f"   {r['apt']:5s} {r['n']:6d}  {r['om']:+9.1f} {r['orms']:7.1f}  {r['nmn']:+9.1f} {r['nrms']:7.1f}  {r['ub']:.2f}->{r['ua']:.2f}")


if __name__ == "__main__":
    {"gate": gate, "oos": oos, "diag": diag}[sys.argv[1]]()
