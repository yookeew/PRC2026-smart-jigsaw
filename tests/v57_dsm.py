"""v57 day-sum drift model, validations (reports/v57_daysum_model_preregistration.md): C1 and C3.

  1. 2026 (label-free): leave-one-day-out prediction of lam_g per family -> F*; input check of 2026 shares vs 2025.
  2. 2025 (labels, decides): all 12 months in six two-month periods, each period running the procedure on its own
     published sums; per-row RMSE vs R0 (v55's recipe) on the 8 airports (C1) and vs no change on LSZH (C3).
Writes cache/v57/dsm_params.json (F*, s2 per airport from all 2025 periods) for src/post/v57_dsm_build.py.

Run:  python tests/v57_dsm.py
"""
import os, sys, glob, json
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "src" / "post"))
from post import pru_recon2 as R  # noqa: E402
from post import pru_dsm as M  # noqa: E402
import v57_recon2_dev as DV  # noqa: E402

A8 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LTFM"]
PERIODS = {"P01": ("2025-01", "2025-07"), "P02": ("2025-02", "2025-03"), "P04": ("2025-04", "2025-05"),
           "P06": ("2025-06", "2025-08"), "P09": ("2025-09", "2025-10"), "P11": ("2025-11", "2025-12")}
PER_OF = {m: k for k, v in PERIODS.items() for m in v}
OUT = ROOT / "cache" / "v57"


def obs_flags() -> pl.DataFrame:
    det = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_tier"]) for p in
                     sorted((ROOT / "cache/adsb_pushback_v4").glob("day=*.parquet"))], how="diagonal").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64))
    of = pl.read_parquet(ROOT / "cache/opdi_feats/opdi_feats.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "o_lead_first")
    return det.join(of, on="MVT_ID_mvt", how="full", coalesce=True).select(
        "MVT_ID_mvt", trk=pl.col("adsb_tier").is_in(list(R.TRACKED)).fill_null(False),
        opdi=(pl.col("o_lead_first").fill_null(0) >= 60))


def frame25(keep_lirf=False) -> pl.DataFrame:
    ho = pl.read_parquet(ROOT / "cache/v50/chain_all.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("pred"))
    oof = pl.concat([pl.read_parquet(p) for p in sorted((ROOT / "cache/oof/ctxc1sel").glob("fold=*.parquet"))]).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), p=pl.col("pred"))
    pred = pl.concat([ho, oof.join(ho.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")])
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt"]
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))],
                    how="vertical_relaxed").filter((pl.col("PHASE_mvt") == "DEP") & pl.col("TAXITIME_SEC_mvt").is_not_null())
    x = (raw.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(pred, on="MVT_ID_mvt", how="inner")
         .join(obs_flags(), on="MVT_ID_mvt", how="left")
         .with_columns(p=pl.col("p").cast(pl.Float64).clip(0, None), apt=pl.col("ADEP_mvt"),
                       y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), nmu=pl.col("FLIGHT_ID_mvt").is_null(),
                       mo=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
                       trk=pl.col("trk").fill_null(False), ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64))
         .with_columns(obs=pl.col("trk") | pl.col("opdi").fill_null(False), ym=pl.col("mo"),
                       period=pl.col("mo").replace_strict(PER_OF),
                       day=(pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=pl.col("p").round(0).cast(pl.Int64))).dt.date()))
    x = x.join(DV.pub(2025), on=["apt", "day"], how="left")
    return x if keep_lirf else x.filter(pl.col("apt") != "LIRF")


def frame26() -> pl.DataFrame:
    import v57_build as B
    x = B.ranking_frame()
    of = obs_flags().select("MVT_ID_mvt", "opdi").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
    return x.join(of, on="MVT_ID_mvt", how="left").with_columns(
        nmu=pl.col("FLIGHT_ID_mvt").is_null(), hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
        obs=pl.col("trk") | pl.col("opdi").fill_null(False))


def dists_from(c: pl.DataFrame) -> dict:
    return DV.dists_from(c, split=True)


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def by_apt(days):
    out = {}
    for d in days:
        out.setdefault(d["apt"], []).append(d)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x25 = frame25()
    print(f"2025 frame: {x25.height:,} rows (non-LIRF), months {x25['mo'].n_unique()}", flush=True)
    # ---- 2025 day tables, each month with its own eta source
    days25 = []
    for pk, (m1, m2) in PERIODS.items():
        for m in (m1, m2):
            mk = (x25["mo"] == m).to_numpy()
            if pk == "P01":
                cal = x25.filter(pl.col("mo").is_in([m1, m2]) & (pl.col("mo") != m))
            else:
                cal = x25.filter(~pl.col("period").is_in(["P01", pk]))
            xm = x25.filter(pl.Series(mk))
            dd = M.day_table(xm, dists_from(cal), set(A8) | {"LSZH"})
            idx = np.flatnonzero(mk)
            for d in dd:                      # row indices back into x25
                d["g"], d["gm"], d["period"] = idx[d["g"]], idx[d["gm"]], pk
            days25 += dd
        print(f"  {pk}: cumulative airport-days {len(days25)}", flush=True)
    noise = M.noise_s2(x25, days25).with_columns(period=pl.Series([d["period"] for d in days25]))
    noise = noise.with_columns(ec=pl.col("e") - pl.col("e").mean().over("apt", "period"))
    s2_all = dict(noise.group_by("apt").agg((pl.col("ec") ** 2).mean()).iter_rows())
    # ---- 2026 day table
    x26 = frame26()
    ho = x25.filter(pl.col("period") == "P01")
    days26 = M.day_table(x26, dists_from(ho), set(A8) | {"LSZH"})
    d26 = by_apt(days26)
    print(f"2026: {len(days26)} airport-days solved", flush=True)
    # ---- "2026 is different" (label-free): lam dispersion and mean, 2026 vs the 2025 holdout chain (same solve)
    rows = []
    for a in A8 + ["LSZH"]:
        l26 = np.array([d["lam"] for d in d26.get(a, [])])
        l25 = np.array([d["lam"] for d in days25 if d["apt"] == a and d["period"] == "P01"])
        lo25 = np.array([d["lam"] for d in days25 if d["apt"] == a and d["period"] != "P01"])
        c26 = np.concatenate([l26[[d["per"] == m for d in d26[a]]] - l26[[d["per"] == m for d in d26[a]]].mean() for m in (1, 7)])
        rows.append({"apt": a, "mean26": l26.mean(), "sd26_within_month": c26.std(), "mean25_ho": l25.mean(),
                     "sd25_ho": np.concatenate([l25[:len(l25) // 2] - l25[:len(l25) // 2].mean(), l25[len(l25) // 2:] - l25[len(l25) // 2:].mean()]).std(),
                     "sd25_oof": np.std(lo25 - np.mean(lo25)), "s2_noise_sd": np.sqrt(s2_all[a])})
    with pl.Config(tbl_rows=20, float_precision=1, tbl_width_chars=200):
        print("lam (expectation solve, d units, s): 2026 vs 2025 (holdout chain P01, OOF periods); noise sd from 2025")
        print(pl.DataFrame(rows))
    # ---- validation 1: 2026 leave-one-day-out
    fams = {"F0": []} | {k: v for k, v in M.FAMILIES.items()}
    v1 = {}
    for f, cols in fams.items():
        v1[f] = {a: M.loo_rmse(d26[a], cols) for a in A8}
    pooled = {f: float(np.sqrt(sum(r ** 2 * n for r, n in v.values()) / sum(n for _, n in v.values()))) for f, v in v1.items()}
    print("validation 1 (2026 leave-one-day-out RMSE of lam, s):", {f: round(v, 2) for f, v in pooled.items()})
    for f in fams:
        print(f"  {f}:", {a: round(v1[f][a][0], 1) for a in A8})
    kept = [f for f in M.FAMILIES if pooled[f] <= 0.95 * pooled["F0"] and sum(v1[f][a][0] < v1["F0"][a][0] for a in A8) >= 5]
    fstar_cols = [c for f in kept for c in M.FAMILIES[f]]
    if len(kept) > 1:
        vu = {a: M.loo_rmse(d26[a], fstar_cols) for a in A8}
        pu = float(np.sqrt(sum(r ** 2 * n for r, n in vu.values()) / sum(n for _, n in vu.values())))
        ok = pu <= 0.95 * pooled["F0"] and sum(vu[a][0] < v1["F0"][a][0] for a in A8) >= 5
        print(f"  union {kept}: pooled {pu:.2f}, passes {ok}")
        if not ok:
            best = min(kept, key=lambda f: pooled[f]); kept, fstar_cols = [best], list(M.FAMILIES[best])
    print(f"F* = {kept or ['F0']} -> columns {fstar_cols}")
    # ---- input check: 2026 shares within the 2025 range (+-0.05) per airport
    cols_by_apt = {}
    for a in A8:
        okc = True
        for c in fstar_cols:
            w25 = np.array([d["w"][c] for d in days25 if d["apt"] == a]); w26 = np.array([d["w"][c] for d in d26[a]])
            if w26.min() < w25.min() - 0.05 or w26.max() > w25.max() + 0.05:
                okc = False
                print(f"  input check: {a} {c} 2026 [{w26.min():.3f}, {w26.max():.3f}] vs 2025 [{w25.min():.3f}, {w25.max():.3f}] -> F0 at {a}")
        cols_by_apt[a] = fstar_cols if okc else []
    # ---- validation 2: 2025, 12 months, per-row vs R0
    y, p = x25["y"].to_numpy(), x25["p"].to_numpy().astype(float)
    ts, nmu, mo, apt = x25["ts"].to_numpy(), x25["nmu"].to_numpy(), x25["mo"].to_numpy(), x25["apt"].to_numpy()
    variants = {"F0": {a: [] for a in A8}, "F*": cols_by_apt} | {f: {a: c for a in A8} for f, c in M.FAMILIES.items()}
    q = {k: p.copy() for k in ["R0"] + list(variants) + ["C3"]}
    summaries = {}
    for pk, (m1, m2) in PERIODS.items():
        s2p = dict(noise.filter(pl.col("period") != pk).group_by("apt").agg((pl.col("ec") ** 2).mean()).iter_rows())
        dp = by_apt([d for d in days25 if d["period"] == pk])
        for m in (m1, m2):                     # R0 per month
            mk = (mo == m)
            xm = x25.filter(pl.Series(mk)); idx = np.flatnonzero(mk)
            q["R0"][idx] = R.apply(xm, R.solve_days(xm, None, skip=("LIRF", "LSZH")), 0.75)
        for v, cba in variants.items():
            dl, summ = M.shifts(x25, {a: dp.get(a, []) for a in A8}, cba, s2p)
            k = dl != 0
            q[v][k] = np.maximum(p[k] - dl[k], 0)
            if v in ("F0", "F*"):
                summaries[(pk, v)] = summ
        dl, _ = M.shifts(x25, {"LSZH": dp.get("LSZH", [])}, {"LSZH": []}, s2p, level=False)
        k = dl != 0
        q["C3"][k] = np.maximum(p[k] - dl[k], 0)
    body, allr = ~nmu & (y <= 3600), np.ones(len(y), bool)
    in8, lszh = np.isin(apt, A8), apt == "LSZH"
    months = sorted(set(mo))
    out = []
    for v in ["R0"] + list(variants):
        r = {"variant": v, "full": rmse(y[in8] - q[v][in8]) - rmse(y[in8] - p[in8]),
             "body": rmse(y[in8 & body] - q[v][in8 & body]) - rmse(y[in8 & body] - p[in8 & body])}
        r["full_vs_R0"] = rmse(y[in8] - q[v][in8]) - rmse(y[in8] - q["R0"][in8])
        r["body_vs_R0"] = rmse(y[in8 & body] - q[v][in8 & body]) - rmse(y[in8 & body] - q["R0"][in8 & body])
        r["months_full_beat_R0"] = sum(rmse(y[in8 & (mo == m)] - q[v][in8 & (mo == m)]) < rmse(y[in8 & (mo == m)] - q["R0"][in8 & (mo == m)]) for m in months)
        for m in months:
            mm = in8 & (mo == m)
            r[f"f{m[5:]}"] = round(rmse(y[mm] - q[v][mm]) - rmse(y[mm] - q["R0"][mm]), 2)
        out.append(r)
    with pl.Config(tbl_rows=20, tbl_cols=30, float_precision=3, tbl_width_chars=260):
        print("validation 2 (2025, 8 airports): RMSE change vs unshifted p and vs R0; fMM = full vs R0 per month")
        print(pl.DataFrame(out))
    for v in ("R0", "F0", "F*"):
        print(f"  per airport full vs R0 [{v}]:", {a: round(rmse(y[apt == a] - q[v][apt == a]) - rmse(y[apt == a] - q["R0"][apt == a]), 2) for a in A8})
    print("  Jan / Jul (holdout chain) full vs R0:", {v: (round(rmse(y[in8 & (mo == "2025-01")] - q[v][in8 & (mo == "2025-01")]) - rmse(y[in8 & (mo == "2025-01")] - q["R0"][in8 & (mo == "2025-01")]), 2),
                                                       round(rmse(y[in8 & (mo == "2025-07")] - q[v][in8 & (mo == "2025-07")]) - rmse(y[in8 & (mo == "2025-07")] - q["R0"][in8 & (mo == "2025-07")]), 2)) for v in ("F0", "F*")})
    for (pk, v), s in sorted(summaries.items()):
        if v == "F0":
            print(f"  {pk} F0 EB a:", {a: s[a]["a"] for a in s})
    # C3
    c3 = {"full": rmse(y[lszh] - q["C3"][lszh]) - rmse(y[lszh] - p[lszh]),
          "body": rmse(y[lszh & body] - q["C3"][lszh & body]) - rmse(y[lszh & body] - p[lszh & body]),
          "months_full_better": sum(rmse(y[lszh & (mo == m)] - q["C3"][lszh & (mo == m)]) < rmse(y[lszh & (mo == m)] - p[lszh & (mo == m)]) for m in months)}
    print("C3 (LSZH day variation, 2025, vs no change):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c3.items()},
          {m: round(rmse(y[lszh & (mo == m)] - q["C3"][lszh & (mo == m)]) - rmse(y[lszh & (mo == m)] - p[lszh & (mo == m)]), 2) for m in months})
    json.dump({"fstar_cols_by_apt": cols_by_apt, "kept": kept, "s2": s2_all}, open(OUT / "dsm_params.json", "w"), indent=1)
    print("wrote", OUT / "dsm_params.json")


if __name__ == "__main__":
    main()
