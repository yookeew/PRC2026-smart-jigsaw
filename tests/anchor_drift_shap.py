"""Which base features move LSZH / EDDM / LTFM / EDDF predictions up in 2026? (follow-up to anchor_drift_stages.py)

A proxy of the base direct regressor (same feature prep as models.fit.fit_predict_months: priors + group
encodings fit on the 10 training months, Huber 800, direct taxi target), fitted on these airports only with a
faster learning rate. First it checks that the proxy reproduces the 2026 offset move vs the AOBT_3 anchor;
then per airport x month it compares mean SHAP contributions, 2026 ranking minus 2025 holdout, and prints the
features that move the prediction most, with the feature's own 2025 -> 2026 mean.

Run:  .venv/Scripts/python.exe -u tests/anchor_drift_shap.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.encode import add_group_encodings_oof, apply_group_encodings, apply_priors, fit_group_encodings, fit_priors  # noqa: E402
from models.fit import FEAT_DIR, HOLDOUT_MONTHS, HUBER_ALPHA, TRAIN_MONTHS, VALID_MONTH, _fit_lgb, _matrix  # noqa: E402

APTS = ["LSZH", "EDDM", "LTFM", "EDDF"]
CLIP = 300
TOP = 8
SAMPLE = 4000


def main() -> None:
    sub = lambda f: f.filter(pl.col("ADEP_mvt").is_in(APTS))
    feats, feats_ho, feats_rk = (sub(pl.read_parquet(FEAT_DIR / f)) for f in ("train2025.parquet", "holdout_gap2025.parquet", "ranking.parquet"))
    lab = sub(pl.read_parquet(FEAT_DIR / "labels2025.parquet")).join(
        feats.select("MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt"), on="MVT_ID_mvt", how="left")
    tr_lab = lab.filter(pl.col("ym").is_in(TRAIN_MONTHS))
    ho_lab = lab.filter(pl.col("ym").is_in(HOLDOUT_MONTHS))
    priors = fit_priors(tr_lab)
    f_tr = apply_priors(feats.join(tr_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors)
    tr_lab = f_tr.select("MVT_ID_mvt").join(tr_lab, on="MVT_ID_mvt")
    f_tr = add_group_encodings_oof(f_tr, tr_lab)
    enc = fit_group_encodings(tr_lab)
    f_ho = apply_group_encodings(apply_priors(feats_ho.join(ho_lab.select("MVT_ID_mvt"), on="MVT_ID_mvt"), priors), enc)
    f_rk = apply_group_encodings(apply_priors(feats_rk, priors), enc)

    Xtr, names, cats, categories = _matrix(f_tr)
    Xho, Xrk = _matrix(f_ho, categories)[0], _matrix(f_rk, categories)[0]
    taxi = tr_lab["taxi"].to_numpy()
    keep = (taxi >= 30) & (taxi <= 7200)
    iv = tr_lab["ym"].to_numpy() == VALID_MONTH
    m, _, best = _fit_lgb(Xtr[keep & ~iv], taxi[keep & ~iv], cats, eta=0.08, rounds=3000, es=100,
                          valid=(Xtr[keep & iv], taxi[keep & iv]), objective="huber", alpha=HUBER_ALPHA)
    print(f"proxy best_iter {best}", flush=True)

    raw = pl.concat([pl.scan_parquet(str(ROOT / "data" / "raw" / "training_*.parquet")),
                     pl.scan_parquet(str(ROOT / "data" / "ranking" / "ranking.parquet"))], how="diagonal_relaxed").filter(
        pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64), mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                                              y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).collect()
    out = []
    for f, X, yr in ((f_ho, Xho, 2025), (f_rk, Xrk, 2026)):
        d = f.select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", a=pl.col("aobt3_taxi").cast(pl.Float64)).with_columns(
            yr=pl.lit(yr), p=pl.Series(m.predict(X, num_iteration=best))).join(raw, on="MVT_ID_mvt", how="left")
        # SHAP on a fixed-seed sample of SAMPLE rows per airport x month (full pred_contrib is too slow)
        rng = np.random.default_rng(0)
        k = (d["a"].is_not_null() & d["mo"].is_in([1, 7])).to_numpy()
        idx = []
        for apt in APTS:
            for mo in (1, 7):
                ii = np.flatnonzero(k & (d["ADEP_mvt"] == apt).to_numpy() & (d["mo"] == mo).to_numpy())
                idx.append(rng.choice(ii, min(SAMPLE, len(ii)), replace=False))
        idx = np.sort(np.concatenate(idx))
        Xs = X.iloc[idx]
        c = m.predict(Xs, num_iteration=best, pred_contrib=True)
        out.append((d, d[idx], c, Xs))

    print("\nproxy offset clip-mean(x - a), x = proxy prediction (truth in brackets, 2025)")
    for apt in APTS:
        for mo in (1, 7):
            row = []
            for d, _, _, _ in out:
                k = (d["ADEP_mvt"] == apt) & (d["mo"] == mo) & d["a"].is_not_null()
                e = d.filter(k)
                row.append(f"{np.mean(np.clip(e['p'].to_numpy() - e['a'].to_numpy(), -CLIP, CLIP)):6.0f}")
                if e["y"].null_count() < e.height:
                    row[-1] += f" [{np.mean(np.clip(e['y'].to_numpy() - e['a'].to_numpy(), -CLIP, CLIP)):4.0f}]"
            print(f"  {apt} {mo}: 2025 {row[0]}   2026 {row[1]}")

    (_, d25, c25, X25), (_, d26, c26, X26) = out
    for apt in APTS:
        for mo in (1, 7):
            k25 = ((d25["ADEP_mvt"] == apt) & (d25["mo"] == mo) & d25["a"].is_not_null()).to_numpy()
            k26 = ((d26["ADEP_mvt"] == apt) & (d26["mo"] == mo) & d26["a"].is_not_null()).to_numpy()
            dc = c26[k26].mean(0) - c25[k25].mean(0)
            da = float(np.mean(d26["a"].to_numpy()[k26].clip(0, 3600)) - np.mean(d25["a"].to_numpy()[k25].clip(0, 3600)))
            print(f"\n{apt} month {mo}: mean prediction moves {dc.sum():+.0f} s (anchor a moves {da:+.0f} s); top feature contributions:")
            for j in np.argsort(-np.abs(dc[:-1]))[:TOP]:
                n = names[j]
                if n in cats:
                    v = ""
                else:
                    v25 = np.nanmean(X25[n].to_numpy(dtype=float)[k25]); v26 = np.nanmean(X26[n].to_numpy(dtype=float)[k26])
                    v = f"   mean {v25:10.2f} -> {v26:10.2f}"
                print(f"   {n:32s} {dc[j]:+7.1f}{v}")


if __name__ == "__main__":
    main()
