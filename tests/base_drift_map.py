"""2025 -> 2026 drift map on the BASE model's inputs (PROGRESS §110; label-free, no model training except the
adversarial classifier). §89's drift map covered only the corrector inputs; the v37/v41/v42 wins were all 2026-only
input misreadings, so this looks for the same kind of thing in cache/features_nb (what lgb_mixed_nb / cat_mixed read).

Jan+Jul 2025 (holdout_gap2025) vs Jan+Jul 2026 (ranking): same months, so seasonality cannot separate them.
  1. per airport x feature: null-rate change, unseen-category rate (vs the 10 training months' levels),
     standardised mean difference and single-feature AUC for numerics; flagged rows printed;
  2. adversarial LightGBM (folds by UTC day), pooled and per airport, top features by gain;
  3. the same with real-conditions features (weather, delays vs plan, calendar) removed, so what is left is
     structure: stands, runways, links, counts, neighbour features.
The group encodings (op_* etc.) are label-derived and added at fit time; not covered here.

Run:  .venv/Scripts/python.exe -u tests/base_drift_map.py
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.encode import feature_matrix  # noqa: E402

FEAT = ROOT / "cache" / "features_nb"
PARAMS = dict(objective="binary", num_leaves=63, learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.8,
              seed=42, deterministic=True, force_row_wise=True, verbose=-1, max_bin=127)
DROP = {"MVT_ID_mvt", "inbound_mvt_id", "doy", "month"}
# real 2026 conditions (weather, how late flights ran vs plan, calendar) -- expected to drift, removed in pass 3
CONDITIONS = {"hour", "dow", "is_weekend", "minute_of_day", "sched_takeoff_offset", "sched_to_takeoff", "eobt_delay",
              "iobt_delay", "lobt_delay", "inbound_arr_delay", "aobt3_vs_eobt", "aobt3_taxi", "actual_ground",
              "has_ceiling", "vis_mi", "ceiling_ft", "temp_c", "dewpoint_c", "temp_dewpoint_spread_c", "below_freezing",
              "deicing_risk", "wx_snow", "wx_freezing", "wx_tstorm", "wx_obscuration", "flight_category", "wind_dir",
              "wind_kt", "gust_kt", "precip_1h_in"}


def sfauc(lab, x):
    ok = ~np.isnan(x)
    if ok.sum() < 500 or len(np.unique(lab[ok])) < 2 or np.nanstd(x[ok]) == 0:
        return np.nan
    a = roc_auc_score(lab[ok], x[ok])
    return max(a, 1 - a)


def adversarial(A, lab, day, cols, cats, tag, top=15):
    fold = (day * 2654435761 % 2**32) % 5
    p, gain = np.zeros(len(lab)), np.zeros(len(cols))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(A[tr], lab[tr], categorical_feature=cats, free_raw_data=False), 200)
        p[te] = m.predict(A[te])
        gain += m.feature_importance("gain")
    print(f"\n[{tag}] adversarial OOF AUC {roc_auc_score(lab, p):.3f}; top features by gain share:")
    for i in np.argsort(-gain)[:top]:
        print(f"  {cols[i]:24s} {gain[i] / gain.sum():6.1%}")
    return p


def main() -> None:
    tr = pl.read_parquet(FEAT / "train2025.parquet")
    _, _, _, categories = feature_matrix(tr)
    del tr
    ho = pl.read_parquet(FEAT / "holdout_gap2025.parquet")
    rk = pl.read_parquet(FEAT / "ranking.parquet")
    Xh, names, cats, _ = feature_matrix(ho, categories)
    Xr, _, _, _ = feature_matrix(rk, categories)
    cols = [c for c in names if c not in DROP]
    apt_h, apt_r = ho["ADEP_mvt"].to_numpy(), rk["ADEP_mvt"].to_numpy()
    day = np.concatenate([(ho["T"].dt.epoch("s") // 86400).to_numpy(), (rk["T"].dt.epoch("s") // 86400).to_numpy()])
    H = Xh.select(cols).to_numpy().astype(np.float64)
    R = Xr.select(cols).to_numpy().astype(np.float64)
    print(f"rows: 2025 Jan+Jul {len(H):,}, 2026 Jan+Jul {len(R):,}; features {len(cols)} ({len([c for c in cats if c in cols])} categorical)")

    # 1. per airport x feature
    print("\n=== 1. per airport x feature flags: |d null| >= 0.05, unseen-level rate up >= 0.02, |SMD| >= 0.3 or single AUC >= 0.62")
    flags = []
    for a in sorted(set(apt_r)):
        kh, kr = apt_h == a, apt_r == a
        for j, c in enumerate(cols):
            h, r = H[kh, j], R[kr, j]
            nh, nr = np.isnan(h).mean(), np.isnan(r).mean()
            msg = []
            if abs(nr - nh) >= 0.05:
                msg.append(f"null {nh:.2f}->{nr:.2f}")
            if c in cats:
                uh, ur = (h == -1).mean(), (r == -1).mean()
                if ur - uh >= 0.02:
                    msg.append(f"unseen {uh:.3f}->{ur:.3f}")
            else:
                hv, rv = h[~np.isnan(h)], r[~np.isnan(r)]
                if len(hv) > 500 and len(rv) > 500:
                    sd = np.sqrt((hv.var() + rv.var()) / 2)
                    smd = (rv.mean() - hv.mean()) / sd if sd > 0 else 0.0
                    auc = sfauc(np.r_[np.zeros(len(hv)), np.ones(len(rv))], np.r_[hv, rv])
                    if abs(smd) >= 0.3 or (auc == auc and auc >= 0.62):
                        msg.append(f"SMD {smd:+.2f} AUC {auc:.3f} median {np.median(hv):.4g}->{np.median(rv):.4g}"
                                   f" p90 {np.quantile(hv, .9):.4g}->{np.quantile(rv, .9):.4g}")
            if msg:
                flags.append((a, c, "; ".join(msg)))
    for a, c, m in flags:
        tag = "  (condition)" if c in CONDITIONS else ""
        print(f"  {a} {c:24s} {m}{tag}")
    print(f"  {len(flags)} flags, {sum(c not in CONDITIONS for _, c, _ in flags)} on structural features")

    # 2. pooled adversarial, all features
    A = np.vstack([H, R])
    lab = np.r_[np.zeros(len(H)), np.ones(len(R))]
    apt = np.r_[apt_h, apt_r]
    ci = [cols.index(c) for c in cats if c in cols]
    adversarial(A, lab, day, cols, ci, "all features, pooled")
    # 3. structure only, pooled and per airport
    sc = [c for c in cols if c not in CONDITIONS]
    si = [cols.index(c) for c in sc]
    S = A[:, si]
    sci = [sc.index(c) for c in cats if c in sc]
    p = adversarial(S, lab, day, sc, sci, "structural features only, pooled")
    print("\nper-airport AUC of the structural pooled classifier, and per-airport structural classifiers:")
    for a in sorted(set(apt_r)):
        k = apt == a
        print(f"  {a}: pooled-model AUC {roc_auc_score(lab[k], p[k]):.3f}", flush=True)
        adversarial(S[k], lab[k], day[k], sc, sci, f"structural, {a} only", top=6)


if __name__ == "__main__":
    main()
