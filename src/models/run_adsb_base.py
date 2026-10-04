"""Holdout fit of the LightGBM with ADS-B detector fields as base features
(reports/adsb_base_preregistration.md, stage A).

Run:  .venv/Scripts/python.exe src/models/run_adsb_base.py
First build the features: src/features/adsb_base_feats.py -> cache/features_nb_adsb/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.fit import run  # noqa: E402

FEAT = ROOT / "cache" / "features_nb_adsb"
NAME = "lgb_mixed_nba"
EV = ROOT / "cache" / "eval" / f"{NAME}_holdout_ev.parquet"

if __name__ == "__main__":
    assert (FEAT / "train2025.parquet").exists(), "run src/features/adsb_base_feats.py first"
    run(engine="lgb", target="mixed", seed=42, feat_dir=FEAT, name=NAME, submit=False, ev_out=EV)
