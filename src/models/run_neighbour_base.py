"""Resumable LightGBM runs with the model-free neighbour features (reports/base_neighbour_preregistration.md).

Every piece is written as soon as it finishes, and a rerun skips what already exists, so
the job can be stopped (Ctrl+C, sleep, shutdown) and restarted at any time. The longest
uninterrupted piece is one fit, about 30-60 min.

  holdout  10-month fit, Jan+Jul 2025 predictions   -> cache/eval/lgb_mixed_nb_holdout_ev.parquet
           (+ .rounds.json with the early-stopping round counts the refit reuses)
  oof      one fold per training month              -> cache/oof/lgb_mixed_nb/fold=*.parquet
  prod     refit on all 2025, predict the ranking   -> data/submissions/lgb_mixed_nb.parquet

Run:  .venv/Scripts/python.exe src/models/run_neighbour_base.py holdout
      .venv/Scripts/python.exe src/models/run_neighbour_base.py oof       (only after the stage-B gate passes)
      .venv/Scripts/python.exe src/models/run_neighbour_base.py prod      (only after the stage-B gate passes)
First build the features: src/features/neighbour_feats.py -> cache/features_nb/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.fit import run, run_oof  # noqa: E402

FEAT = ROOT / "cache" / "features_nb"
NAME = "lgb_mixed_nb"
EV = ROOT / "cache" / "eval" / f"{NAME}_holdout_ev.parquet"
OOF = ROOT / "cache" / "oof" / NAME
SUBMISSION = ROOT / "data" / "submissions" / f"{NAME}.parquet"


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "holdout"
    assert (FEAT / "train2025.parquet").exists(), "run src/features/neighbour_feats.py first"
    if stage == "holdout":
        if EV.exists() and EV.with_suffix(".rounds.json").exists():
            print(f"holdout: {EV.name} exists, nothing to do")
            return
        run(engine="lgb", target="mixed", seed=42, feat_dir=FEAT, name=NAME, submit=False, ev_out=EV)
    elif stage == "oof":
        run_oof(engine="lgb", target="mixed", seed=42, feat_dir=FEAT, name=NAME, out_dir=OOF)  # skips finished folds
    elif stage == "prod":
        if SUBMISSION.exists():
            print(f"prod: {SUBMISSION.name} exists, nothing to do")
            return
        run(engine="lgb", target="mixed", seed=42, feat_dir=FEAT, name=NAME, submit=True, ev_out=EV,
            resume=True)   # reuses the holdout predictions and round counts; refits only
    else:
        raise SystemExit(f"unknown stage {stage!r} (holdout | oof | prod)")


if __name__ == "__main__":
    main()
