"""Which ranking set a run targets (PROGRESS.md §132).

  PRC_SCOPE=jj       (default) the live leaderboard: Jan + Jul 2026, data/ranking/ranking.parquet
                     and submitting.parquet, outputs at the original paths. Every script that is not
                     scope-aware behaves exactly like this.
  PRC_SCOPE=final4m  the blind board: Jan + Feb + Jun + Jul 2026, template final_submitting.parquet.
                     Outputs go under a final4m/ subfolder of each output directory, so nothing an
                     existing submission was built from is overwritten.

RANKING is the movement file a scope reads. For final4m it is final_ranking.parquet, for every row
including Jan/Jul: the organisers ruled that the final phase uses final_ranking.parquet /
final_submitting.parquet (challenge Discord, 2026-10-08). Each export carries its own random
+-~6 s dither of the actual times (§132), so the final release's Jan/Jul rows differ slightly
from ranking.parquet's; the live board keeps using ranking.parquet.

Training inputs (2025 features, labels, holdout frames, OOF folds) do not depend on the scope.
The Jan/Jul board file is the final4m output filtered to the IDs in submitting.parquet.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = os.environ.get("PRC_SCOPE", "jj")
if SCOPE not in ("jj", "final4m"):
    raise ValueError(f"PRC_SCOPE must be 'jj' or 'final4m', got {SCOPE!r}")
FINAL = SCOPE == "final4m"

_RANK_DIR = ROOT / "data" / "ranking"
RANKING_JJ = _RANK_DIR / "ranking.parquet"
TEMPLATE_JJ = _RANK_DIR / "submitting.parquet"
RANKING = _RANK_DIR / ("final_ranking.parquet" if FINAL else "ranking.parquet")
TEMPLATE = _RANK_DIR / ("final_submitting.parquet" if FINAL else "submitting.parquet")


def ranking_frame(columns: list[str] | None = None):
    """The ranking movements for this scope."""
    import polars as pl
    return pl.read_parquet(RANKING, columns=columns)


def out_dir(base: Path) -> Path:
    """`base` for the live-board scope, `base/final4m` for the blind board (created)."""
    if not FINAL:
        return base
    d = base / "final4m"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Per-day caches (ADS-B detections, cache/<stage>/day=YYYY-MM-DD.parquet). 2025 days come from the
# training files and are shared by both scopes; 2026 days are built from the scope's ranking file,
# so under final4m they live in <stage>/final4m/. For jj both helpers reduce to the plain base dir.
def day_path(base: Path, day) -> Path:
    """Where the cache file for `day` (a date) is written in this scope."""
    return (out_dir(base) if day.year >= 2026 else base) / f"day={day.isoformat()}.parquet"


def day_files(base: Path, pattern: str = "day=*.parquet") -> list[Path]:
    """Cache files matching `pattern`: 2025 days from `base`, 2026 days from this scope's dir."""
    if not FINAL:
        return sorted(base.glob(pattern))
    old = [p for p in base.glob(pattern) if not p.name.startswith("day=2026")]
    return sorted(old + list(out_dir(base).glob(pattern)), key=lambda p: p.name)
