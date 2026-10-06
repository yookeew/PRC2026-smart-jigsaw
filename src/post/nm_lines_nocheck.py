"""src/post/nm_lines.py's main() without its holdout printout.

holdout_check() is diagnostic only (it never feeds the 12-month fit) and reads cache/eval/v31_chain_holdout.parquet,
which is not in every local cache. Everything else is nm_lines.main() unchanged.

Run:  .venv/Scripts/python.exe src/post/nm_lines_nocheck.py IN.parquet OUT.parquet
"""
from __future__ import annotations

import os
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from post import nm_lines as NL  # noqa: E402

if __name__ == "__main__":
    NL.holdout_check = lambda raw: print("(holdout printout skipped: diagnostic only)")
    NL.main()
