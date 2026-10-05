# Context corrector, block 8 (airline + stand rotation) — pre-registration

Written 2026-10-05, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **The trend:** v36 (corrector + runway block 7) scored 254.51. Corrector
  blocks now transfer at 80–200% of their holdout gain.
- **The source:** outgoing-boat's corrector has an airline + stand-rotation
  bundle (part of their "plano 13", −2.9 on their board together with METAR
  and NM consistency). Their "copy rate by airline" is one of their external
  features.
- **Our base already has** the operator as a categorical and the rotation
  columns from stand linking. The corrector has neither. The corrector models
  the residual of our stack, so what it can use is how that residual and the
  copy pattern vary by airline, plus the rotation columns. The idea is
  outgoing-boat's; the definitions and code are ours.

## Block 8 (`tests/context_corrector_test.py::airline_block`, enabled by `CTX_AIR=1`)

**Airline statistics.**
- **Inputs:** the stack residual r = clip(y − s, ±3,600) and the copy flag
  |y − (T − SCHED)| ≤ 60 s.
- **Rows used:** NM-matched rows with y ≤ 5 h.
- **Grouping:** per (airport, operator).
- **Leave-one-month-out:** a labelled row never uses its own month. Ranking
  rows use all 12 months of 2025.
- **Minimum size:** a cell needs ≥ 30 rows, otherwise null.

| column | definition |
|---|---|
| `al_res` | mean stack residual of the (airport, operator) cell |
| `al_copy` | copy rate of the cell |
| `al_n` | rows in the cell |
| `al_op` | operator code (categorical) |

**Rotation**, read from the base feature cache (`cache/features_nb`) by
MVT_ID. These are test-visible stand-link outputs.

| column | definition |
|---|---|
| `ro_inb_delay` | inbound in-block − inbound scheduled time |
| `ro_sched_ground` | scheduled turnaround |
| `ro_actual_ground` | own in-block to own takeoff |
| `ro_link_conf` | link confidence (coded) |

## Test

1. **Stage 1** (`tests/ctx_airline_test.py`):
   - same setup as `tests/ctx_runway_test.py`;
   - both arms are the ensemble2 corrector, with block 7 on;
   - control = blocks 1–7, treatment = blocks 1–8;
   - also prints a 2025-vs-2026 SMD of the new numeric columns.
2. **Stage 2:** `PRC_LGB=ctxcorr2r CTX_ENS=2 CTX_RWY=1 CTX_AIR=1 tests/oof_context_corrector_stage2.py`.
   The base is v36's chain.

## Decision

**Stage 1 — proceed iff, on holdout body RMSE**, treatment vs control:
1. pooled gain ≥ 0.5 s;
2. Jan > 0 and Jul > 0;
3. trimmed is not worse;
4. no new column has |SMD| > 0.25 in either month.

**Stage 2 — adopt iff, vs v36's chain:**
1. pooled body gain > 0;
2. Jan > 0 and Jul > 0.

**Then:** production (`ctxcorr2ra`) and v37 through the same chain. Keep it
iff the board is < 254.51.

## Amendments

**2026-10-05, before any result: encoding bug found while smoke-testing block 8.**
- **The bug.** Categorical codes in the corrector came from Polars `Categorical -> to_physical()`. Those codes follow each frame's order of first appearance.
- **Effect on production:**
  - `apt_c` happened to agree between the training and ranking frames;
  - `stand_pfx` agreed for only 17 of 140 shared prefixes.
  - So v34–v36 applied the corrector to 2026 rows with wrong stand-prefix codes.
  - The holdout gates could not see it, because holdout rows sit in the training frame.
- **Fix:** `code()` in `tests/context_corrector_test.py` uses a sorted vocabulary over all 2025 + 2026 departures for `apt_c`, `stand_pfx` and `al_op`; unseen values become null.
- **v37** is v36 rebuilt with the fix only (no block 8), so the fix's board effect is measured on its own.
- This test runs on the fixed encoding, in both arms.
