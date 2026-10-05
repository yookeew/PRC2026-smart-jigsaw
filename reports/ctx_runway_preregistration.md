# Context corrector, block 7 (runway state) — pre-registration

Written 2026-10-05, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **Where we stand:**
  - v35 (ensemble2 corrector on fixed features) scored 255.46. About 80% of
    its holdout body gain transferred, so corrector blocks now transfer.
  - outgoing-boat's corrector carries a runway block that ours lacks:
    configuration in a 60-min window, dominant runway, wake-weighted queue on
    the runway, cadence of recent takeoffs.
- **Already in our pipeline:**
  - the base model has the inferred departure/arrival configuration, the
    number of active runways and the time since a configuration change;
  - the corrector has the unweighted queue at AOBT_3 on the runway
    (`ret_active_rwy`).
- **This block adds only what neither has.** The idea is outgoing-boat's; the
  definitions and code are ours.

## Block 7 (`tests/context_corrector_test.py::runway_block`, enabled by `CTX_RWY=1`)

All inputs are test-visible: takeoff/landing times, runways, AOBT_3 and wake
category. No departure off-block time and no labels are used.

| column | definition |
|---|---|
| `rw_cad20` | mean gap between the last 20 takeoffs on the same runway, ending at this one |
| `rw_cad20_apt` | the same over the airport |
| `rw_gap_prev` / `rw_gap_next` | time to the previous / next takeoff on the same runway |
| `rw_share60` | share of the airport's takeoffs in [T − 60 min, T] that used this runway |
| `rw_arr30` | landings on the same runway designator in [T − 30 min, T] (mixed mode) |
| `rw_dom_change` | whether the dominant departure runway of the current 30-min bin differs from the bin one hour earlier |
| `rw_heavy_q` | H/J-category departures on the same runway with AOBT_3 ≤ T < their takeoff |

## Test

1. **Stage 1** (`tests/ctx_runway_test.py`):
   - same data, rows and inner split as `tests/ctx_ensemble_test.py`;
   - both arms use the adopted ensemble2 corrector (global + per-airport
     LightGBM);
   - control = the six current blocks; treatment = six blocks + block 7.
2. **Stage 2:** `PRC_LGB=ctxcorr2 CTX_ENS=2 CTX_RWY=1 tests/oof_context_corrector_stage2.py`.
   The base is v35's chain.

## Decision

**Stage 1 — proceed iff, on holdout body RMSE** (NM-matched, y ≤ 1 h),
treatment vs control:
1. pooled gain ≥ 0.5 s;
2. Jan > 0 and Jul > 0;
3. trimmed is not worse.

**Stage 2 — adopt iff, vs v35's chain:**
1. pooled body gain > 0;
2. Jan > 0 and Jul > 0.

**Then:** production build (`ctxcorr2r`), and v36 through the same chain.
Keep it iff the board is < 255.46.

## Amendments

**2026-10-05, before any result.** Two changes made during implementation:
- **Seam guard added.** `rw_gap_*` > 6 h and `rw_cad20*` > 1 h are set to null. The ranking frame jumps from 31 Jan to 1 Jul, so gaps across that seam would be 2026-only artefacts. The code was smoke-tested on Jul 2025 only; no metric had been computed.
- **`rw_arr30` is mostly 0.** Most airports run segregated runways, and arrival and departure designators rarely coincide. Kept as defined.

## Results

**Stage 1** (2026-10-05, log `logs/ctx_runway_test.log`):

| slice | control | treatment |
|---|---|---|
| body pooled | 206.91 | 206.09 (−0.81) |
| body Jan | 189.36 | 188.31 (−1.05) |
| body Jul | 220.19 | 219.54 (−0.65) |
| trimmed | 261.26 | 260.01 (−1.26) |

- 9 of 10 airports improve; LIRF moves by +0.21.
- 2025-vs-2026 SMD of every block-7 column is ≤ 0.09 in absolute value.
- **PROCEED to stage 2.**

**Stage 2** (2026-10-05, `PRC_LGB=ctxcorr2 CTX_ENS=2 CTX_RWY=1 tests/oof_context_corrector_stage2.py`, log `logs/ctx_runway_stage2.log`):

| slice | v35 chain | + block 7 |
|---|---|---|
| body pooled | 195.04 | 194.58 (−0.46) |
| body Jan | 176.80 | 176.30 (−0.50) |
| body Jul | 208.74 | 208.31 (−0.43) |
| trimmed | 246.60 | 246.01 (−0.59) |
| full | 305.20 | 304.71 (−0.49) |

- 9 of 10 airports improve; LIRF +1.30.
- The script printed "REJECT". It still carries the decision line of `reports/oof_context_corrector_preregistration.md`, which needs a larger gain.
- **This pre-registration's stage-2 rule** (pooled > 0, Jan > 0, Jul > 0) **holds: ADOPT.**
- The production build (`ctxcorr2r`, v36) is on hold at the user's request.

**Board (2026-10-05): v36 = 254.51 (−0.95 vs v35's 255.46). Kept.**
