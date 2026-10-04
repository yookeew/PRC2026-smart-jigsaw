# OOF context corrector on the base stack — pre-registration

Written 2026-10-04, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **The context blocks carry signal**
  (`reports/context_corrector_preregistration.md`).
- **But a corrector fit on one holdout month and applied to the other learns
  month-specific bias.**
- **outgoing-boat fits theirs on out-of-block base predictions over the
  training year:** ~1.7M rows, every season. This tests that setup with our
  stack.

## Setup (`tests/oof_context_corrector_test.py`)

- **Stack s** = NNLS(LightGBM-nb, CatBoost) + the LOBT window.
  - Weights are fitted on the 10 OOF months.
  - OOF months use the fold predictions (`cache/oof/lgb_mixed_nb`,
    `cache/oof/cat_mixed`).
  - The holdout uses `lgb_mixed_nb` and `cat_mixed` holdout predictions
    (fit on the 10 months).
- **Blocks:** the six from the context-corrector pre-registration.
  - Cell statistics are leave-one-month-out on OOF rows, and all 10 months
    for the holdout.
  - Surface and distance use s.
- **Corrector:** LightGBM L2 on (y − s), trained on OOF-month rows
  (NM-matched, y ≤ 5 h, not served by a fixed line). Rounds come from the
  inner split day % 5 == 0. Applied to the holdout, then the LOBT window.
- **Control:** s, airport, hour, `aobt3_taxi`, sched offset.
  **Treatment:** control + blocks.

## Decision (stage 1: stack level)

**Proceed to stage 2 iff, on holdout body RMSE (NM-matched, y ≤ 1 h):**
1. treatment beats s by ≥ 2 s pooled;
2. treatment beats control by ≥ 1 s pooled;
3. both hold in Jan and in Jul;
4. trimmed RMSE: treatment ≤ s.

**Stage 2** (pre-registered as an amendment before its result): the corrected
stack replaces s as the input to the v30 chain. The OOF-month corrected stack
needs month-blocked cross-fitting of the corrector.

## Amendments

**2026-10-04, stage 2 pre-registered after the stage-1 result and before any
stage-2 result.**

- **The corrected stack goes in as the chain's LightGBM input:**
  - **Holdout:** the corrector fitted on all 10 OOF months.
  - **OOF months:** cross-fitted over two-month blocks (Feb–Mar, Apr–May,
    Jun–Aug, Sep–Oct, Nov–Dec); each block is predicted by a corrector
    fitted on the other OOF months.
  - **Files:** `cache/eval/ctxcorr_holdout_ev.parquet` and
    `cache/oof/ctxcorr/`.
- **The test:** v30's chain cross-fit (as in `tests/base_neighbour_chain_test.py`),
  base vs treatment.
  - **Base:** the current v31 chain, LightGBM-nb + CatBoost-corrected.
  - **Treatment:** the same chain with the LightGBM input replaced by the
    corrected stack. The chain's NNLS still mixes it with the corrected
    CatBoost.
  - Then the LOBT window and the NM-unmatched lines on both.
- **Adopt iff, on body RMSE** (NM-matched, y ≤ 1 h):
  1. pooled gain ≥ 1.5 s;
  2. the gain holds in Jan and Jul;
  3. trimmed is not worse.
- **If adopted:**
  - the prod corrector is fitted on all 12 months of out-of-sample stack
    predictions (10 OOF months + the Jan/Jul holdout predictions);
  - it's applied to the ranking stack (prod LightGBM-nb + prod CatBoost);
  - the chain is built with `PRC_LGB` pointing at it;
  - keep iff the board < 257.35 (v33).

## Results — stage 1 (2026-10-04, `tests/oof_context_corrector_test.py`, log `logs/oof_context_corrector_test.log`)

**PROCEED.** Holdout body RMSE:

| | stack | control | treatment |
|---|---|---|---|
| pooled | 212.99 | 212.25 | 209.17 |

- **Treatment vs stack:** −3.82 (Jan −1.97, Jul −5.14).
- **Treatment vs control:** −3.08 (Jan −1.23, Jul −4.40).
- **Trimmed:** 264.70 -> 263.60.
- **Per airport (body, vs stack):** LIRF −14.8, LTFM −5.4, LEMD −3.9, EDDM
  −3.6, EHAM −2.8, LEBL −2.4; LFPG +0.7.

## Results — stage 2 (2026-10-04, `tests/oof_context_corrector_stage2.py`, log `logs/oof_context_corrector_stage2.log`)

**ADOPT** (against the corrected baseline below).

**A baseline error in the script.** Its base arm used the old `lgb_mixed`
(the chain loader's default without `PRC_LGB`), not v33's `lgb_mixed_nb`. It
reported body −4.69, part of which is v31's already-banked gain.

**Re-scored against the true v33 method** (`cache/eval/v31_chain_holdout.parquet`
+ window + lines):

| | v33 method | + context corrector | change |
|---|---|---|---|
| body, pooled | 199.59 | 196.64 | −2.94 |
| body, Jan | | | −2.37 |
| body, Jul | | | −3.36 |
| trimmed | | | −1.25 |
| full | | | −1.08 |

- **Every airport improves:** LIRF −7.71, LTFM −3.57, LFPG −3.51, EGLL −3.08,
  EDDF −1.77, LEMD −1.45, LEBL −1.36, EHAM −1.29, LSZH −1.23, EDDM −0.82.
- **Treatment OOF stack weights:** [0.931, 0.077]. The corrected stack
  takes almost all the weight.

## Results — production (2026-10-04)

- **v34** = the v31 chain on the corrected stack (`PRC_LGB=ctxcorr`), then the LOBT window and the NM-unmatched lines. RMS 45.5 s from v33.
- **Board: v34 = 256.78 (−0.58 vs v33). Kept; new best.** About 20% of the holdout body gain transferred.
