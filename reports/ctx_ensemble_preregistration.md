# Context corrector as a three-model ensemble — pre-registration

Written 2026-10-04, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **outgoing-boat's corrector is an equal-weight mean** of a global
  LightGBM, one LightGBM per airport and a CatBoost, on the same inputs.
- **Their notes:** "the gain comes from the average, not from any single
  one of them" (README §1.2). Their CatBoost alone scored 309.53 against
  309.27 for the LightGBM.
- **Ours is a single global LightGBM.**

## Test (`tests/ctx_ensemble_test.py`)

- **Same as stage 1** of `reports/oof_context_corrector_preregistration.md`:
  data, blocks, training rows (10 OOF months), inner split, and holdout
  application.
- **The arms:**
  - single (global LightGBM);
  - per-airport LightGBM (airports with ≥ 20k training rows, else the
    global);
  - CatBoost (depth 8, lr 0.08, early stopping on the inner split, refit);
  - **ensemble = equal-weight mean of the three corrections.**

## Decision

**Adopt the ensemble iff, on holdout body RMSE** (NM-matched, y ≤ 1 h),
ensemble vs single:
1. pooled gain ≥ 1.0 s;
2. the gain holds in Jan and Jul;
3. trimmed is not worse.

**If adopted:**
- the stage-2 inputs (holdout + two-month-block OOF) are rebuilt with the
  ensemble;
- the chain gate (as stage 2) is rerun against the chain with the single
  corrector;
- then a production build.

## Amendments

**2026-10-04, before any result.** The first run crashed at the CatBoost
member: catboost isn't installed in the local environment, and it isn't a
project dependency. No metric had been computed.

- **Added arm, "ensemble2":** the equal-weight mean of the global and
  per-airport LightGBM.
- **It's decided by the same rule** (vs single).
- The three-model ensemble runs once catboost is available. Installing it
  needs the user's approval, because it changes `pyproject.toml` and
  `uv.lock`.

## Results (2026-10-04, `tests/ctx_ensemble_test.py`, log `logs/ctx_ensemble_test.log`)

The run printed the arm table, then crashed in the decision block (a leftover
reference to the three-model arm, since fixed). The rule is applied here to
the printed numbers.

| arm | body | Jan | Jul | trimmed |
|---|---|---|---|---|
| stack | 212.99 | 194.85 | 226.71 | 264.70 |
| single | 209.17 | 192.88 | 221.57 | 263.60 |
| per-airport | 208.61 | 191.54 | 221.56 | 262.55 |
| ensemble2 | 207.81 | 191.05 | 220.54 | 261.79 |

**ensemble2 vs single: ADOPT.**
- body −1.36 pooled (Jan −1.83, Jul −1.03);
- trimmed −1.81.

The three-model arm is pending catboost.

**2026-10-04, amendment: feature bug found, gate stopped before its result.**

- **Where:** a 2025-vs-2026 shift check on the corrector's inputs.
- **What:** `ret_active_rwy` (mean |SMD| 0.60, max 1.47; the others ≤ 0.21)
  and `sur_arr_taxi` were computed as differences of two cumulative counts
  over *unequal* sets:
  - flights without AOBT_3 counted in one set but not the other;
  - arrivals without an in-block time counted in one set but not the other.
- **The effect:** they drift with the row's position in the frame, so they
  act as a hidden time index. That means one thing in a 12-month training
  frame and another in a 2-month 2026 frame. This plausibly explains v34's
  weak transfer (20%).
- **The fix:** both counts now use the same set (`tests/context_corrector_test.py`).
- **Status:**
  - The stage-2 run for ensemble2 was stopped before it produced a result.
  - All gates are rerun on the fixed features, starting with this screen.
  - The fixed corrector must beat v34's chain in stage 2 (base = the
    `PRC_LGB=ctxcorr` chain).

## Results on fixed features (2026-10-04, log `logs/ctx_ensemble_test_fixed.log`)

catboost was installed locally (not declared in `pyproject.toml`) for this run only.

| arm | body | Jan | Jul | trimmed |
|---|---|---|---|---|
| stack | 212.99 | 194.85 | 226.71 | 264.70 |
| single | 207.97 | 190.57 | 221.16 | 262.65 |
| per-airport | 207.59 | 189.77 | 221.06 | 262.12 |
| ensemble2 | 206.91 | 189.36 | 220.19 | 261.26 |
| catboost | 208.25 | 190.38 | 221.76 | 273.30 |
| ensemble (3 models) | 206.74 | 189.17 | 220.04 | 263.82 |

**ensemble2 vs single: ADOPT.**
- body −1.06 pooled (Jan −1.21, Jul −0.97);
- trimmed −1.39.

**Three-model ensemble: REJECT.**
- Its trimmed score is worse than single (263.82 vs 262.65), because CatBoost alone is 273.30 on trimmed.
- catboost was uninstalled again, and `catboost_info/` removed. No dependency change.

The fix alone improves single body by 1.20 (209.17 → 207.97).

Next: the stage-2 chain gate for ensemble2. The base is v34's chain (`PRC_LGB=ctxcorr`).

## Stage 2 and production (2026-10-04)

**Stage 2** (`PRC_LGB=ctxcorr CTX_ENS=2 tests/oof_context_corrector_stage2.py`, log `logs/ctx_ensemble_stage2.log`; base = v34's chain):

| slice | v34 chain | ensemble2 (fixed features) |
|---|---|---|
| body pooled | 196.64 | 195.04 (−1.60) |
| body Jan | 179.18 | 176.80 (−2.38) |
| body Jul | 209.82 | 208.74 (−1.07) |
| trimmed | 247.59 | 246.60 (−0.99) |
| full | 305.94 | 305.20 (−0.74) |

- 9 of 10 airports improve; LIRF moves by +0.40. **ADOPT.**

**Production:**
- `CTX_ENS=2 src/post/ctx_corrector.py` writes `ctxcorr2.parquet`. Global best_iter is 1,975.
- The chain then runs with `PRC_LGB=ctxcorr2` and writes `smart-jigsaw_v35.parquet`. Its RMS difference from v34 is 28.2 s.
