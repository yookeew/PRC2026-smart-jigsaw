# Model-free neighbour features in the base LightGBM — pre-registration

Written 2026-10-03, before any model was fit on these features. The
features were built (label-free) before writing this. Changes made after
results exist go under Amendments, dated.

## Why

- **The errors are shared in time** (§70): the oracle neighbour
  correlation is 0.30-0.48 at most airports.
- **v30's post-hoc corrector banked −0.73 on the board** but transferred
  at about half strength (§72-73). It is:
  - trained on 115 sample days;
  - built on pseudo-residuals that depend on the stack prediction.
- **The model-free versions of these signals can be base-model inputs**,
  learned from all ~1.7M training rows in interaction with every
  existing feature:
  - neighbours' NM taxi;
  - neighbours' delay;
  - arrivals' taxi-in excess.
- **Expected:** +1-3 board points, aimed at LTFM, LFPG and LIRF
  (together ~63% of v30's holdout squared error). It overlaps with v30's
  corrector, so the net gain after the full chain is uncertain.

## Features (`src/features/neighbour_feats.py` -> `cache/features_nb/`)

- **Neighbour departures** (other departures taking off within ±W of T,
  W ∈ {900, 1800} s; by airport and by airport × runway; null when fewer
  than 2 neighbours):
  - `nbnm`: mean NM taxi (T − AOBT_3, 0-5,400 s);
  - `nboff`: mean delay (T − SOBT, clipped to −3,600..14,400).
- **Within-day anomalies:** `nbnm_anom_900`, `nboff_anom_900`, the airport
  ±15 min mean minus the airport's UTC-day mean.
- **Arrivals:** `arrx_900`, `arrx_1800` and `arrx_prev` ([T − 1,800,
  T − 900]). Each is the mean arrival taxi-in excess over the 2025 median
  for that (airport, stand, runway).
- **13 columns in all.** No departure off-block time and no model output
  is read.
- **Each frame is built on its own** (train2025, holdout_gap2025,
  ranking), so neighbours come from the same frame.

## Stage B — holdout gate (decides whether stage C runs)

`src/models/run_neighbour_base.py holdout`, then
`tests/base_neighbour_test.py`:
- **Treatment:** LightGBM `target="mixed"`, seed 42, default settings, on
  `cache/features_nb/` (the existing features plus the 13). It's fit on
  the 10 training months and predicts the Jan+Jul 2025 holdout.
- **Base:** the existing `lgb_mixed` holdout predictions (same settings
  without the 13 columns).
- **Both arms** are stacked with the existing `catcorr` via NNLS,
  cross-fit Jan ↔ Jul. Scored at the stack level, before ADS-B and
  neighbour stages.

**Proceed to stage C iff all of:**
1. pooled trimmed delta < 0 with P(worse) < 0.05;
2. trimmed delta < 0 in both Jan and Jul;
3. guard: full-RMSE P(worse) < 0.9.

## Stage C — full-chain gate (decides the build)

- **Steps:**
  - `src/models/run_neighbour_base.py oof` → `cache/oof/lgb_mixed_nb/`;
  - then `tests/base_neighbour_chain_test.py`.
- **The test:** v30's method cross-fit (v27 combiner + v2 within-day
  neighbour corrector), computed twice:
  - with the new LightGBM (holdout predictions and OOF folds);
  - with the old one.
- **The rule:** the same 1-3 as stage B, on the final chain predictions.
- **If it passes:**
  - `run_neighbour_base.py prod`;
  - rebuild the chain with the new LightGBM in place of the old (stack ->
    v23 stages -> full-year combiner -> v2 neighbour corrector) as v31;
  - keep iff the board < 261.40 (v30).

**If either gate fails: stop.** No feature subsets, windows or
hyperparameter variants.

## Amendments

(none)

## Results — stage B (2026-10-03, `tests/base_neighbour_test.py`, log `logs/base_neighbour_test.log`; run after commit 222150c)

**Stage B: PASS -> stage C.**
- **Holdout fit:** `logs/lgb_mixed_nb_holdout.log`, 4,669 s; best_iter
  flip 7,996, direct 2,896; clf 397.
- **LightGBM alone (trimmed):** 273.63 -> 271.99.

| trimmed RMSE (stack with catcorr) | base | + neighbour features | delta | 95% CI | P(worse) |
|---|---|---|---|---|---|
| A: Jul (fit Jan) | 295.34 | 293.47 | −1.87 | [−2.90, −0.93] | 0.000 |
| B: Jan (fit Jul) | 230.93 | 228.77 | −2.15 | [−3.85, −1.13] | 0.000 |
| pooled | 268.51 | 266.55 | −1.97 | [−2.89, −1.16] | 0.000 |
| guard: pooled full | 331.72 | 330.12 | −1.60 | [−2.47, −0.91] | 0.000 |

- **Per airport (trimmed):** LTFM −6.81, EGLL −3.18, LEBL −1.99, LEMD
  −1.87, EDDM −1.78, LSZH −1.16, EDDF −1.11, LFPG −1.05, EHAM −0.51,
  LIRF −0.01.
- **Stage C started:** `src/models/run_neighbour_base.py oof` (log
  `logs/oof_lgb_mixed_nb.log`).
