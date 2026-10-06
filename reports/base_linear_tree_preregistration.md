# Linear leaves in the direct base regressor — pre-registration (screen)

Written 2026-10-07, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **Where the error is:** v42's holdout decomposition (`logs/error_decomp_v42.log`) puts 16% / 14% of squared error in NM-matched rows with 20–40 min / 40 min–5 h taxis.
  - By prediction decile the bias is ≤ 13 s everywhere. So that error is variance, not calibration.
- **Linear leaves:** LightGBM's `linear_tree` fits a linear model inside each leaf, so predictions can scale with strong continuous inputs (MVT − AOBT_3, congestion counts) instead of stepping. Two public results:
  - joyous-rainbow: board 333.7 → 301.8;
  - kind-mango: −15.6 together with other changes.
  - Both at weaker model stages, so a much smaller effect is expected here.
- **Known risk:** extrapolation on unfamiliar rows. kind-mango later had to cap wild 2026 predictions. Our smoke test on 60k rows already gave raw predictions from −1,627 to 3,554 s.
- **Idea from those public READMEs; code ours.**

## Change

`PRC_LINEAR_TREE=1` in `src/models/fit.py`: `linear_tree=True`, `linear_lambda=1.0` (kind-mango's deployed value, fixed here) on regression fits only; the echo classifier is untouched.

**Screen fit:** the direct regressor (`target="direct"`, LIRF rows included, as in the base) with everything else as `lgb_mixed_nb`:
- features, Huber 800, seed 42;
- 10-month fit, holdout Jan+Jul 2025.

## Treatment (from cached predictions)

As in `reports/base_direct_nolirf_preregistration.md`:
- **non-LIRF rows that don't use the median prior:** `echo_prob × clip(offset, 0, 140000) + (1 − echo_prob) × clip(new raw, 0, 10800)`;
- **LIRF (flip path) and prior rows:** unchanged.

## Measurement

Base-level slices vs `lgb_mixed_nb`:
- body (pooled / Jan / Jul), trimmed, full, per airport;
- the NM-matched 20–40 min and 40 min–5 h label buckets (information only; outcome-conditioned).

**Extrapolation check:** the share of raw direct predictions on the changed rows below 0 or above 7,200 s, new vs cached.

## Decision (screen)

**Qualifies for the full build iff:**
1. body pooled gain ≥ 1.0 s;
2. Jan > 0 and Jul > 0;
3. full RMSE not worse;
4. the share of raw predictions outside [0, 7200] is at most 2× the cached regressor's.

**Full build if it qualifies:**
- direct regressor out-of-fold months and production;
- `lgb_mixed_nb`'s flip regressor and classifier kept;
- correctors and chain on top of v42's recipe;
- its own stage 2.

## Amendments

(none)

## Results

**Screen** (2026-10-07; logs `logs/base_linear_fit.log`, `logs/base_linear_compare.log`):
- Linear-leaf direct regressor: best_iter 1,074 (cached Huber direct: 2,896).

| holdout | base | linear leaves | delta |
|---|---|---|---|
| body pooled | 215.86 | 218.78 | +2.92 |
| body Jan | 197.82 | 200.66 | +2.84 |
| body Jul | 229.53 | 232.52 | +2.99 |
| trimmed | 271.99 | 273.23 | +1.23 |
| full | 334.33 | 335.34 | +1.00 |

- Worse at 8 of 9 changed airports (LEBL +11.55, EHAM +6.88, LTFM +6.16, EDDF +4.06); EGLL −2.22.
- Extrapolation: raw predictions outside [0, 7200] 0.0003% → 0.0067% (22×), range −2,484 … 16,544 s, on 2025 data.
- Label 40 min–5 h (outcome-conditioned, info only): 1,065.5 → 1,044.2. 20–40 min: 264.5 → 267.9.
- **Fails rules 1, 3 and 4. Closed.** The comparison table's "no LIRF" column header is a leftover label from the reused code; the column is the linear-leaf arm.
