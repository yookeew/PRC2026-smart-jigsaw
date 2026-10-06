# Anchored base member: LightGBM on y − (MVT − AOBT_3) — pre-registration (screen)

Written 2026-10-06, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **kind-mango** (public README, skylinkapi/prc-data-challenge-2026-kind-mango) got **−3.73 on the board** from this member (their v74).
  - It is a LightGBM trained on `y − clip(MVT − AOBT_3, 0, 7200)`.
  - It's served as anchor + predicted offset, averaged 50/50 with their plain member on rows that have an AOBT_3.
  - Their holdout showed −818 MSE against the plain member, with all 9 airports better.
- **We have no member like it.**
  - Our base predicts taxi directly (non-LIRF) or via `d` (LIRF).
  - AOBT_3 enters only as features (`aobt3_taxi` etc.).
  - An anchored target makes the model learn the *correction to NM's own off-block time*, a differently shaped function. So its errors should be partly decorrelated from the plain member's, which is what an average needs.
- **Idea theirs; target definition and code ours.**

## Change

`target="anchored"` in `src/models/fit.py`:
- **Anchor:** `clip(aobt3_taxi, 0, 7200)`, where `aobt3_taxi` = MVT − AOBT_3 (an existing base feature).
- **Training rows:** taxi in [30, 7200] *and* AOBT_3 present; target = taxi − anchor.
- **Loss and features:** Huber 800 and all other settings exactly as `lgb_mixed_nb`: same features, seed 42, 10-month fit, holdout Jan+Jul 2025.
- **Raw prediction:** anchor + model output (NaN where there's no AOBT_3).

## Treatment (built from cached predictions, no refit of the existing member)

On the Jan+Jul 2025 holdout, starting from the cached `lgb_mixed_nb` holdout predictions:
- **non-LIRF rows with AOBT_3:**
  - raw taxi = 0.5 × cached direct raw + 0.5 × anchored raw;
  - then the existing reconstruction: `echo_prob × clip(offset, 0, 140000) + (1 − echo_prob) × clip(raw, 0, 10800)`;
- **all other rows** (LIRF, no AOBT_3, prior rows): unchanged.

**Sanity check:** the same formula applied to the cached raw prediction must reproduce the cached `pred` on those rows.

**Weight fixed at ½**, as kind-mango served it. The anchored-only arm (weight 1) is reported as information, not used for the decision.

## Measurement and decision (screen)

`src/models/run_anchored.py compare`, at base level vs `lgb_mixed_nb`.

**Qualifies for the full build iff:**
1. body pooled gain ≥ 1.0 s;
2. Jan > 0 and Jul > 0;
3. full RMSE not worse.

**The full build, if it qualifies:**
- an anchored member for the 10 out-of-fold months and production;
- the stack and the v41 chain rebuilt on it;
- its own stage-2 check against v41's chain.

## Amendments

(none)

## Results

**Screen** (2026-10-06; logs `logs/base_anchored_fit.log`, `logs/base_anchored_compare.log`):
- Anchored fit: best_iter 2,680 (Huber).
- Reconstruction check: exact on the 312,885 changed rows.

| holdout | base | ½ + ½ anchored | delta | anchored only (info) |
|---|---|---|---|---|
| body pooled | 215.86 | 215.28 | −0.58 | +1.33 |
| body Jan | 197.82 | 196.68 | −1.14 | +0.75 |
| body Jul | 229.53 | 229.35 | −0.18 | +1.74 |
| trimmed | 271.99 | 270.69 | −1.30 | −0.22 |
| full | 334.33 | 333.27 | −1.06 | −0.18 |

- The average beats the plain member at 8 of 9 affected airports (LTFM +0.33), though the anchored member alone is worse on body. Real decorrelation, but small.
- **Does not qualify** (body −0.58 < 1.0; Jul only −0.18).
- Not pursued further. The anchored target code stays in `fit.py` as an option.
