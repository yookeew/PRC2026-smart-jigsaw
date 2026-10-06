# Direct base regressor trained without LIRF rows — pre-registration (screen)

Written 2026-10-06, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **kind-mango's biggest single board gain (v51, −5.99)** was retraining their base without the LIRF and y > 80,000 rows. Those rows bent the base toward predictions it then made on other airports' 2026 rows.
- **In our mixed base:**
  - LIRF predictions come from the flip regressor; every other airport uses the direct regressor;
  - but the direct regressor still trains on all 10 airports, including LIRF's 160k labelled rows.
  - LIRF has a 20% schedule-copy share and the noisiest labels, and the direct model's LIRF predictions are never used.
- **So excluding LIRF from the direct fit** changes nothing at LIRF and removes a possible distortion everywhere else. (Our labels > 7,200 s are already excluded.)
- **Idea kind-mango's; code ours.**

## Change

`target="direct_nolirf"` in `src/models/fit.py`: the direct regressor with LIRF rows removed from its training (and early-stopping) rows. Everything else as `lgb_mixed_nb`:
- features, Huber 800, seed 42;
- 10-month fit, holdout Jan+Jul 2025.

## Treatment (from cached predictions)

- **Non-LIRF rows that don't use the median prior:** `echo_prob × clip(offset, 0, 140000) + (1 − echo_prob) × clip(new direct raw, 0, 10800)`. This is the same reconstruction, exact on cached predictions as checked in `reports/base_anchored_preregistration.md`.
- **All other rows** (LIRF, prior rows): unchanged.

## Decision (screen)

`src/models/run_direct_nolirf.py compare` vs the cached `lgb_mixed_nb` base. **Qualifies for the full build iff:**
1. body pooled gain ≥ 1.0 s;
2. Jan > 0 and Jul > 0;
3. full RMSE not worse.

**The full build, if it qualifies:**
- the out-of-fold months and production for LightGBM;
- CatBoost gets the same change;
- the v41 chain;
- its own stage 2 against v41's chain.

## Amendments

(none)

## Results

**Screen** (2026-10-06; logs `logs/base_direct_nolirf_fit.log`, `logs/base_direct_nolirf_compare.log`):
- best_iter 2,594. Reconstruction check exact on 312,885 rows.

| holdout | base | no-LIRF direct | delta |
|---|---|---|---|
| body pooled | 215.86 | 215.55 | −0.30 |
| body Jan | 197.82 | 197.47 | −0.36 |
| body Jul | 229.53 | 229.26 | −0.27 |
| trimmed | 271.99 | 271.97 | −0.02 |
| full | 334.33 | 334.31 | −0.02 |

- 8 of 9 changed airports improve (LEBL −0.99, LFPG −0.88, EGLL −0.83); LTFM +0.33.
- **Does not qualify** (body −0.30 < 1.0).
- Smaller than kind-mango's −5.99 because their gain came mostly from removing labels > 80,000 s, and we already drop labels > 7,200 s.
