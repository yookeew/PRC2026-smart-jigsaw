# Combined base change (no-LIRF direct + anchored member) through v41's chain — pre-registration

Written 2026-10-06, before any result for the combination. Changes made after
results exist go under Amendments, dated.

## Why

Two base-level screens today gave small, consistent gains below the 1.0 bar:
- **anchored member** (`reports/base_anchored_preregistration.md`): body −0.58 (Jan −1.14, Jul −0.18);
- **direct regressor without LIRF rows** (`reports/base_direct_nolirf_preregistration.md`): body −0.30 (Jan −0.36, Jul −0.27).

The 1.0 bar was a judgment call. **We have never measured how much of a base-level gain survives the stack, corrector and chain.**

This test:
1. combines the two changes;
2. records the combination's base-level gain;
3. measures its gain through v41's full chain;
4. reports the ratio (the base→chain transfer factor) for future base decisions.

## Treatment: base `lgb_comb_nb`

- **LightGBM members,** all with the `lgb_mixed_nb` features, Huber 800 and seed 42:
  - `direct_nolirf`: the direct regressor without LIRF training rows;
  - `anchored_nolirf`: the anchored regressor (target taxi − clip(MVT − AOBT_3, 0, 7200)) without LIRF training rows.
- **Per row:**
  - **LIRF, and rows served by the median prior (NM-unmatched non-LIRF):** the `lgb_mixed_nb` prediction, unchanged.
  - **Non-LIRF rows with AOBT_3:** `echo_prob × clip(offset, 0, 140000) + (1 − echo_prob) × clip(0.5 × direct_nolirf raw + 0.5 × anchored_nolirf raw, 0, 10800)`.
  - **Any other non-LIRF row:** the same with the direct_nolirf raw only.
- **Coverage:** built for the holdout (10-month fits), the 10 OOF months (month-wise folds, as `lgb_mixed_nb`) and the ranking set (12-month refits).
- **CatBoost (`cat_mixed`)** and everything downstream are unchanged in method.

## Chain

v41's recipe on the new base:
- the `ctxcorr2r` and `ctxcorr2rb` correctors, rebuilt on `lgb_comb_nb` (`PRC_BASE=lgb_comb_nb`);
- `ctxsel` selection (block-9 corrector at EHAM/LSZH/EDDF/LEBL);
- stack, ADS-B blend, combiner, neighbour state, LOBT window, NM lines.

**Comparison:** v41's chain on the same holdout rows (`cache/eval/v41_chain_holdout.parquet`, saved before this test).

## Decision

**Adopt iff, vs v41's chain, on holdout body RMSE:** pooled > 0, Jan > 0, Jul > 0.
- **If adopted:** v42 through the same chain. Keep it iff the board is < 253.68.
- **Recorded either way:**
  - the base-level gain of `lgb_comb_nb` vs `lgb_mixed_nb` (same slices);
  - the chain-level gain;
  - their ratio.

## Amendments

**A1 (2026-10-07, before any result for the combination; the member fits were still running).** v42 (Arm C,
PROGRESS §88) replaced v41 as the best submission while this test was paused. The chain and the comparison
move to v42's recipe; the treatment and the form of the rule are unchanged.
- **Chain:** Arm C's `c1` correctors (`src/post/ctx_armc.py`, NM-unmatched rows + indicator) rebuilt on
  `lgb_comb_nb` (`PRC_BASE=lgb_comb_nb` -> `ctxc1r_comb`, `ctxc1rb_comb`, `ctxc1sel_comb`), then the same stage 2
  as `tests/v42_holdout_baseline.py`; LIRF NM-unmatched rows take v41's chain value, as in v42 (team rule).
- **Comparison:** v42's chain on the same holdout rows (`cache/eval/v42_chain_holdout.parquet`, built in §97
  before this amendment).
- **Decision:** adopt iff, vs v42's chain, holdout body RMSE pooled > 0, Jan > 0, Jul > 0. If adopted, build
  the submission through v42's recipe; keep it iff its board score is below v42's.
- The v41-chain path (`tests/base_comb_test.py`) is not run.

## Results

**Base level** (2026-10-07; `logs/base_comb_combine.log`, `logs/base_comb_compare.log`): `lgb_comb_nb` vs `lgb_mixed_nb`, holdout.
- Rows replaced: 312,885 of 344,339 on the holdout and 313,035 of 344,841 on the ranking set. All of them use the anchored member.

| slice | lgb_mixed_nb | lgb_comb_nb | delta |
|---|---|---|---|
| body pooled | 215.86 | 215.12 | −0.74 |
| body Jan | 197.82 | 196.54 | −1.29 |
| body Jul | 229.53 | 229.17 | −0.36 |
| trimmed | 271.99 | 270.79 | −1.20 |
| full | 334.33 | 333.36 | −0.98 |

- **Per airport (body):** better at 8 of 9 changed airports, from EGLL −1.68 and LFPG −1.47 to EHAM −0.72; LTFM +0.33. LIRF is unchanged by construction.
- The two members alone gave −0.58 (anchored) and −0.30 (no-LIRF direct), so their gains roughly add.

**Chain level vs v42 (the gate):** pending.
