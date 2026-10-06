# Corrector member average (v37 + v40 correctors) — pre-registration

Written 2026-10-06, before any averaged prediction or metric was computed.
Changes made after results exist go under Amendments, dated.

## Why

- **outgoing-boat's last gains came from averaging correctors**, not from adopting or rejecting blocks one at a time:
  - v32: two correctors over the same base, −0.65;
  - v36/v37: a third and fourth member with different blocks.
  - Their README says the ensemble exploits "decorrelation of the blocks — each member reads a different part of the airport state".
- **We have never averaged members.** Every corrector block has been an adopt-or-reject switch.
- **v40's corrector is a strongly decorrelated member:**
  - it reads ADS-B, which v37's corrector doesn't;
  - on the holdout the two correctors differ by 68.4 s RMS;
  - on the ranking set the final files differ by 32.2 s RMS.

## Disclosure

- v37 (254.01) and v40 (254.12) both have board scores. Board scores of two submissions plus their RMS difference determine the board score of their equal-weight mean exactly.
- That identity was rejected as a decision tool on 2026-10-06 (PROGRESS.md §82). It isn't used here.
- In full honesty: the implied number was noticed while planning this test. It played no part in choosing the weights or the decision rule, which follow outgoing-boat's equal-weight practice and our standard holdout gate.
- The weights are fixed at ½ / ½ before any result, and the decision is made on the 2025 holdout alone, as for every other change.

## Treatment

`ctxavg` = equal-weight mean of the `ctxcorr2r` (v37) and `ctxcorr2rb` (v40) corrector outputs, applied identically at three levels:
- the holdout predictions (`cache/eval/*_holdout_ev.parquet`, column `pred`);
- the five out-of-fold blocks (`cache/oof/*/fold=*.parquet`);
- the ranking set (`data/submissions/*.parquet`).

It is then fed through v37's chain like any corrector output.

**Excluded members:**
- `ctxcorr2` (v35): its ranking file predates the stand-prefix encoding fix;
- `ctxcorr2ra`: the airline table leaked holdout labels into its gate.

No other weights or member sets are tried.

## Test

`tests/ctx_member_avg_test.py`: builds the averaged files, then runs the stage-2 chain comparison of `tests/oof_context_corrector_stage2.py` with the base = v37's chain (`PRC_LGB=ctxcorr2r`).

## Decision

**Adopt iff, vs v37's chain, on holdout body RMSE** (NM-matched, y ≤ 1 h):
1. pooled gain > 0;
2. Jan > 0 and Jul > 0.

**Then:** v41 = the averaged corrector through v37's chain. Keep it iff the board is < 254.01.

## Amendments

**2026-10-06, before any averaged prediction or metric was computed.** The gate is made stricter.
- **The problem with the original rule:** v40's chain already beats v37's on holdout body (−1.26), so "beats v37" would pass almost by construction. It also wouldn't show any gain from averaging itself.
- **New rule — adopt iff the average beats both members on holdout body:**
  - vs v37's chain: pooled > 0, Jan > 0, Jul > 0;
  - **and** vs v40's chain (`ctxcorr2rb` through the same chain): pooled > 0, Jan > 0, Jul > 0.
- If the average lands between the members, it is not submitted.
- The test script additionally scores the v40 chain on the same holdout rows for this comparison.

## Results

**Result** (2026-10-06, log `logs/ctx_member_avg_test.log`):
- Members differ by 42–48 s RMS on the OOF folds, 68.4 s on the holdout and 90.3 s on the ranking set.

| holdout body | v37 chain | v40 chain | average | avg − v37 | avg − v40 |
|---|---|---|---|---|---|
| pooled | 194.58 | 193.32 | 193.83 | −0.75 | +0.51 |
| Jan | 176.30 | 175.50 | 175.44 | −0.86 | −0.06 |
| Jul | 208.31 | 206.73 | 207.63 | −0.68 | +0.90 |

- The average lands between the members. It beats v37 but not v40, so averaging adds nothing beyond the better member on the holdout.
- **Amended gate: REJECT. Not submitted.**
- `data/submissions/ctxavg.parquet` (corrector level only) is left on disk, unused.
