# ADS-B corrector block at coverage-stable airports only — pre-registration

Written 2026-10-06, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **v40** (corrector block 9, ADS-B track state; `reports/ctx_adsb_preregistration.md`) gained −1.26 on holdout body and +0.11 on the board.
- **The suspect named in its shift check, before the board result:** the matched-track rate changes a lot between 2025 and 2026 at some airports:
  - EGLL Jul 0.07 → 0.98;
  - LEMD Jan 0.00 → 0.96;
  - EDDM Jul 0.37 → 0.98;
  - LIRF Jul 0.72 → 0.00.
  - There the corrector applies ADS-B splits learned from few or no comparable 2025 rows.
- **At four airports coverage is similar in both years, and v40's holdout gains there were among its largest:**

| airport | v40 body change | matched 2025 (Jan / Jul) | matched 2026 (Jan / Jul) |
|---|---|---|---|
| EHAM | −2.75 | 0.97 / 1.00 | 1.00 / 1.00 |
| LSZH | −2.52 | 0.73 / 0.70 | 0.85 / 0.85 |
| EDDF | −1.50 | 0.68 / 0.73 | 0.50 / 0.63 |
| LEBL | −1.16 | 0.95 / 0.76 | 0.85 / 0.88 |

- **Hypothesis:** the block transfers where coverage is stable. Using it only there keeps the transferable part of v40.

## Disclosure

- This is the second test of the block-9 idea. The airport set comes from the shift-check table (written before the v40 board result), not from any board feedback by airport, which we don't have.
- **A null board result closes the block-9 idea.**

## Treatment

`ctxsel` = the `ctxcorr2rb` corrector output for departures from EHAM, LSZH, EDDF and LEBL, and the `ctxcorr2r` (v37) output everywhere else.
- It is applied identically to the holdout predictions, the five out-of-fold blocks and the ranking set.
- It then goes through v37's chain.
- No other airport sets are tried.

## Test

`tests/ctx_adsb_stable_test.py` (base = v37's chain, `PRC_LGB=ctxcorr2r`).

## Decision

**Adopt iff, vs v37's chain, on holdout body RMSE:** pooled > 0, Jan > 0, Jul > 0.

**Then:** v41 through v37's chain. Keep it iff the board is < 254.01.

## Amendments

(none)

## Results

**Stage 2** (2026-10-06, log `logs/ctx_adsb_stable_test.log`):

| slice | v37 chain | ctxsel |
|---|---|---|
| body pooled | 194.58 | 194.13 (−0.45) |
| body Jan | 176.30 | 175.66 (−0.64) |
| body Jul | 208.31 | 208.00 (−0.32) |
| trimmed | 246.01 | 246.06 (+0.05) |
| full | 304.71 | 304.77 (+0.06) |

- **By airport, the selected four:** EHAM −1.79, LSZH −2.36, EDDF −1.44, LEBL −0.78.
- **Unselected airports move too**, through the refitted combiner and neighbour stages: EDDM +0.96, LIRF +1.13, LFPG −0.62.
- **Rule** (pooled, Jan, Jul > 0) **holds: ADOPT.** The script's printed REJECT is the old ≥ 1.5 rule.

**v41** (`smart-jigsaw_v41.parquet`; logs `logs/v41_*.log`):
- 344,841 rows; IDs match the template; no nulls.
- RMS 24.9 s from v37, 26.3 s from v40.
- 13 rows under 120 s (minimum 6).
- Keep iff board < 254.01; otherwise block 9 is closed.

**Board (2026-10-06): v41 = 253.68 (−0.33 vs v37's 254.01). ADOPTED; new best.**
- About 73% of the −0.45 holdout body gain transferred.
- Consistent with the coverage-shift explanation of v40's failure: the block transfers where 2025 and 2026 ADS-B coverage match.
