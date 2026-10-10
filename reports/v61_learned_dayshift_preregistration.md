# v61: learned day-shift allocation on top of v60 (+ v58's LSZH rows) -- pre-registration

Written 2026-10-10 **before any number below was computed**. Base: v60 (v55 + EDDF OPDI demeaned, §130), fallback
upload. Continues §130's C1b (shift = s_{airport, delay band} · λ̂, slopes learned on 2025 labels): vs v55's recipe (R0)
it gave full −0.38, trimmed −0.40, 11/12 months better, but body +0.29, all from LTFM's 19 / 23 Feb 2025 snow days.
**Disclosed:** the 2025 data below were already used for C1b; these variants were designed after seeing its failure.

## Fixed parts (as §130 / tests/v57_dsm_c1b.py)

λ̂_g = expectation solve (uniform) per airport-day; 8 airports (not LIRF, not LSZH); periods (Jan, Jul) on the holdout
chain, five two-month OOF periods on the corrector OOF; η and leave-period-out rules unchanged. Slopes by OLS through
the origin of (p − y) on λ̂ over rows with y ≤ 18,000 s in the cell, from the other five periods. A cell with < 500 rows
or < 10 days falls back to the airport's single slope. Bands b of d̂: < 0, 0–15, 15–60, ≥ 60 min.

## Variants (four, fixed now)

- **S1** per-airport slope s_a (no bands).
- **S2** per-(airport, band) slopes in two regimes: normal days |λ̂_g| ≤ 60 s, disrupted days > 60 s (separate slopes).
- **S3** per-(airport, band) slopes clipped to [0, 1] (no amplification beyond the measured day level).
- **S4** per-(airport, band, time of day) slopes, time of day UTC [0, 9), [9, 15), [15, 24).

## Decision (2025, all 12 months, 8 airports, vs R0)

A variant passes iff: (1) pooled full ≤ −0.10 s; (2) pooled body ≤ 0; (3) pooled trimmed (y ≤ 5 h) ≤ 0; (4) full better
in ≥ 8/12 months. **Pick the passing variant with the lowest pooled full.** None passes → no v61 from this; v60 stays
the upload. Reported: per month, per airport, the two holdout-chain months separately (slopes learned on OOF periods
and applied to chain predictions: the transfer test closest to production), and the 2026 projection
(2025 per-airport ΔMSE per row vs R0 × 2026 row counts, onto 239.40).

## Build if a variant passes

v61 = v60 with the 8 airports' rows re-solved: p = v54 + v60's EDDF demeaned OPDI delta (EDDF) / v54 (others),
λ̂_g from the 2026 sums (η from the 2025 holdout chain), δ = s·λ̂ with slopes from all six 2025 periods, LOBT window
last; LSZH rows = v58 (passed in §130); LIRF rows = v55. **Keep iff board < min(239.40, v60's score if known).**

## Outcome (2026-10-10; `tests/v61_learned_dayshift.py`, `logs/v61_learned_dayshift.log`, `logs/v61_s3_info.log`)

vs R0, 2025, 12 months, 8 airports:

| variant | full | body | trimmed | months full better | pass |
|---|---|---|---|---|---|
| S1 airport slope | −0.094 | −0.042 | −0.093 | 9 | no (1: misses −0.10) |
| S2 two regimes | −0.387 | +0.353 | −0.409 | 10 | no (2) |
| **S3 slopes ≤ 1** | **−0.510** | **+0.025** | −0.540 | 11 | **no (2, by 0.025 s)** |
| S4 × time of day | +0.415 | +0.387 | +0.455 | 10 | no |

**No variant passes; no v61 from this rule.** S3's body loss is February (+1.70, LTFM's 19 / 23 Feb snow days) and
January (+0.83, holdout chain, §106); every other month −0.05 to −0.71 (Dec +0.12). Without February: full −0.185, body
−0.110. On the holdout-chain months (slopes learned on OOF periods): Jan full +0.31, Jul −0.22. 2026 projection (information)
S3 −0.39 s (−0.13 without February-type days), S1 −0.07.
