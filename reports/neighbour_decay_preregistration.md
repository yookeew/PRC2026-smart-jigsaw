# Neighbour-state transfer decay -> v31 strength — pre-registration

Written 2026-10-03, after v30's board result (261.40, adopted) and
before any forward-in-time fit. Changes made after results exist go
under Amendments, dated.

## Why

- **The neighbour stage transfers to the board at about half strength.**
  - v29's implied SSE-optimal scale is ~0.49.
  - v30's is ~0.55, so v30 is still ~2× too strong. Its zero-mean
    within-day correction would score ≈ 260.0 at the right strength.
- **The Jan↔Jul gate can't see this.** Its training rows include OOF days
  from the months next to Jan and Jul (Feb, Jun, Aug, Dec), so it
  measures transfer to an *adjacent* month.
- **Two candidate causes:**
  - (i) the neighbour relationships decay with temporal distance;
  - (ii) something specific to 2026 (e.g. the corrector is fit against
    weaker OOF/10-month base predictions and applied to the 12-month
    production stack).
- This test measures (i) on 2025 alone, with no board and no holdout.

## Test (`tests/neighbour_decay_test.py`)

**Data:** the 115 OOF-month ADS-B days (Feb-Jun, Aug-Dec 2025), with the
OOF stack `s`, the neighbour features of `tests/neighbour_state_test.py`
(same lag, same arrivals baseline), and labels ≤ 5 h.

**Model:** exactly v30's part (b): same inputs, hyperparameters, early
stopping split and fit. The correction is demeaned per (airport, day)
over the rows it's applied to, which are rows without ADS-B information.

**Two directions, equal-sized fits:**

| | fit on | near (1 month away) | far (5-8 months away) |
|---|---|---|---|
| forward | Feb, Mar, Apr | May | Oct, Nov, Dec |
| backward | Oct, Nov, Dec | Sep | Feb, Mar, Apr |

**For each eval set:**
- residual r = taxi − s; correction c̃;
- α = Σc̃·r / Σc̃², the SSE-optimal strength (1 = transfers fully);
- the RMSE gain at α = 1.

## Rule

- **α_far** = the mean of the two directions' far α.
- **Build v31 iff both:**
  1. α_far < 0.8 in both directions (decay is real, not noise); and
  2. α_far > 0.2 (enough signal survives to bother).
- **If built:** v31 = v27 + α_far × (v30 − v27). Only v30's changed rows
  move. Rows with ADS-B information keep v27.
- **Keep iff the board < 261.40 (v30).**
- **Otherwise:** no v31 from this. Cause (ii) becomes the working
  hypothesis and needs its own test.

**Reported:** α_near per direction (should sit near 1 if the decay is
temporal); per-airport α_far.

## Amendments

(none)

## Results (2026-10-03, `tests/neighbour_decay_test.py`, log `logs/neighbour_decay_test.log`; run after the pre-registration commit)

**DECISION: no v31.** Far α is 0.801 (forward) and 2.369 (backward), so
it isn't < 0.8 in both directions.

| direction | fit (rows, best_iter) | near α (RMS change) | far α (RMS change) |
|---|---|---|---|
| forward | Feb-Apr (194,558, 45) | May 0.615 (−0.32%) | Oct-Dec 0.801 (−1.40%) |
| backward | Oct-Dec (173,387, 90) | Sep 1.078 (−0.88%) | Feb-Apr 2.369 (−2.28%) |

- **No temporal decay within 2025.** The far months transfer at least as
  well as the near ones; in the backward direction the correction is
  even too *small* (the 3-month fits are underfit, best_iter 45 / 90).
- **Per-airport far α:** forward LFPG 0.55, EGLL 0.49, the rest ~0.8-1.8;
  backward 0.9-3.8.
- **Conclusion:** within 2025, the neighbour relationships don't decay
  with temporal distance. v29/v30's ~0.5 board scale is 2026-specific,
  i.e. cause (ii): something differs between the 2025 data and 2026, not
  a time-distance effect. Per the rule, no v31 and no shrinkage.
