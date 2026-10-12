# LIRF NM-unmatched hedge, round 4: flight-number echo history -- pre-registration

Written 2026-10-11 **before v65 was built or uploaded**. Base: v64 = 237.4538 (§140). Team decision 2026-10-11: LIRF is
unfrozen for this one pre-registered change; **option 1: no further 2025 gate, the board decides** (the screens below
already used the 2025 data, and January 2025 has no earlier 2025 history, so a holdout-chain gate cannot test the
January mechanism).

## Disclosed screens (PROGRESS §141; `tests/lirf_round4_scope.py`, `tests/lirf_round4_past.py`)

Harness of §118 (v45 / v48 refitted leave-one-month-out on 2025; scope = v48's treated rows, 1,213 in 2025). Round 4 =
v48's q2 replaced by a logistic on [v48's design, flight-number history], same late-branch bound. History = late (≥ 1 h)
LIRF departures of the same FLIGHT_mvt: NM-unmatched and NM-matched counts and smoothed echo rates (§111).
- Two-sided history ≥ 1 / 3 / 6 months away: in-scope SSE 0.811 / 0.934 / 0.988 (11 / 10 / 8 of 12 months better).
- **Past-only** history (as 2026 has it): ≥ 1 month back (stands in for Jan 2026; Feb–Dec 2025) **0.823, 9/11 months**;
  ≥ 7 months back (stands in for Jul 2026; Aug–Dec) 0.982, 4/5. March is worse in every variant (1.41 past-only).
- 2026 reach: Jan 78 scope-like rows, 59 with a 2025 history (58 from Nov/Dec 2025); Jul 244, 74.
- Projection (1:1 transfer as v47 / v48): ≈ −0.65 s (Jan) − 0.2 s (Jul) ≈ **−0.85 s**, wide uncertainty.

## Rule (v65)

- Classifiers: v45's (`lirf_signals.fit`) and v48's (`lirf_hedge2.fit2`) exactly as in production; **c4** = logistic
  (C = 1, as round 3) on [`lirf_hedge2.design2`, h(unmatched rate) · has, log1p(n_unmatched), h(matched rate) · has,
  log1p(n_matched)], fitted on all 2025 training rows of v48 with **past-only** history (months ≤ own month − 1).
- 2026 history: all late LIRF departures of 2025 with a block time (every 2025 month is in the past).
- Rows: v48's treated rows in 2026 (LIRF NM-unmatched, 1–6 h late, p < 0.99, not arrival twins). On each,
  Q4 = q4 · offc + (1 − q4) · min(P_ne, max(off − 3,606, 0)), P_ne from v47's value with v45's q, exactly as v48.
- Applied to v64 only on rows whose v64 value still equals v48's (rows changed by a later stage stay as v64).
  Every other row = v64.

## Decision

**v65 = v64 + round 4. Keep iff board < 237.4538.** Blind board: if kept, the same rule on `blind_v5`'s
LIRF rows (Jan/Jul = v65; Feb/Jun via the four-month LIRF inputs), batched with the other end-of-run blind changes.

## Outcome (2026-10-11)

Built as our upload **v60** (= v59 + round 4; v59 is the teammate's "v64"), md5 3cbe283ffa83c6f6948efcf3999f5a05.
**Board 237.6439 vs 237.4538: REJECTED.** v59 stays best.
