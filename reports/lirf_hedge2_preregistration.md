# LIRF NM-unmatched hedge, round 2: airline matched-rate signal + late-branch bound — pre-registration

Written 2026-10-08, before the gate script was run and before any board result.
Changes made after results exist go under Amendments, dated.

## Mechanism

The NM join is a time window (PROGRESS §116). In 2025, a LIRF NM-unmatched
departure that is ≥ 1 h late carries one of two labels:
- **echo** (block = SCHED);
- **late pushback with a stale LOBT ≈ SCHED**, so taxi ≤ off − 3,606.

Together these cover 97–99% of rows in every 1–6 h band (§116 scoping).

1. **Late-branch bound.** The hedge's non-echo branch should never exceed
   off − 3,606. In v45/v47 it does on 35–39% of in-scope rows.
2. **Airline matched rate (label-free).** Per airline (`FLIGHT_mvt` prefix,
   3 characters): the share of its LIRF departures ≥ 1 h late that are NM-matched,
   counted over other periods.
   - It separates echo strongly: unmatched rows of airlines with a rate of 0.6–0.8 are echoes 95% of the time (p 0.70); those of airlines above 0.95, 19% (p 0.41).

## Exploration already done (disclosed; all 12 months of 2025 seen)

Scratch scripts, 2026-10-08 (PROGRESS §117). Leave-one-month-out, base level, in-scope SSE:
- v45 signals 0.947 × base;
- adding the matched rate: 0.921 (8/12 months better than v45 signals);
- the bound on top of v45: −1.25% (9/12 months).
- Holdout chain, bound only: Jul −1.5e8, Jan −0.8e6.

## Treatment

**Scope:** as v45 (LIRF, NM-unmatched, 1 h ≤ off < 6 h, p < 0.99), excluding rows with an arrival twin (v47's bounds stay).

**Features:**
- `rate` = NM-matched share of the airline's LIRF departures with off ≥ 1 h, computed from the history:
  - 2025 rows: the other 11 months;
  - 2026 rows: all 12 months of 2025.
- `few` = 1 when the history has fewer than 5 such departures (then rate = 0.75);
- `lN` = log1p(history count).

**Classifier:** as v45 (`lirf_signals.design`) plus [rate, few, lN]. Same
LogisticRegression; training rows as v45; 12-month fit for 2026.

**Prediction.** P_ne is recovered from the input file with v45's q (P_in = q45 · offc + (1 − q45) · P_ne); then:

P' = q2 · offc + (1 − q2) · min(P_ne, max(off − 3,606, 0)).

## Gate (`tests/lirf_hedge2_test.py`)

- **A.** Holdout chain: v42's holdout chain with v45's treatment as the comparison base (10-month fits), against the same with this treatment. Pooled full RMSE < 0; Jan ≤ 0 and Jul ≤ 0.
- **B.** Leave-one-month-out at base level (base = v45 treatment): in-scope SSE lower in ≥ 8/12 months, pooled lower, and still lower without the 3 largest single-row gains.

The 8/12 bar is set at the exploration's level; it's disclosed as such.

## Build

- `src/post/lirf_hedge2.py smart-jigsaw_v47.parquet smart-jigsaw_v48.parquet`.
- Only in-scope non-twin rows change.
- **Label-free drift check (reported):** per-airline matched rate 2025 vs Jan+Jul 2026.
- **Keep iff the board < 247.7092 (v47).**

## Amendments

(none)

## Results

**Gate (2026-10-08, `logs/lirf_hedge2_test.log`): PASS (4/4).**
- **A. Holdout chain:** 328 scope rows. Full RMSE 301.966 → 300.183 (**−1.78**); Jan −0.50, Jul −3.03.
- **B. Leave-one-month-out:** in-scope SSE 9.93e9 → 9.06e9 (0.912); 11/12 months better (Feb 1.11 worse); still −7.6e8 without the top 3.

**v48 build** (`logs/v48_build.log`):
- 287 non-twin scope rows; 284 moved. Mean +12 s, RMS on scope 777 s, Σ(Q−P)² 1.73e8.
- **Drift:** per-airline matched rate 2025 vs 2026, corr 0.54 (54 airlines); means 0.921 / 0.911.
- **Keep iff the board < 247.7092.**

**Board (2026-10-08): v48 = 245.8701 (−1.84 vs v47's 247.7092). Kept; new best.** The holdout gain was −1.78, so transfer was again about one to one.
