# LIRF NM-unmatched departures: hard bounds from the arrival twin's LOBT — pre-registration

Written 2026-10-07, before the gate script was run and before any board result.
Changes made after results exist go under Amendments, dated.

## Mechanism (label-free in 2026)

- **The join between movements and NM flights is a time window.**
  - Every matched 2025 departure has |BLOCK − LOBT| ≤ 3,606 s (2,062,577 rows).
  - Every matched arrival has |MVT_TIME − ARVT_3| ≤ ~3,606 s.
- **The twin.** A departure can fail the join while the same flight's arrival at
  another of the 10 airports matches, because arrivals match on landing time.
  That arrival row carries the flight's LOBT.
  - Twin = the NM-matched arrival with the same `FLIGHT_mvt`, ADEP and ADES, landing within 8 h after the departure's MVT_TIME; earliest if several.
  - ADES ≠ ADEP.
- **What the failed join implies.** In 2025, 97.5% of twin rows have BLOCK outside
  LOBT_twin ± 3,606. With tl = T − LOBT_twin:
  - **not forced** (tl ≥ 3,726): the block is after the window, so taxi ≤ up = tl − 3,606. Late in 820/825 rows in 2025.
  - **forced** (tl < 3,726): the block can only be before the window, so taxi ≥ lo = tl + 3,606 (unless the row is one of the 2.5% in-window identity failures).

## Exploration already done (disclosed; all 12 months of 2025 seen)

Scratch scripts, 2026-10-07 (PROGRESS §116).
- **All airports, hard bounds, base level:** −5.0e8.
  - LIRF: upper −4.6e8 (11/0 months), lower −1.35e8 (10/0).
  - Other airports: upper neutral, lower +9.6e7 (harmful, the in-window rows).
  - **So the scope is LIRF only.**
- **LIRF variants (106 rows, 12 months):**
  - hard bounds −5.98e8;
  - bounds + dropping the echo component where the bound excludes the echo −7.74e8 (12/0 months).
- **Holdout chain (29 rows):** −2.86e8 of 3.38e8 SSE.
- **The variant was chosen after seeing these numbers.**

## Treatment (variant b)

**Population:** departures with ADEP = LIRF, FLIGHT_ID_mvt null, with a twin.

**Definitions:**
- off = T − SCHED; offc = clip(off, 0, 140,000);
- p = stack echo probability (as `src/post/lirf_signals.py`);
- P_ne = (P − p · offc) / (1 − p) if p < 0.99, else P.

**Rules:**
- **forced** (tl < 3,726): P' = max(P, tl + 3,606).
- **not forced and off > up** (the echo label would break the bound): P' = clip(P_ne, 0, up).
- **not forced otherwise:** P' = min(P, up).

## Gate (`tests/lirf_twin_bounds_test.py`)

- **A.** Holdout chain (v42), pooled full-RMSE delta < 0, and Jan ≤ 0 and Jul ≤ 0.
- **B.** 12 months, base level: SSE lower in ≥ 9/12 months; pooled gain still < 0 without the 3 largest single-row gains.

## Build

- `src/post/lirf_twin_bounds.py smart-jigsaw_v45.parquet smart-jigsaw_v47.parquet`.
- p for 2026 from `lirf_signals.p_2026()`.
- Only rows in scope change.
- **Keep iff the board < 249.0543.**
- **Team rule:** this alters LIRF NM-unmatched rows. The rule was lifted for v45 and this is a separate decision; the build is made, the upload is the team's call.

## Amendments

(none)

## Results

**Gate (2026-10-07, `logs/lirf_twin_bounds_test.log`): PASS (4/4).**
- **A. Holdout chain:** 29 twin rows, 24 moved. Full RMSE 304.075 → 302.706 (**−1.37**); Jan −0.10, Jul −2.58.
- **B. 12 months, base level:** 106 rows, SSE 7.55e9 → 6.77e9; 12/12 months better; still −6.45e8 without the top 3 rows.

**v47 build** (`logs/v47_build.log`): 36 LIRF NM-unmatched 2026 rows have a twin; 27 moved, mean −552 s, Σ(Q−P)² 1.89e8. 27 rows differ from v45.

**Board (2026-10-08): v47 = 247.7092 (−1.35 vs v45's 249.0543). Kept; new best.** ΔSSE −2.30e8, larger than Σ(Q−P)² (1.89e8): the moves were right, and the truths lie beyond them on average.
