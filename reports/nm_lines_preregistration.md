# Lines for long-delayed NM-unmatched departures — pre-registration

Written 2026-10-04, before the v33 upload. Changes made after results exist go
under Amendments, dated.

## Source

The idea comes from team `outgoing-boat`'s public repository
(<https://github.com/EnioAguiar/prc-taxiout-2026>, GPLv3).
- **Lines:** `TwoStageNM`, `nm_min_ms = 21600`.
- **Rome rule:** `pos_regras.roma`.
- **Their board history:**
  - lines on all NM-unmatched rows: 338.67 -> 337.18;
  - lines only where delay > 6 h: -> 330.95 (−6.2);
  - Rome rule: 264.74 -> 264.35 (−0.39, 4 rows).

The implementation is ours (`src/post/nm_lines.py`), and every line is fitted on
our own data. Their fitted constants aren't used.

## Treatment

Applied to v32. Here d = MVT − SCHED.
- **Lines:** for NM-unmatched departures with d > 6 h, predict a + b·d.
  - (a, b) is fitted per airport by least squares on all NM-unmatched
    departures.
  - Airports with fewer than 50 rows use the global line.
- **Rome:** for LIRF NM-unmatched departures with d in (15 h, 30 h], predict
  a + b·d.
  - The line is fitted on those rows only. In 2025 they're either exact SCHED
    copies or ~87,000 s labels.
  - It overrides the airport line.
- **Fitting:** on the 10 training months for the holdout check, on all 12
  months for the ranking.

## Holdout check (`logs/v33_nm_lines.log`)

v32's method on Jan+Jul 2025, 10-month fit:

| | v32's method | + lines + Rome |
|---|---|---|
| full | 316.76 | 307.02 |
| trimmed (≤ 5 h) | 248.81 | 248.84 |
| Jan | 337.31 | 333.37 |
| Jul | 299.19 | 284.01 |
| body (y ≤ 1 h) | 218.72 | 218.77 |

52 line rows, 7 Rome rows.

## Risk, stated before the result

- **The entire gain is in the tail.** Our §46 and §55 lessons say tail rules
  learned on 2025 often fail on the board (v20, v22). The other team's own
  transfer study also finds that tail gains are anti-correlated with board gains.
- **What argues the other way:** this exact rule family moved *their* board by
  −6.2 and −0.39 on the same 2026 test set. So it's the one tail rule with
  direct evidence that it transfers.

## Decision

- **v33** = v32 + lines + Rome.
- **Keep iff the board < 258.05 (v32).**

## Amendments

(none)

## Results (2026-10-04)

**Board: v33 = 257.35 (−0.69 vs v32). Kept; new best.** About 7% of the holdout gain (−9.7) transferred.
