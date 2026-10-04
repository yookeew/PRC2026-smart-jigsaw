# LOBT window projection — pre-registration

Written 2026-10-04, before any submission with it. Changes made after results
exist go under Amendments, dated.

## Source of the idea

Read in team `outgoing-boat`'s public repository
(<https://github.com/EnioAguiar/prc-taxiout-2026>, GPLv3, README §1.1,
"LOBT window"). They report that another team found it first. We re-derived
the property on our own data and implemented the projection ourselves
(`src/post/lobt_window.py`); no code was copied.

## The property (measured 2026-10-04, all 2025 labels)

- 2,062,577 of 2,085,047 departures (98.9%) have `LOBT_flt`.
- `|BLOCK − LOBT| ≤ 3,606 s` for **100%** of them; the extreme is exactly
  3,606 s at both ends.
- Most likely an artefact of how the movements were matched to NM flights,
  so it's a hard property of the data, not a tendency. The other team's
  diagnostic submission (their v10) showed it also holds on the 2026
  ranking set.

## Treatment

Clip each prediction into `[max(0, MVT − LOBT − 3606), MVT − LOBT + 3606]`.
Rows without LOBT are unchanged. Projecting onto an interval that contains
the label can't increase any row's error.

## Evidence already in hand (diagnostic, not a gate)

- **LightGBM-with-neighbour-features alone, Jan+Jul 2025 holdout:**
  - 153 rows move;
  - trimmed 271.99 -> 269.02, full 334.33 -> 331.92.
- **OOF months:** 379 rows move; trimmed 231.51 -> 230.35.
- **v31 on the ranking set:** 94 rows outside the window (EHAM 25, LFPG
  28, LTFM 23).
  - The RMS shift over all rows is 19.3 s.
  - So if the window holds, the board gain is at least ≈ 0.7.

## Decision

- **No holdout gate:** the change is a projection onto a hard data
  constraint.
- **v32** = v31 projected.
- **Keep iff the board < 259.89 (v31).** A worse score would mean the
  window doesn't hold in 2026; then the projection is dropped everywhere.

## Amendments

(none)

## Results (2026-10-04)

**Board: v32 = 258.05 (−1.84 vs v31). Kept; new best.** The window holds in 2026.
