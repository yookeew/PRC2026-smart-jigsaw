# Stand-free ADS-B pushback fields in the combiner — pre-registration

Written 2026-10-04, before any holdout result with these fields. Changes made
after results exist go under Amendments, dated.

## Why

- **The diagnosis** (v32 method, Jan+Jul 2025 holdout, NM-matched, y ≤ 1 h):
  - At EHAM, rows with a clean stand detection (dwell/appear) score 86.3
    RMSE (median |error| 16 s).
  - The ~13k EHAM rows that have a matched track but no stand detection, or
    only a fallback tier, score ~150–166. They carry ~60% of EHAM's body SSE.
  - EDDF, LEBL and LSZH have the same structure: matched but untiered rows
    are 160 / 192 / 122.
- **The cause:** every tier needs the run to come within 100 m of the own
  stand's coordinates. A track that clearly sits still and then moves gives
  us no pushback if those coordinates are missing or off.
- **What the other team does:** outgoing-boat reads pushback as the run's
  first moving sample (`adsb_taxi_move`), with no stand at all. They score
  102.9 on EHAM body against our 115.8.

## Fields (`src/link/adsb_pushback.py --v6` -> `cache/adsb_pushback_v6/`)

- **Unchanged:** v4's fields and logic (verified on 2025-01-15: identical up
  to floating-point noise).
- **Added on every matched run:**
  - `adsb_stop_end_ts`: end of the last stationary stretch (gs ≤ 1 kt)
    lasting ≥ 180 s;
  - its duration, and its distance to the own stand (when known);
  - the number of ≥ 30 s stops after it;
  - `adsb_move0_ts`: the first moving sample;
  - `adsb_gap_max_s`: the largest sampling gap in the run.

## Test (`tests/adsb_standfree_test.py`)

v27's full-year combiner method, cross-fit Jan ↔ Jul with the OOF-month days,
computed twice:
- **Control:** on v4's inputs.
- **Treatment:** on v6's inputs, with the new fields added as combiner
  features, relative to takeoff: `T − stop_end`, `T − move0`, and both minus
  the stack prediction.

Same rows, same parameters.

**Primary metric: body RMSE** (NM-matched, y ≤ 1 h), on the rows the
combiner serves (rows with ADS-B information).

**Adopt iff:**
1. pooled body gain on served rows ≥ 2 s;
2. the gain holds in both Jan and Jul;
3. trimmed RMSE over all rows is not worse.

**If adopted:**
- rebuild the chain with v6 (stack -> v23 stages -> v6 combiner -> v2
  neighbour corrector -> window -> lines) as the next version;
- keep iff the board < the best at the time.

## Amendments

(none)

## Results (2026-10-04, `tests/adsb_standfree_test.py`, log `logs/adsb_standfree_test.log`)

**REJECT.**
- **Served-row body:** 145.05 -> 144.24 (−0.82; Jan −0.65, Jul −0.94). The
  bar was −2.
- **All rows:** body −0.25, trimmed −0.25.
- **Per airport:** EDDM −2.19, EHAM −1.72; the rest are within ±0.9 (LTFM's
  few served rows +3.7).
- **Why it's small:** a long stop (≥ 180 s) exists on only 9.9% of holdout
  rows. Most tracks begin when the transponder comes on at pushback, so the
  untiered rows don't hide a readable stand dwell.
