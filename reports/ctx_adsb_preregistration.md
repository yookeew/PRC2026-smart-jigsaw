# Context corrector, block 9 (ADS-B track state) — pre-registration

Written 2026-10-06, before any result. Changes made after results exist go
under Amendments, dated.

## Why

- **Our gap to outgoing-boat is visible on our own holdout.**
  - Board/holdout-full ratio: ours 254.51 / 304.71 = 0.835, theirs 243.95 / 294.86 = 0.827.
  - So their ~10-point board lead shows up as a ~10-point holdout lead.
  - Measured with their slice definitions on our v36-method chain (`cache/eval/ctxcorr_chain_holdout.parquet`):
    - `sem_loteria` 244.7 vs their 229.2;
    - EDDF full 164.2 vs their 146.0, with no tail rows involved at EDDF on our side;
    - EGLL 258.2 vs 247.2.
- **Where ADS-B enters differs.**
  - Ours: a post-hoc combiner (`src/post/adsb_combiner.py`). It sees airport, hour, the stack and track statistics, but not stand, runway, congestion or any other corrector context.
  - Theirs: ADS-B fields are corrector inputs next to all other context. Their notes report that removing the first-sighting position cost 1.54 on the board.
  - EDDF tracks typically start ~1.9 km from the stand (PROGRESS.md §67). There our partial-track nudge (§41) gives up, but a model that also knows the stand and runway can learn how much of the taxi went unseen.
- The idea is outgoing-boat's; the definitions and code are ours.

## Block 9 (`tests/context_corrector_test.py::adsb_block`, enabled by `CTX_ADSB=1`)

Source: the v4 detector output (`cache/adsb_pushback_v4/`), the same files the production combiner reads. It has one row per departure on every fetched day:
- all days of Jan/Jul 2025 and Jan/Jul 2026;
- ~12 days per month in the other 10 months of 2025.

All inputs are test-visible (ADS-B traces, takeoff time). No departure off-block time and no labels are used. Here `p` is the stack input to the corrector.

| column | definition |
|---|---|
| `ab_avail` | 1 if the departure's day was fetched (detector row exists), else 0 |
| `ab_cov` | `adsb_day_coverage` |
| `ab_matched` | a track was matched to the departure |
| `ab_tier` | pushback detection tier, integer code over `none, appear, dwell, pass, fb_dwell, fb_appear` |
| `ab_fallback` | the fallback matcher was used |
| `ab_taxi`, `ab_taxi_rel` | T − detected pushback, and that minus `p` |
| `ab_L`, `ab_L_rel` | T − first sighting, and that minus `p` |
| `ab_first_gs` | ground speed at the first sighting |
| `ab_first_own_m`, `ab_min_own_m` | distance from own stand at the first sighting / closest approach |
| `ab_n_pts` | track points |
| `ab_takeoff_gap_s` | detected takeoff − T |
| `ab_pb_gap_s`, `ab_pb_dist_m`, `ab_pb_gs` | pushback event gap, distance from stand, speed |

**Not included:** the first sighting's lat/lon. The detector doesn't store it. Distance from own stand plus the corrector's stand prefix and runway is the stand-in. Adding positions needs a detector re-run, which is out of scope here.

**Coverage mix, stated up front.**
- In the 10 training months, ~62% of rows have `ab_avail = 0` (day not fetched).
- On the holdout and on the 2026 ranking set, every row has `ab_avail = 1`.
- So the corrector learns the ADS-B splits from the ~115 fetched training days (~660k rows), which is the same data the production combiner is trained on.

## Shift check (run first, before stage 1 numbers)

- `tests/ctx_adsb_test.py` prints the 2025-vs-2026 standardized mean difference (SMD) per month for every continuous block-9 column, computed **among matched rows only**.
- It also prints the matched rate per airport.
- Matched/tier rates are expected to move: EGLL detection rose from ~3% in 2025 to 22–75% in 2026, a real coverage change.
- **Rule:** any continuous column with |SMD| > 0.25 in either month is dropped from the treatment before stage 1 is run. The drop is recorded under Amendments.

## Test

1. **Stage 1** (`tests/ctx_adsb_test.py`):
   - same data, rows and inner split as `tests/ctx_runway_test.py`;
   - both arms use the adopted ensemble2 corrector, with v37's blocks 1–7;
   - control = blocks 1–7; treatment = blocks 1–7 + block 9.
2. **Stage 2:** `PRC_LGB=ctxcorr2r CTX_ENS=2 CTX_RWY=1 CTX_ADSB=1 tests/oof_context_corrector_stage2.py`.
   - The base is v37's chain, ADS-B combiner included, so any double counting with the combiner shows up here.

## Decision

**Stage 1 — proceed iff, on holdout body RMSE** (NM-matched, y ≤ 1 h),
treatment vs control:
1. pooled gain ≥ 0.5 s;
2. Jan > 0 and Jul > 0;
3. trimmed is not worse.

**Stage 2 — adopt iff, vs v37's chain:**
1. pooled body gain > 0;
2. Jan > 0 and Jul > 0.

**Then:**
- production build (`ctxcorr2rb`), and v39 through v37's chain;
- keep it iff the board is < 254.01.

**If stage 2 fails but stage 1 passed:**
- the likely cause is double counting with the downstream combiner;
- a combiner-off variant would be a new, separately pre-registered test, not a rescue of this one.

## Amendments

(none)

## Results

**Shift check** (2026-10-06, before stage 1; log `logs/ctx_adsb_shift.log`):
- **Among matched rows:** every continuous column has |SMD| ≤ 0.215 (largest: `ab_first_gs`, Jul). Nothing dropped.
- **Matched rate moves a lot by airport**, as expected (2025 → 2026):
  - EGLL Jul 0.07 → 0.98;
  - LEMD Jan 0.00 → 0.96;
  - EDDM Jul 0.37 → 0.98;
  - LIRF Jul 0.72 → 0.00.
- That mix shift is real and can't be tested on the 2025 holdout. It's the main transfer risk of this block.

**Stage 1** (2026-10-06, log `logs/ctx_adsb_test.log`):

| slice | control | treatment |
|---|---|---|
| body pooled | 205.99 | 193.42 (−12.56) |
| body Jan | 188.32 | 174.05 (−14.27) |
| body Jul | 219.35 | 207.90 (−11.46) |
| trimmed | 259.97 | 248.60 (−11.37) |

- The largest gains are at EHAM (−43.9), EDDM (−43.0), LSZH (−33.9), LEBL (−22.3) and EDDF (−16.2).
- LFPG (+0.2) and LTFM (+0.7) have essentially no tracks.
- **PROCEED.**
- **Caveat:** the control has no ADS-B at all, while v37's chain gets ADS-B downstream through the combiner. The treatment body (193.4) is only ~1 s below the v37 chain's (~194.6), so most of this gain is likely already in production. Stage 2 decides.

**Stage 2** (2026-10-06, `PRC_LGB=ctxcorr2r CTX_ENS=2 CTX_RWY=1 CTX_ADSB=1 tests/oof_context_corrector_stage2.py`, log `logs/ctx_adsb_stage2.log`):

| slice | v37 chain | + block 9 |
|---|---|---|
| body pooled | 194.58 | 193.32 (−1.26) |
| body Jan | 176.30 | 175.50 (−0.80) |
| body Jul | 208.31 | 206.73 (−1.58) |
| trimmed | 246.01 | 245.04 (−0.97) |
| full | 304.71 | 303.98 (−0.73) |

- **Per airport:** 9 of 10 airports improve: LIRF −4.25, EHAM −2.75, LSZH −2.52, EDDF −1.50, LEBL −1.16. LEMD moves +0.15.
- **Overlap with the combiner:** about 90% of the stage 1 gain was already in the chain through the downstream combiner, as expected; −1.26 survives.
- **Script verdict:** it printed "REJECT" because it still carries the older ≥ 1.5 rule (same as the runway block).
- **This pre-registration's stage-2 rule** (pooled > 0, Jan > 0, Jul > 0) **holds: ADOPT.**
- **Next:** production build `ctxcorr2rb`, then v40. v39 was a teammate's file (board ~255), so v37 = 254.01 is still the bar.

**Production build** (2026-10-06, logs `logs/ctxcorr2rb_prod.log`, `logs/v40_*.log`):

1. `CTX_ENS=2 CTX_RWY=1 CTX_ADSB=1 src/post/ctx_corrector.py` -> `ctxcorr2rb.parquet`.
   - best_iter 1,616.
   - RMS change vs the stack is 117.3 s, against 80.0 s for v37's corrector. The corrector now carries the ADS-B signal that the combiner used to add later.
2. With `PRC_LGB=ctxcorr2rb`, v37's chain:
   - `stack_submit --cat-corrected --adsb-v3`
   - `adsb_combiner --full`
   - `neighbour_state --v2`
   - `lobt_window`
   - `nm_lines` -> `smart-jigsaw_v40.parquet`

**Checks on v40:**
- 344,841 rows; IDs match the template; no nulls.
- vs v37: RMS 32.2 s, mean +1.3 s, about the same size as v37 → v38 (31.9 s).
  - Largest movers: LIRF 53.6, EHAM 34.4, LSZH 32.5.
- 11 rows under 120 s (minimum 51); v37 had 13.

**Rule:** keep iff board < 254.01 (v37). v39 was a teammate's file (~255, not adopted).

**Board (2026-10-06): v40 = 254.12 (+0.11 vs v37's 254.01). NOT adopted; v37 stays best.**
- The −1.26 holdout body gain did not transfer.
- The most likely cause is the one named in the shift check: the matched-track mix moves a lot between 2025 and 2026 by airport.
  - EGLL Jul 0.07 → 0.98;
  - LEMD Jan 0.00 → 0.96;
  - EDDM Jul 0.37 → 0.98;
  - LIRF Jul 0.72 → 0.00.
- The corrector learned those airports' ADS-B splits from few or no matched 2025 rows.
- Unprovable without 2026 labels. Not rescued; any follow-up needs its own pre-registration.
