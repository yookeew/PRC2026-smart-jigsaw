# ADS-B detector fields as base-model features — pre-registration

Written 2026-10-04, before any model was fit on these features. Changes made
after results exist go under Amendments, dated.

## Why

- **ADS-B has never been a base-model feature for us.** It arrived (§29,
  §37) when only Jan/Jul had tracks, so it was fitted after the base model
  on the holdout months: the blend, partial-track estimate, combiner and
  neighbour stage.
- **Team outgoing-boat puts ADS-B in the base model:**
  - their v6 moved the board by −16;
  - dropping ADS-B from their corrector cost only 0.17, so for them its
    value sits in the base;
  - their public repo, README §1.4.
- **Our ordinary-flight gap to them is concentrated at the ADS-B airports**
  (holdout, NM-matched, y ≤ 1 h, vs their v28): EDDF +17, EHAM +13, EGLL +11.
- **Ruled out today** (diagnostics, scratch scripts):
  - track features: the combiner already has theirs;
  - match coverage: ours is ≥ theirs;
  - more training days: the combiner's learning curve flattens;
  - context in the combiner: slightly worse.

  Where ADS-B enters is the remaining structural difference.

## Features (`src/features/adsb_base_feats.py` -> `cache/features_nb_adsb/`)

- **Source:** `cache/features_nb/` plus the v4 detector output
  (`cache/adsb_pushback_v4/`), joined on MVT_ID.
- **Columns:**
  - `adsb_day` (1 if the day has a detector file);
  - matched, fallback and day-coverage flags;
  - tier code;
  - `adsb_taxi = T − pushback_ts`, `adsb_L = T − first_ts`;
  - takeoff gap, point count, first speed;
  - first and minimum distance to own stand;
  - pushback gap, distance and speed.
- **No label is read, and the leak invariant holds:** ADS-B is an external
  observation, and the ranking has it on every day.
- **Coverage mismatch, stated up front:** training months have tracks on
  ~12 days each (~40% of rows); the holdout and 2026 have every day.

## Stage A — base-level screen (decides whether stage B runs)

`run(engine="lgb", target="mixed", feat_dir=cache/features_nb_adsb)`: the
10-month fit, predicting Jan+Jul 2025. It's compared with `lgb_mixed_nb` (same
settings, without the ADS-B columns), LightGBM alone, after the LOBT window.

- **Primary metric: body RMSE** (NM-matched, y ≤ 1 h). This is the slice
  whose gains have transferred to the board (§76 lesson; outgoing-boat's
  transfer study).
- **Proceed to stage B iff:**
  1. pooled body gain ≥ 5 s;
  2. the gain holds in Jan and in Jul;
  3. trimmed RMSE is not worse.
- **Also reported:** body at the ADS-B airports, compared with the v32
  chain's body there. That shows how much of the post-hoc stages' work the
  base now does by itself.

## Stage B — chain gate (only if stage A passes; pre-registered separately)

- The 10 OOF folds and the chain rebuild.
- The decision is on chain-level body RMSE, with the trimmed and full rules
  as guards.
- If the gain survives, the full-year ADS-B download is reconsidered: in this
  setup, training coverage would matter.

## Amendments

(none)

## Results — stage A (2026-10-04, `tests/adsb_base_test.py`, log `logs/adsb_base_test.log`)

**PASS.** The LightGBM alone, with ADS-B base features, after the LOBT window:

| | without ADS-B | with ADS-B | change |
|---|---|---|---|
| body, pooled | 215.71 | 206.00 | −9.70 |
| body, Jan | | | −10.49 |
| body, Jul | | | −9.20 |
| trimmed | | | −8.22 |
| full | | | −6.92 |

- **Per airport (body):** EHAM −32.9, EDDM −32.2, LSZH −27.7, LEBL −19.8,
  EDDF −11.3; LFPG and LTFM flat.
- **Still worse than the v32 chain** on body: 206.0 vs 199.6.

**Chain preview (diagnostic, not the stage-B gate).** The v30 chain plus the
window, with the post-hoc stages fit on the holdout only (no OOF days), both
arms (scratch script):
- body 202.56 -> 201.66 (−0.90; Jan −0.77, Jul −1.00);
- trimmed −0.36, full −0.77.

The post-hoc ADS-B stages already capture almost all of what the base gains
from ADS-B.
