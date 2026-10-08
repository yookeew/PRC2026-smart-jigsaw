# v50: stand re-occupation + ADS-B first-sighting position (holdout gate, 2026-10-08)

Written after the holdout gate and before any board upload. The board rule below is fixed before the upload.

## Idea

Two label-free information sources that the chain never used:

1. **Stand re-occupation.** The Stage 1 linker (`src/link/stand_link.py`) defines `A_next` as the first arrival
   after `T - 720 s`, so `L_sec` is capped at 12 min. It is blind to a stand that another aircraft takes while the
   departure is still taxiing. In 2025 that happens on 9.8% of departures (EGLL 26%), on 27% of 30–60 min taxis
   and 36% of 1–2 h taxis. When it happens, the re-occupying arrival's in-block is a hard lower bound on the label.
   As a hard clip it fails: when the last arrival came ≥ 30 min before takeoff, re-occupation is rare and no
   label-free rule identifies it reliably (precision 22–55%). So the information enters the post-stages as
   features and the models weigh it.
2. **ADS-B first-sighting position.** The combiner had the first-sighting time, speed and straight-line distance
   to the own stand, but not where on the airport the aircraft was first seen. On untiered tracks the unknown is the
   time between pushback and that first sighting, which depends on the route geometry.
   outgoing-boat's README reports that removing the same information cost +1.54 s on their board (their v14).

## Inputs (all label-free, identical construction in 2025 and 2026)

- `ro_*` (scratch builder `reocc_feats.py`): from arrival in-blocks, departure takeoffs, stands, types and
  operators only:
  - gap from the last arrival at the stand to takeoff;
  - type and operator match of that arrival with the departure and with the next departure;
  - whether two departures share one arrival gap;
  - gap to the previous arrival and its type match;
  - arrivals at the stand between SOBT + 5 min and T, and the earliest of them.
  - Links further than 24 h away are treated as absent (the ranking frame jumps 31 Jan → 1 Jul).
  - 2025 and 2026 distributions agree after that guard.
- `adsb_first_x/y/dx/dy`: a scratch copy of the v4 detector that also writes the run's first sample position.
  Airport frame centred on the mean stand; dx/dy are the offset from the own stand. All v4 columns are identical
  to `cache/adsb_pushback_v4` (checked on 4 days; only ~1e-10 m float noise in three distance fields).

## Gate

v42's holdout chain (ctxc1sel through v30's method, LOBT window, NM lines; LIRF NM-unmatched rows from v41's chain).
Every arm runs in the same process as its control, so refit noise cancels. Control = production inputs.

| arm | where | full | trimmed | body | body Jan | body Jul | body 95% CI |
|---|---|---|---|---|---|---|---|
| pos | combiner + position | −0.19 | −0.24 | −0.26 | −0.30 | −0.23 | [−0.30, −0.21] |
| ro_c | combiner + re-occupation | −0.40 | −0.50 | −0.27 | −0.14 | −0.36 | [−0.41, −0.13] |
| ro_b | neighbour stage (b) + re-occupation | −1.20 | −1.48 | −0.92 | −0.17 | −1.44 | [−1.22, −0.65] |
| e2 | combiner = mean(global, per-airport) | −0.02 | −0.02 | −0.08 | −0.01 | −0.14 | [−0.13, −0.04] |
| **all** | pos + ro_c + ro_b | **−1.72** | **−2.12** | **−1.38** | **−0.52** | **−1.98** | [−1.75, −1.07] |

Notes:
- **all:** the body share of the full gain is +0.94 s, and the top 20 rows hold 49% of the body gain.
- **all, per airport:** EGLL −8.5, LTFM −3.7, EDDF −2.0, LSZH −1.7, EHAM −1.7, EDDM −1.5, LFPG −1.0, LEBL −0.9,
  LEMD 0.0, **LIRF +0.5**.
- **Projection onto the 2026 composition** (`reweight26.py`): holdout ΔSSE per row in each (airport, month,
  ADS-B information yes/no) cell, times the 2026 rows in that cell.
  - **all: −1.64 s on the board, −1.87 s with LIRF rows unchanged; body-only cells −0.9 to −1.0 s.**
  - EGLL moves from the neighbour stage to the combiner in 2026 (July rows with ADS-B: 3.2k → 20.4k).

## Candidate v50

- v49 + (treatment − control) of the combiner (all inputs) on rows with ADS-B information (non-LIRF), plus
  (treatment − control) of the neighbour stage (b) on rows without it.
- Both are fit in one process on the production rows (`build_comb_delta.py`, `build_nb_delta.py`, PRC_LGB=ctxc1sel).
- Deltas are zero on NM-unmatched rows more than 6 h late (the NM lines override them).
- **Every LIRF row keeps v49's value.**
  - Deviation chosen after seeing the gate's LIRF +0.5: LIRF echo labels do not obey stand physics (§102), and
    the team rule keeps LIRF NM-unmatched rows fixed.
- The LOBT window is applied last. This also fixes 3 LSZH rows that v49's level shift left 16–40 s outside it.

**Board rule: keep iff board < 245.3839 (v49).**
