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

## Amendment A1 (2026-10-08, after v50's board result, before v51's upload)

- **v50 board: 243.4565 (−1.93 vs v49). Kept.**
- **Follow-up arms, same gate** (controls bit-identical across processes):
  - all2 adds `ro_last_tin_x` (the last arrival's taxi-in minus its 2025 stand/runway median) and `ro_last_land_gap`.
  - e2 makes the combiner the equal-weight mean of a global and a per-airport LightGBM.
  - all2_e2 vs v50's recipe: full −0.18, **body −0.43 (Jan −0.50, Jul −0.38), CI [−0.55, −0.30]**.
  - 2026 projection of the increment: −0.27 (full) / −0.48 (body-only).
  - A previous-same-stand-departure block (all3) added nothing beyond all2 and is not used.
- **v51** = v49 + (treatment − control) for the all2_e2 combiner and the 11-feature neighbour stage, built exactly as v50 (LIRF unchanged, NM-line rows frozen, LOBT window last).
  - vs v50: RMS 22.1 s, mean +0.03.
  - **Keep iff board < 243.4565 (v50).**

**v51 board: 244.0503 (+0.59 vs v50). REJECTED.** The per-airport combiner members (e2) are the suspect: thin-2025-coverage airports, many more 2026 ADS-B rows (the v40 / §86 mechanism).

## Amendment A2 (2026-10-08, before upload): identity of the first arrival at the stand

- **Why:** the first arrival at our stand after SOBT + 5 min is usually our own late inbound, not a re-occupier.
  - 2025, non-LIRF, arrival ≥ 30 min before T: type differs, operator matches and origin ≠ our destination → 77% re-occupier.
  - Type and origin both match → 0.1%.
  - v50's `ro_*` inputs can't tell the two apart.
- **Inputs** (`cache/v53/id_feats.py` → `reocc_feats6.parquet` = v50's 9 `ro_*` columns plus 5 new ones), all label-free and null when there is no such arrival:
  - `fa_tmatch`: type match;
  - `fa_omatch`: operator match;
  - `fa_origin`: arrival ADEP == our ADES;
  - `fa_flnear`: same 2-letter prefix and flight number ±1;
  - `fa_tin_x`: the arrival's taxi-in minus its 2025 stand/runway median.
- **2025 vs 2026:** coverage (6–27% of departures per airport) and match rates agree at every airport. This is movement data only, so ADS-B coverage plays no part.
- **Gate:** v50's recipe + the 5 inputs in the combiner and neighbour stage (b), one change, same process as its control (control bit-identical to the earlier runs).

| | full | trimmed | body | body Jan / Jul |
|---|---|---|---|---|
| all_id vs v50's recipe | −0.35 | −0.44 | −0.41 | −0.51 / −0.34 |

- Body CI [−0.52, −0.29], P(worse) 0.000.
- Per airport: EGLL −1.91, LTFM −0.45, LFPG −0.28, EHAM −0.26, EDDF −0.24; EDDM/LSZH +0.04/+0.05.
- 2026 projection: −0.22 full, −0.16 body-only.
- **Identity-only file** (`cache/v53/identity_only_v50base.parquet`, not uploaded) = v49 + (treatment − control) for the all_id combiner and the 14-feature neighbour stage (`cache/v53/`), built exactly as v50: every LIRF row keeps v49's value, NM-line rows frozen, LOBT window last.
  - vs v50: RMS 16.4 s, mean +0.03 (holdout arm vs control: 15.7 s RMS).
  - No per-airport members (the v51 suspect).
- **Board rule: keep iff board < 243.4565 (v50).**
- **Uploaded form (2026-10-09): v53** = teammate's v52 (OPDI, LSZH only, board 242.7323) + (identity-only file − v50). **Keep iff board < 242.7323.**
- **Board (2026-10-09): v53 = 242.6657 (−0.07 vs v52). Kept.**

## Amendment A3 (2026-10-09, before v54's upload): stand-aware context corrector

- Corrector `ctxc1sel_ro` = both members of ctxc1sel refit with the 11-column stand re-occupation block (`CTX_RO=1`, `tests/context_corrector_test.py`).
- **Gate (PROGRESS §124)** vs ctxc1sel, both through v50's post-stage recipe:
  - full −1.12, body −0.83 (Jan −1.29, Jul −0.51), CI [−1.16, −0.47];
  - every airport except LIRF improves (LIRF is frozen in production);
  - 2026 projection −0.96.
- **v54** = v53 + [(new chain + identity deltas on it) − (old chain + v53's identity deltas)], LIRF frozen, NM-line rows frozen, LOBT window last; vs v53 RMS 30.8 s.
- **Board rule: keep iff board < 242.6657 (v53).**
