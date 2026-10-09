# Design

This document describes the architecture of our taxi-out model for the 2026
PRC / OpenSky Network Data Challenge. The dated history of what was tried,
kept and rejected is in [PROGRESS.md](PROGRESS.md); how to rebuild the
submission is in [REPRODUCE.md](REPRODUCE.md).

## Problem

Predict `TAXITIME_SEC_mvt` (taxi-out, in seconds) for departures at the
challenge airports, scored by RMSE. Training data is every movement
(arrivals and departures) at 11 airports in 2025. The ranking data is
January, July (+ later February and June)  2026, with `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` blanked on departures. Antalya (LTAI) has no departures
in either set, so the target covers 10 airports.

## Core insight: retrospective reconstruction

Takeoff time (`MVT_TIME_UTC_mvt`) is not blanked, and the arrival side of
the ranking data is complete. So the task is reconstructing what happened
after the fact, not forecasting taxi-out before pushback. Three
consequences shape the design:

- **The target can be flipped.** Predict the pushback delay
  `d = AOBT − SOBT` and recover `taxi = (T − SOBT) − d`. The loss is the
  same, but the inductive bias is better. Inbound delay becomes a negative
  predictor of taxi-out.
- **Departures can be linked to their inbound aircraft through stand
  occupancy.** Arrival in-block times are known exactly, so each departure
  can be paired with the arrival that brought its aircraft to the stand.
- **That pairing bounds the pushback time** between the aircraft's own
  in-block and the next occupant's in-block. We validated these bounds:
  they hold (0.1% violation in the high-confidence tier), but they are far
  too wide to constrain predictions (median width ≈ 14× the taxi-out
  standard deviation). The link is used for rotation features instead.

The model therefore uses the realised takeoff time. It is meant for
post-operations analysis and does not transfer to tactical prediction
before pushback.

## Hard invariant: split-blind features

The feature builder never sees a departure's off-block time or taxi label.
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are nulled on every departure,
in training and ranking data alike, before any feature code runs. Labels
live in a separate table keyed on `MVT_ID_mvt` and are joined only at fit
time. `build_features()` takes one frame in the ranking schema, so it cannot
leak by construction.

## Pipeline

**Stage 0 — ingest.** Training and ranking movements are combined per
airport, arrivals and departures together, with a split column.

**Stage 1 — stand linking** (`src/link/stand_link.py`). For each
(airport, stand), an as-of join finds the last same-type arrival before
the estimated off-block and the next arrival after it. Link rate is 99.9%
at every airport; the high-confidence tier covers 92.4% of departures.

**Stage 2 — features** (`src/features/build_features.py`), in families:
- physical baseline: unimpeded taxi-out percentiles per stand group,
  runway, aircraft type and wake category;
- takeoff-anchored congestion: rolling departure and arrival counts,
  inter-departure gaps, saturation runs;
- runway configuration, inferred per 5-minute bin, and time since the last
  change;
- rotation and schedule: inbound delay, scheduled and actual ground time,
  same-airframe flag, stand-link bounds as soft features, EOBT/IOBT deltas;
- weather (METAR);
- categorical and calendar features.

**External data** (see [DATA_SOURCES.md](DATA_SOURCES.md)): ADS-B surface
tracks from adsb.lol, used to detect pushback (`src/link/adsb_pushback.py`);
stand and runway positions from the X-Plane Scenery Gateway; METAR
observations.

**Stage 3 — models.**
- LightGBM and CatBoost base models, stacked with non-negative weights
  fitted across folds.
- A second-stage corrector trained on out-of-fold base predictions.
- Post-stages that use ADS-B pushback detections, stand re-occupation and
  neighbouring flights' errors. See PROGRESS.md for each stage's evidence.

**Ranking-specific stages.** Some post-stages target the ranking period or
data-recording artefacts (notably Rome-Fiumicino records where block time
equals scheduled time) rather than taxi-out itself. PROGRESS.md's reader's
guide lists them; they should not be read as part of a general taxi-out
model.

## Evaluation

- **Holdout:** January + July 2025, models trained on the other ten months.
  Never random k-fold: neighbouring flights share congestion state.
- The holdout frame is built on those two months alone, so gaps in the
  ranking data's timeline are reproduced in validation.
- Changes are gated on the holdout, most with a pre-registration committed
  before results (`reports/*_preregistration.md`).
- Leaderboard scores are used only as keep/reject verdicts on candidates
  that already passed; see the reader's guide in PROGRESS.md.

## Not built

- **Reference (unimpeded) model.** A model on a restricted feature set
  (stand, runway, aircraft type, wake category, configuration; no
  congestion or takeoff-time inputs), fitted to low-traffic periods or a
  low quantile. It would give `excess = observed − unimpeded`, the quantity
  needed for the fuel and CO2 analysis the challenge describes.
- **Distributional output** (quantile predictions).

OSM routed taxi distance from stand to runway was screened before building
and dropped: it is a fixed function of (airport, stand, runway), which the
model already sees, and the screen bounded its gain near zero
(PROGRESS.md §23).

## Compute and conventions

- Developed on two laptops with the same specification: Intel Core Ultra 7
  155H (16 cores / 22 threads), 32 GB RAM, Windows 11, integrated Intel Arc
  graphics (not used for training). GPU work (CatBoost) ran on free-tier
  Google Colab (NVIDIA T4). Polars for features, LightGBM on CPU. No neural
  networks.
- Fixed seeds; LightGBM with `deterministic=true` and `force_row_wise=true`.
- Fetch scripts and small derived tables are committed; caches are local
  and rebuildable from source.
- Every external source has a row in DATA_SOURCES.md: name, URL, licence,
  date accessed, derived artefact and fetch script.
