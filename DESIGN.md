# Design

This document describes the architecture of our taxi-out model for the 2026 PRC / OpenSky
Network Data Challenge and the reasoning behind its main decisions. How to rebuild the
submission is in [REPRODUCE.md](REPRODUCE.md). The full experiment log, including everything we
rejected, is in [PROGRESS.md](PROGRESS.md).

## Problem

Predict `TAXITIME_SEC_mvt` (taxi-out, in seconds) for departures, scored by RMSE.

- **Training data:** every movement (arrivals and departures) at 11 European airports in 2025.
- **Ranking data:** January and July 2026, plus February and June 2026 for the final ranking.
  `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blanked on departures.
- **Airports scored:** 10. Antalya (LTAI) has no departures in either set.

## Core insight: retrospective reconstruction

The takeoff time (`MVT_TIME_UTC_mvt`) is not blanked, and the arrival side of the ranking data is
complete. So the task is reconstructing what happened after the fact, not forecasting taxi-out
before pushback. For each departure we know:

- the takeoff time `T`;
- the scheduled off-block `S`;
- the network manager's planned and recorded off-block times (LOBT, EOBT, `AOBT_3`);
- everything that happened around it on the airport: other takeoffs, landings, stand occupancy.

Taxi-out is `T − AOBT`, so the problem becomes estimating one unknown timestamp, the off-block,
from a well-constrained context. This has two consequences:

- **Much of the signal is in the planned times and the delay `T − S`.** Pushback delay
  `d = AOBT − S` is often easier to model than taxi itself, and the two are linked exactly:
  `taxi = (T − S) − d`.
- **The model is for post-operations analysis.** It uses the realised takeoff time and does not
  transfer to tactical prediction before pushback.

## Hard invariant: split-blind features

The feature builder never sees a departure's off-block time or taxi label:

- `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are nulled on every departure, in training and
  ranking data alike, before any feature code runs.
- Labels live in a separate table keyed on `MVT_ID_mvt` and are joined only at fit time.
- `build_features()` takes one frame in the ranking schema, so it cannot leak by construction.

Every later stage that adds inputs follows the same rule: only quantities that exist for 2026
rows.

## Architecture

```
features ──► base models (LightGBM, CatBoost) ──► OOF corrector on CatBoost
                                │
                                ▼
          stack + context corrector (out-of-fold, two members, 3 seeds)
                                │
                                ▼
   observation stages: ADS-B pushback, ADS-B combiner, OPDI, neighbour state
                                │
                                ▼
        physical bounds (LOBT window) and NM-unmatched long-delay rules
                                │
                                ▼
          day-level reconciliation with published daily delay sums
                                │
                                ▼
            recording-artefact stages (LIRF echoes, NM-unmatched rows)
```

### 1. Features

Built by `src/features/build_features.py` and `src/features/neighbour_feats.py`, all
label-free. The families are:

- **Physical baseline:** unimpeded taxi-out percentiles per stand group, runway, aircraft type and
  wake category.
- **Congestion anchored on the takeoff time:** rolling departure and arrival counts, gaps between
  departures, saturation runs.
- **Runway configuration**, inferred per 5-minute bin, and the time since the last change.
- **Rotation and schedule:**
  - the inbound flight, linked through stand occupancy (`src/link/stand_link.py`; 99.9% link rate);
  - inbound delay, scheduled and actual ground time;
  - the differences between the planned and recorded off-block times (LOBT, EOBT, IOBT, `AOBT_3`).
- **Weather (METAR):** wind, visibility, precipitation and temperature, which signals de-icing.
- **Neighbour state:** what other flights around the same time and runway show: their recorded
  off-block error, their delay, and arrivals' excess taxi-in.
- **Categorical and calendar** features.

### 2. Base models

- **LightGBM and CatBoost**, trained on all ten airports together.
- **Mixed target.** At LIRF (Rome) the model predicts pushback delay `d` and reconstructs taxi.
  At every other airport it predicts taxi directly.
- **Echo classifier.** Some records have the block time equal to the scheduled time, so the
  recorded taxi absorbs the whole departure delay. We call these "echoes"; they are concentrated
  at LIRF. A classifier gives each row an echo probability `p`, and the prediction is the mixture
  `p·(T − S) + (1 − p)·model`.

### 3. Correctors

- **OOF corrector on CatBoost.** A small Huber LightGBM learns CatBoost's systematic errors from
  month-wise out-of-fold predictions.
- **Stack.** LightGBM and the corrected CatBoost are combined with non-negative weights.
- **Context corrector** (`src/post/ctx_armc.py`). A LightGBM on the stack's residual, trained only
  on out-of-fold predictions, with label-free context blocks:
  - stand prefix and out-of-month cell statistics;
  - consistency of the network-manager times;
  - runway state and stand re-occupation;
  - hour-window and multi-scale runway windows;
  - flow state.

  It has two members, one with ADS-B inputs. The ADS-B member is used only at the four airports
  whose ADS-B coverage is similar in 2025 and 2026 (EHAM, LSZH, EDDF, LEBL). The final corrector
  is the average of fits with three LightGBM seeds.

### 4. Observation stages

- **ADS-B pushback.**
  - A detector (`src/link/adsb_pushback.py`) finds the aircraft leaving its stand in adsb.lol
    surface tracks, using stand coordinates from the X-Plane Scenery Gateway, plus 50 positions
    we inferred from ADS-B for stands the Gateway lacks.
  - A learned combiner merges the observed pushback with the model on rows that have ADS-B
    information.
  - It is trained on the 2025 holdout plus a sample of ADS-B days from every other month.
- **OPDI ground events** (EUROCONTROL) at LSZH, where coverage is high, and at EDDF, where they are
  used demeaned per day because of a clock offset.
- **Neighbour-state corrector.** On rows without ADS-B it uses neighbouring flights' observed
  errors. Its correction is demeaned per airport and UTC day.

### 5. Physical bounds

The off-block is clipped to `|BLOCK − LOBT| ≤ 3,606 s`. This held on every 2025 departure that
has a LOBT.

NM-unmatched departures more than 6 h late follow fitted lines in `T − S`. At airports other than
LIRF these lines are removed again, because the corrector predicts those rows better.

### 6. Day-level reconciliation

EUROCONTROL publishes, per airport and day:

- the sum of pre-departure delay, Σ max(0, AOBT − SOBT);
- the number of departures.

In 2025 these sums equal the sum over our labels. For 2026 we:

1. solve for each airport-day the shift λ that makes our predictions match the published sum;
2. give each flight a share of it, from a slope model learned on 2025 (by delay band, hour,
   ADS-B / OPDI coverage, size of λ).

LSZH gets its own day-level variant. LIRF is left out: a LIRF version that used the daily sums
to estimate per-row echo probabilities was tried and dropped, because it did not hold up.

### 7. Recording-artefact stages

These target how the ranking labels were recorded rather than taxi-out itself. Most concern LIRF
departures that the network manager could not match (NM-unmatched). For those rows the label is
one of a few cases:

- an echo of the schedule;
- a real pushback;
- a block time logged in local time (+1 h in winter, +2 h in summer);
- a date one day off.

These stages:

- estimate each row's probability of each case, from label-free evidence:
  - published daily departure counts;
  - arrival "twins" that bound the off-block;
  - the airline's 2025 echo record;
  - the flight-number format;
- then predict the expected value.

The rule for each case is fitted on 2025 and checked month by month.

## Key decisions and rationale

**Mixed target.** Predicting `d` gives a better inductive bias where the label is tied to the
schedule (LIRF's echoes and long delays). Predicting taxi directly is better wherever taxi is
mostly physical. On the holdout, the direct target beat the delay target at 8 of 10 airports but
lost clearly at LIRF; the mixed model matched the better one at every airport. A side effect: a
feature can help one target and hurt the other, so every change was tested under the mixed
target, the one we submit.

**Out-of-fold training for every second stage.** A corrector trained on the base model's
in-sample predictions learns a residual distribution the 2026 rows will never have. Every second
stage is trained on predictions the base model made for months it did not see:

- the CatBoost corrector;
- the context corrector;
- the ADS-B combiner;
- the neighbour stage.

Within each fold, priors, encodings and early stopping also exclude the held-out month.

**Errors are shared in time.** Neighbouring departures share much of their error: the same
disruption, de-icing or runway closure. This drove three stages:

- the neighbour-state stage, which uses label-free signals from nearby flights;
- the context corrector's time-window blocks;
- the day-level reconciliation, which uses the one label aggregate published for 2026.

The neighbour stage is demeaned per airport-day, so it can redistribute error within a day but
cannot move an airport's level. Only the day-level stage, anchored to published data, moves the
level.

**2025 → 2026 coverage shifts.** ADS-B coverage changed a lot between the two years at several
airports (receiver growth; LEMD, EGLL and LIRF in opposite directions). Fits that learn
per-airport ADS-B behaviour from thin 2025 data generalised badly. So:

- ADS-B-dependent pieces are used only where coverage is similar in both years;
- per-airport ADS-B models are avoided;
- every new input is checked for 2025-vs-2026 drift before it is gated.

**RMSE is dominated by a few labels.** Some labels are hours long: date errors, echoes, real
disruption days. A handful of them can decide a holdout comparison. So:

- changes are judged on a fixed trimmed metric (labels ≤ 5 h) and on "body" rows (NM-matched,
  ≤ 1 h), with full RMSE as a guard;
- where a row's label is one of a few discrete cases, we predict the probability-weighted
  expected value. Under squared error that beats committing to one case.

**Recording artefacts are part of the target.** The LIRF echoes and the NM-unmatched long-delay
rows hold a large share of the squared error, and no physical model can predict them. Treating
them as recording processes, with rules fitted on 2025 and validated out of sample, was worth more
than further model capacity.

**Deltas instead of rebuilds.** Refits are noisy (two CatBoost GPU runs differ by ~27 s RMS). So a
new stage is built as (treatment − control) with both fitted in the same process, and added to
the previous best file. Rows where a change did not transfer to 2026 keep their earlier values;
in particular, LIRF NM-matched rows are frozen after the hedges. Averaging the corrector over
three seeds serves the same purpose: it lowers the noise of each delta.

**Stand linking gives features, not bounds.** Pairing each departure with its inbound aircraft
should bound the off-block between the inbound's in-block and the next occupant's in-block. The
bounds hold (0.1% violations in the high-confidence tier), but their median width is about 14
times the taxi-out standard deviation, so they constrain nothing. The link is kept for rotation
features: inbound delay and ground time.

**Leaderboard use.** Every change is gated on the 2025 holdout, most with a pre-registration
(`reports/*_preregistration.md`) written before the result. A change that passed was uploaded
with a keep/reject rule set in advance. Board scores were used only as that verdict, never to fit
values, weights or scales.

## Validation

- **Holdout:** January + July 2025, models trained on the other ten months. These months match
  the ranking period's season.
- **No random k-fold:** neighbouring flights share congestion state, so random folds would leak
  it.
- **Isolated holdout frame:** the holdout features are built on a frame of those two months alone,
  so the gap in the ranking data's timeline (and features that cross it) is reproduced in
  validation.
- **Out-of-fold check:** stages that touch the final ranking's February and June rows were also
  checked on Feb/Jun 2025 out of fold.
- **Uncertainty:** a paired (airport, day) cluster bootstrap (`tests/_harness.py`).

## Not built

- **Reference (unimpeded) model:** a model on a restricted feature set (stand, runway, aircraft
  type, wake category, configuration; no congestion or takeoff-time inputs), fitted to
  low-traffic periods or a low quantile. It would give `excess = observed − unimpeded`, the
  quantity needed for the fuel and CO2 analysis the challenge describes.
- **Distributional output** (quantile predictions).
- **OSM routed taxi distance:** it is a fixed function of (airport, stand, runway), which the model
  already sees. Stand and runway geometry added nothing as features.

## Compute and conventions

- Developed on two laptops with the same specification: Intel Core Ultra 7 155H (16 cores /
  22 threads), 32 GB RAM, Windows 11, integrated Intel Arc graphics (not used for training).
  GPU work (CatBoost) ran on free-tier Google Colab (NVIDIA T4). Polars for features, LightGBM on
  CPU. The final model has no neural networks (an MLP corrector member was tried and rejected,
  PROGRESS.md §146).
- Fixed seeds; LightGBM with `deterministic=true` and `force_row_wise=true`.
- Fetch scripts and small derived tables are committed; caches are local and rebuildable from
  source.
- Every external source has a row in [DATA_SOURCES.md](DATA_SOURCES.md): name, URL, licence, date
  accessed, derived artefact and fetch script.
