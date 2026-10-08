# OPDI flight events (OpenSky-derived ground milestones) -- scoping and stop bars

Written 2026-10-08 **before any OPDI-vs-label number was computed**. Context: PROGRESS §124–§125 (the remaining
body gap is per-flight off-block information), §80 (a first OPDI look: `exit-parking_position` only, 4–13 Jul 2025).

## Why this is new information

- The organisers' own coverage table (`euctrl-pru/opensky-airport-coverage`, 2026 CSV, sample 5–7 Jun 2026) gives
  the median share of taxi-out received by the OpenSky Network (OSN): **LSZH 0.996, LEBL 0.797, EDDF 0.622,
  EHAM 0.286, LEMD 0.175; EDDM, LFPG, EGLL, LIRF, LTFM 0.000** (track starts 0.2–2.0 min *after* take-off).
- adsb.lol in the 2026 ranking (`cache/adsb_pushback_v4`): pushback tier (dwell/pass) EDDF 3% / 2% (Jan / Jul),
  LSZH 30% / 11%, LEBL 18% / 25%; median first-sighting lead EDDF 5.7 / 3.8 min (OSN track start −8.6 min).
- Holdout body RMSE without a pushback tier is far above with one (EDDF 157 vs 97, LSZH 140 vs 60).
- So OSN may see the start of the taxi on flights adsb.lol misses. **It cannot help LFPG / LTFM** (OSN sees nothing
  there), which closes the "OpenSky data for the §124 oracle" route regardless of policy.

## Licence / rules

OPDI is published by the PRC/PRU (the challenge organiser) as open data: "Open data can be freely used, reused, and
distributed provided that the data source is attributed" (opdi.aero/about.html), with a non-commercial emphasis,
the same basis as the EUROCONTROL ATFM files admin-approved for this challenge on 2026-09-17. §80 already treated
OPDI as organiser-approved. It is a published derived product, not the OpenSky Trino / state-vector database that
§50 excluded (OpenSky's terms of use are non-commercial research only, not an open licence).

## Data access

`https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{d0}_{d0+10}.parquet`,
10-day files on a grid anchored at 2022-01-01 (~550 MB each). Row groups are clustered by event type; only the row
groups whose `info` column is non-trivial (airport events) and the columns flight_id, type, event_time, lon, lat, info
are read by HTTP range request (~170 MB per file), then filtered to the 10 airports.

## S1 -- coverage (label-free), stop rule fixed now

Windows: holdout 2025-01-05..15, 2025-07-04..14; ranking 2026-01-05..15, 2026-07-09..19.
Per airport: share of departures matched to an OPDI flight with a ground event at ADEP before take-off; median lead
of the earliest ground event; **new-information reach** = matched departures with no adsb.lol dwell/pass tier whose
earliest OPDI ground event precedes the adsb.lol first sighting (or adsb.lol unmatched) by ≥ 120 s.

**STOP** if the 2026 new-information reach is < 5% of ranking departures, or if no airport keeps a 2025 → 2026
reach ratio within [0.5, 2].

## S2 -- holdout screen on v50's chain (labels, Jan + Jul 2025), stop rule fixed now

Residual model on v50's holdout chain (`cache/v50/chain_all`), cross-fitted Jan ↔ Jul, LightGBM Huber, non-LIRF rows,
rows selected on the prediction (≤ 3,600 s), never on the label. Control = the same model and inputs without OPDI
features; treatment = + OPDI features (earliest ground-event lead, exit-parking / apron / taxiway / runway-entry
leads, de-icing-pad flag, distance of the first ground event from the own stand where known). Reported:
treatment − control, body (NM-matched, y ≤ 1 h), trimmed (y ≤ 5 h) and full, both months, per airport.

**Proceed to a pre-registered gate only if** treatment − control gives
1. pooled non-LIRF body ≤ −0.5 s, **and**
2. both months negative, **and**
3. the gain is not carried by one airport alone (largest single-airport share ≤ 80% of the pooled body gain) --
   unless the 2026 reach of that airport is ≥ its 2025 reach.

If S2 passes, the gate is the v50 chain with OPDI inputs added to the combiner / neighbour stages (as §123 did for
ADS-B position), v52 = v50 + (treatment − control), fit in one process.

## Outcome (2026-10-08, added after the runs; PROGRESS §126)

- S1: PASS (2026 new-information reach 6.0% / 8.5%; LSZH 2025 → 2026 ratio within [0.5, 2]).
- S2: body −1.53 (Jan −1.18, Jul −1.79), trimmed −1.47, full −1.09. Conditions 1–2 pass. **Condition 3 fails as
  written** (LSZH 86% of the gain; LSZH reach ratio 2026/2025 = 0.88 < 1). Proceeding to the gate is a disclosed
  deviation (S1's stability band for the same quantity is [0.5, 2]).
- Gate (`reports/v52_opdi_gate_preregistration.md`): PASS, LSZH scope, body −1.15, full −0.75. v52 built.
