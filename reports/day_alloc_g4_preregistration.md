# G4: day-shift slope model with day-type inputs -- pre-registration

Written 2026-10-10 **before any number below was computed**, after G3 failed (reports/day_alloc_g3_preregistration.md).
Base: v63 = 238.3173 (G2x8, §133). Same 2025 data as §130–§133 (disclosed).

## Idea

The slope that turns a day's level error into per-flight shifts may depend on the kind of delay day: ATC/airport-
restricted days and ATFM-regulated days put the delay at the gate in a different pattern from airline/turnaround days.
Two published day-level inputs, label-free at row level, both EUROCONTROL files already in `external-data/` and in
DATA_SOURCES.md (admin-cleared):
- `atc_share` = DLY_ATC_PRE_2 / DLY_ALL_PRE_2 (ATC pre-departure delay / all-causes pre-departure delay), clipped to [0, 1];
- `reg_share` = FLT_DEP_REG_1 / FLT_DEP_1 (ATFM slot adherence file: regulated departures share).
Keyed on (airport, predicted off-block date) as λ̂ is.

## Model, decision, build

G4 = G2 (`tests/day_alloc_gbm.py`, same hyper-parameters, label, weight, clip, λ̂, periods, masks) + the two inputs.
One candidate: **G4 at all 8 airports**. Pass iff vs G2x8: pooled full ≤ −0.10, body ≤ 0, trimmed ≤ 0, ≥ 8/12 months
full better, February and June each not worse. **And, given G3: full without February ≤ −0.05** (so a gain that is only
LTFM's February 2025 cannot pass). Build if it passes: `src/post/day_alloc_build.py` with G4; keep iff board < 238.3173.

## Outcome (2026-10-10; `tests/day_alloc_g4.py`, `logs/day_alloc_g4.log`)

G4x8 vs G2x8: full +0.175, body −0.133, trimmed +0.186, without February +0.002, 5/12 months, Feb +2.12 → **FAIL**. Not built.
atc_share averages 0.006 on the 8 airports' rows (ATC-attributed pre-departure delay is a tiny part of the total);
LTFM full +0.93. 2026 projection (information) +0.14.
