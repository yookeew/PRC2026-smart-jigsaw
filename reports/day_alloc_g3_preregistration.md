# G3: day-shift slope model with per-flight inputs -- pre-registration

Written 2026-10-10 **before any number below was computed**. Base: v63 = 238.3173 (our upload `smart-jigsaw_v58`, §133):
G2 at the 8 airports. **Disclosed:** the same 2025 data were used for §130, §131 and §133. Two scratch checks ran before
this was written and are not candidates: a 2025 oracle (G2 205.86 on the 8 airports; perfect day mean 206.08, day × band
203.81, day × 3 h block 201.18) and block-averaged AOBT_3 gaps as an intra-day level signal (corr −0.14, dropped).

## Model

As G2 (`tests/day_alloc_gbm.py`: label (p − y)/λ̂, weight λ̂², same hyper-parameters, slope clipped to [0, 1], same λ̂,
periods, masks, leave-period-out rule), with these inputs added from the feature store (`cache/features/train2025.parquet`,
`ranking.parquet`, `final4m/ranking.parquet`; all label-free):
wake category, market segment (categorical); n_dep_30m_prev, dep_pressure, inbound_arr_delay, eobt_delay, sched_ground,
aobt3_taxi; deicing_risk, wx_snow, below_freezing.

One candidate only: **G3 at all 8 airports.**

## Decision (2025, 12 months, the 8 airports' rows, vs G2x8 = v63's recipe)

Pass iff: (1) pooled full ≤ −0.10 s; (2) pooled body ≤ 0; (3) pooled trimmed ≤ 0; (4) full better in ≥ 8/12 months;
(5) February and June each not worse on full. Fail → nothing is built. Reported: per airport, per month, 2026
projection onto 238.3173.

## Build if it passes

`src/post/day_alloc_build.py` with the G3 model; LSZH, LIRF as v63; LOBT window last. **Keep iff board < 238.3173.**
Then the four-month file the same way.

## Outcome (2026-10-10; `tests/day_alloc_g3.py`, `logs/day_alloc_g3.log`)

G3x8 vs G2x8: full −0.208, **body +0.027**, trimmed −0.223, 12/12 months (Feb −2.20, Jun −0.01) → **FAIL (2)**. Not built.
LTFM full −1.00 but body +0.31; without February the full gain is ≈ −0.04. The extra inputs mostly re-fit LTFM's
February 2025 snow days. 2026 projection (information) −0.16.
