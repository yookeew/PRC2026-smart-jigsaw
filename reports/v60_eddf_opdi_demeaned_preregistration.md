# v60: EDDF OPDI delta, demeaned within each airport-day (offset-immune) -- pre-registration

Written 2026-10-10 **before the demeaned numbers were computed**. Follows §129 (EDDF OPDI stopped: the 2026
exit-parking clock sits ≈ +58 s from late 2025; the raw treatment was worth body −5.11 on 2025 EDDF rows) and the
team's suggestion (cf. §72's day-demeaning).

## Idea

A clock offset common to a day's OPDI-covered flights moves their treatment − control delta by a roughly common
amount. Removing the delta's mean over the covered rows of each EDDF airport-day removes that, and with it any real
day-level information OPDI carried. The daily reconciliation, re-solved afterwards, re-pins the day level from the
published sums. What remains is within-day information (which flight pushed earlier or later than its neighbours),
which a constant clock offset does not touch.

## Definition

Covered = OPDI ground event ≥ 60 s before T. Day = UTC date of T. Only EDDF airport-days with covered share ≥ 0.2
(drops 2026-01-02..10). On those days: δ_dm = (trt − ctl) − mean over the day's covered rows of (trt − ctl), applied
to covered rows only; uncovered rows get 0 (their delta was ≈ 0 in 2025: −0.01 body).

## Evidence that decides (2025, labels): the leave-one-month-out run of §129

Population P as §129's gate (EDDF, OOF ADS-B days 2025-09-16..12-31, day share ≥ 0.2; `cache/v57/lomo_2025-{09..12}.parquet`).
Arms: ctl and ctl + δ_dm. **PASS iff** (same bars as §129's gate G):
1. body RMSE change on P ≤ −2.0 s;
2. < 0 in ≥ 3 of the 4 months;
3. day-cluster bootstrap 95% CI upper bound < 0;
4. full RMSE change on P ≤ 0.
Reported: covered / uncovered, the raw (not demeaned) arm for comparison, and the 2026 projection
(P's ΔMSE per covered row × 2026 covered EDDF rows on kept days, onto v55's 239.40). Label-free 2026 check, reported:
mean raw delta on covered rows per day in 2026 vs 2025 P (the part the demeaning removes; the clock offset shows up there).

## Build if PASS

δ_dm from v52's production deltas (`cache/v52/comb_delta_opdi.parquet` + `nb_delta_opdi.parquet`, EDDF rows), as
defined above, NM-line rows zero. v60 = v55 with EDDF rows replaced by: v54 + δ_dm, then v55's day reconciliation
(R0: point solve, 0.75) re-solved on that, LOBT window last. Every other row = v55. **Keep iff board < 239.40.**
Rows changed: EDDF only (disjoint from v58's LSZH rows).

## Outcome (2026-10-10; `tests/v60_eddf_demeaned_eval.py`, `logs/v60_eddf_demeaned_eval.log`)

P: 19,332 EDDF rows, 31 days, covered 0.472. Demeaned vs ctl: **body −4.99** (covered −11.33, uncovered 0.00), full −4.84;
months −4.94 / −3.82 / −6.93 / −4.05; day-cluster CI [−6.42, −3.73]. **PASS (4/4).** Raw arm for comparison: body −5.11,
full −4.96, so demeaning costs ≈ 0.1 s of EDDF body. Daily mean raw delta on covered rows: 2025 P median +2.8 s
(IQR +1.2 to +4.6); 2026 (16,937 covered rows on 51 kept days) median +2.1 s (IQR −1.9 to +6.6): no level anomaly in the
treatment in 2026. Projection −0.25 s on the board.
