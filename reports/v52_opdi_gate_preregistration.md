# v52: OPDI ground events in the post-stages -- gate pre-registration

Written 2026-10-08 **before any gate number was computed**. Follows `reports/opdi_scoping.md` (S1 coverage PASS;
S2 screen body −1.53, Jan −1.18, Jul −1.79; condition 3 failed as literally written, see PROGRESS §126).

## Inputs

`src/features/opdi_feats.py` → `cache/opdi_feats/opdi_feats.parquet`: one row per departure (2025 training files
and the 2026 ranking file, identical code), label-free. OPDI flight matched by callsign (CALLSIGN_flt, else
FLIGHT_mvt) + ADEP + nearest exit-runway event within 300 s of the takeoff time T; events of that flight at ADEP in
[T − 3 h, T + 60 s].

Features O (all seconds before T, null when absent): `o_matched`, `o_lead_first` (earliest ground event),
`o_lead_exitpark` (last exit-parking_position), `o_lead_entrypark` (first entry-parking_position),
`o_lead_taxi` (first entry-taxiway), `o_lead_apron` (first apron event), `o_lead_rwy` (last entry-runway),
`o_deice` (entry-deicing_pad seen), `o_n_ground`; plus, computed in the stage from its base s:
`o_rel_first`, `o_rel_exitpark`, `o_rel_taxi` (= lead − s).

## Arms (one process, `tests/v52_opdi_gate.py`, PRC_LGB=ctxcorr2r, ARMC_BASE=ctxc1sel)

- **ctl** = v50's chain (arm `all` of `tests/v50_chain_gate.py`: POS + RO in the combiner, RO in neighbour stage b).
  Must reproduce `cache/v50/chain_all.parquet` (asserted, max |diff| < 1e-6).
- **opdi** = ctl + O in the combiner and in neighbour stage b.

## Airport scope (label-free, fixed now)

An airport keeps its OPDI delta only if
(a) its holdout (Jan + Jul 2025) share of departures with an OPDI ground event ≥ 60 s before T is ≥ 0.20, **and**
(b) its 2026 ranking share is within [0.5, 2] × the holdout share, **and**
(c) the 2025-holdout vs 2026 standardised mean difference of `o_lead_first` on covered rows is ≤ 0.25.
Elsewhere the delta is zero (the chain keeps v50). LIRF keeps v50 in any case (team rule; no OPDI coverage).
From S1 this is expected to keep LSZH only; EDDF fails (a) (2025 Jan/Jul coverage ≈ 0, 2026 ≈ 0.55) and is left
for a separate test.

## Decision (primary = scoped arm vs ctl on v50's holdout chain)

**PASS iff** all of:
1. body (NM-matched, y ≤ 1 h) pooled ≤ −0.5 s;
2. body Jan < 0 and body Jul < 0;
3. day-cluster bootstrap 95% CI of the body delta has upper bound < 0;
4. full RMSE delta ≤ 0.

Reported: trimmed (y ≤ 5 h), full, per airport, the unscoped arm, the 2025-vs-2026 input check table.

## Build if PASS

v52 = v50 + (treatment − control) of the production combiner (`src/post/v52_comb_delta.py`) and neighbour stage b
(`src/post/v52_nb_delta.py`), both fit in one process with PRC_LGB=ctxc1sel, control = v50's production inputs
(POS + RO / RO), treatment = + O; deltas zero outside the scoped airports, on LIRF, and on NM-line rows; LOBT window
last (`src/post/v50_assemble.py`). **Keep iff board < 243.4565 (v50).**
