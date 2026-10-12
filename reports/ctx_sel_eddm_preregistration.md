# Member `rb` at EDDM (and LEMD) in the corrector selection -- pre-registration

Written 2026-10-11 **before any holdout-chain number of these arms was computed**. Base: v66 = 236.8 (team build).

**Why.** `sel` (§86, v41) takes member `rb` (own-row ADS-B inputs, block 9) at EHAM / LSZH / EDDF / LEBL and member `r`
elsewhere. EDDM was left out because the July 2025 holdout had 31% ADS-B coverage against 84% in July 2026; but the OOF
days the corrector trains on have 71%, January 2025 86%, and 2026 85–88% (§144 table). v40's loss (rb at every airport)
was never attributed to one airport; EGLL (3% → 75%) is the clear coverage-shift case.

**Disclosed (2025 data already seen, corrector level, this session):** with v65's members refitted here (`ctxc1r_rh2`,
`ctxc1rb_rh2`, OOF + holdout), rb − r RMSE (y ≤ 5 h): EDDM −19.7 overall, better in 12/12 months (−9.6 … −72.7);
LEMD −6.1 overall, 10/12 months better (Jan +0.45, Feb +0.32); EGLL −2.7.

## Arms (no new fits: both members exist)

- **S1** `ctxc1sel_rh2e`: rb at EHAM / LSZH / EDDF / LEBL **+ EDDM**; r elsewhere.
- **S2** `ctxc1sel_rh2el`: S1 **+ LEMD**.
- Control: `ctxc1sel_rh2` (the same two member files, the current four airports).

## Stop rule first (label-free; the v67 lesson)

Block-9 inputs (`adsb_block`) at each added airport, 2025 rows on fetched ADS-B days vs 2026 rows: if any input has
|SMD| > 0.25 at EDDM, S1 and S2 stop; if only LEMD exceeds it, S2 stops.

## Gate (conditions of reports/ctx_hour_preregistration.md)

Holdout chain (`tests/v50_chain_gate.py`, `ARMS=all`) and OOF months (`tests/ctx_hour_gate.py`, `CTL=rh2`,
`CTL_DIR=cache/v66`), each arm vs the control. **Pass iff:** (a) trimmed ≤ −0.10; (b) Jan and Jul trimmed better; (c) full
≤ 0; (d) body ≤ +0.10; (e) OOF Feb and Jun better and ≥ 7/10 months. **Pick:** lower holdout-chain trimmed among passes.
The holdout's July EDDM coverage (31%) is below 2026's, so the holdout understates S1; that is accepted, not adjusted for.

## Build if it passes

Production sel from the existing production members (`ctxc1r_rh2.parquet`, `ctxc1rb_rh2.parquet`); production chain and
full build B(·) for the pick (the control's B(ctxc1sel_rh2) exists: `cache/v66/B_rh2.parquet`); **v68 = v66 +
(B(pick) − B(ctxc1sel_rh2))**, LOBT window, clip, round; LIRF / LSZH unchanged. Keep iff board < v66's score.

## Outcome (2026-10-11; `tests/ctx_sel_drift.py`, `logs/v66/ctx_sel_drift.log`) -- STOPPED at the drift rule, no gate run

- EDDM max |SMD| 0.52, LEMD 0.55 → both stop as written.
- What moved: the coverage flags only (EDDM `ab_matched` 0.8 → 1.0, `ab_cov` +0.37, `ab_tier` +0.26, `ab_n_pts` +0.32; LEMD
  likewise). The ADS-B timing inputs are stable (EDDM |SMD| ≤ 0.11: `ab_taxi_rel` −0.06, `ab_L` +0.11, gaps / speeds ≤ 0.05;
  LEMD ≤ 0.09).
- The rule ("any input") could not tell more tracked rows (in range) from changed tracks; an amendment for EDDM was
  offered and declined (team decision): the stop is honoured as written. Lesson for later rules: state the drift stop on
  the inputs that carry the measurement, and report coverage flags separately.
