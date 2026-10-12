# Multi-scale / directional hour windows + ADS-B window in the context corrector (CTX_HOUR2) -- pre-registration

Written 2026-10-11 **before any gate number below was computed**. Base: v64 = 237.4538 (team v64 = our upload
`smart-jigsaw_v59`, §135).
**Disclosed:** one scratch screen already used the 2025 data (scratchpad `hour2_screen.py`, PROGRESS §140). On top of the
**pre-v64** corrector's own OOF + holdout output (`ctxc1sel_ro`), paired LightGBM residual models, leave-period-out over
the six 2025 periods, trimmed RMSE vs a control with (q, airport, hour, count):
- T1, the ±30 min means recomputed with the corrector output as the neighbours' p (≈ the information v64 added): −0.79;
  holdout period +0.09.
- T3 = T1 + ±15 / ±60 min and before-only / after-only windows: −1.30 (holdout −0.35), better in 6/6 periods.
- T4 = T3 + ADS-B window (appear/dwell tiers): −1.42 (body −0.92, holdout −0.58), better in 6/6 periods.
- EGLL got worse in every variant (+0.69 to +0.89); every other airport better or flat (EHAM).
A second scratch screen of the same design on top of `ctxc1sel_rohb` (v64's corrector) will run before the gate and be
reported in §140; it does not change anything below.

## Arms

Block 12 (`CTX_HOUR2=1`, `tests/context_corrector_test.py::hour2_block`), label-free, the frame's stack prediction as
the neighbours' p, same airport, other departures only:
`h2_rl15`, `h2_rl60`, `h2_ra15`, `h2_ra60` (mean of T − LOBT − p / T − AOBT_3 − p over ±15 / ±60 min);
`h2_rl_pre`, `h2_rl_post`, `h2_ra_pre`, `h2_ra_post` (over [T − 30 min, T] / [T, T + 30 min]); `h2_ab_app`, `h2_ab_dw`
(mean of clip(T − ADS-B pushback − p, ±600 s) over ±30 min, ADS-B v4 tiers appear / dwell separately), `h2_ab_n`.
Min 3 rows for the time windows, 1 for the ADS-B means; else null.

- **ctxc1r_roh2** = ctxc1r_roh + block 12 (`ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 PRC_ARMC_TAG=_roh2`).
- **ctxc1rb_rohb2** = ctxc1rb_rohb + block 12 (as above with `CTX_ADSB=1`, `PRC_ARMC_TAG=_rohb2`).
- **ctxc1sel_rohb2** = the `sel` rule on these two (r at the non-STABLE airports, rb at EHAM / LSZH / EDDF / LEBL).
Nothing else changes (rows, parameters, early stopping, ensemble2).

## Gate

Control = the production files `ctxc1sel_rohb` (holdout, OOF and ranking), **byte copies from the machine that built v64**
(not a refit here). Rows: the airports the build changes, i.e. all except LIRF (frozen), LSZH (v64's build takes it from
v62) and **EGLL (excluded up front, from the disclosed screen)**.
1. Holdout chain (`tests/v50_chain_gate.py`, `ARMS=all`, as §135): chain(ctxc1sel_rohb2) vs chain(ctxc1sel_rohb).
2. OOF months (corrector level): ctxc1sel_rohb2 vs ctxc1sel_rohb, labelled rows, y ≤ 5 h for trimmed.

**Pass iff:** (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul; (c) holdout chain full ≤ 0;
(d) holdout chain body ≤ +0.10; (e) OOF trimmed better in February and in June and in ≥ 7 of 10 OOF months.
Reported: per airport (EGLL, LSZH, LIRF for information), body by month, 2026 projection on the changed rows, 2025 → 2026
drift (SMD) of the eleven inputs.

## Build if it passes

As v64 (§135): production fits of both members and `sel` here; production chain for ctxc1sel_rohb2 and for the control
ctxc1sel_rohb (the control chain must reproduce the other machine's `chain_ctxc1sel_rohb_final`); identity-arm deltas on
both; delta = (new + id) − (control + id), set to 0 at LIRF, LSZH, EGLL and NM-line rows; p = v64's pre-day-stage file +
delta; then v63's recipe (`src/post/day_alloc_build.py`). **v65. Keep iff board < 237.4538.**
Blind board: the same on the four-month chain (`run_v64_final4m.sh` pattern), Feb/Jun from it, Jan/Jul = v65.

## Outcome (2026-10-11): WITHDRAWN before the gate; nothing fitted to completion, nothing built

The second scratch screen (on top of v64's corrector `ctxc1sel_rohb`, the teammate's files; same design; trimmed RMSE vs
control, pooled / holdout period): T1 (±30 min recomputed with q) **+0.57 / +0.33**; T3 (+ multi-scale, directional)
**+0.00 / +0.23**; T2 (ADS-B window only) −0.30 / −0.37, all six periods better, body −0.34; T4 (all) −0.27 / −0.08.
v64's block 11 already holds what the extra windows carried; only the ADS-B window adds, ≈ −0.3 s at screen level,
i.e. ≈ −0.1 s on the board at §135's screen → board ratio. Not worth the ≈ 10 h of corrector fits (team decision,
2026-10-11). The block-12 code stays in `tests/context_corrector_test.py` (off by default). The partial fits were stopped.
Note (2026-10-11): the "v65" named under "Build if it passes" was never built. The team's v65 is a different file (PROGRESS §142).
