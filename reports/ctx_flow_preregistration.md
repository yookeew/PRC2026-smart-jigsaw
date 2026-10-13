# Flow-state block (block 13) in the context corrector (CTX_FLOW) -- pre-registration

Written 2026-10-11 **before any gate number below was computed**. Base: v65 = 237.0694 (§136, board −0.38 vs v64).
Focus (user, 2026-10-11): the Jan/Jul holdout and the live board; a general model improvement, no board fitting.

## Why

PROGRESS §137: after blocks 11 and 12 the within-day window direction is used up (static identity cells, operator
windows and destination-shared residuals all screen dead). The one fresh label-free signal is the **flow state** of a
departure: whether other flights to the same destination / region are running late at the same time (an ATFM-regulated
flow, so a late flight most likely waited at the gate) and where the flight sits in its airport's schedule order. It
helps the corrector split a departure's own delay T − SOBT into gate wait vs taxi, which the window blocks don't see.

**Disclosed screens on the 2025 data** (§137, `tests/flow_screen/`, before this note): residual LightGBM on top of
`ctxc1sel_rh2`'s own OOF + holdout output (7 periods: Jan | Jul | the five OOF blocks, leave-period-out, 300 rounds,
paired with vs without; control inputs p, airport, T − SOBT, T − LOBT, T − AOBT_3, hour, NM flag):
- the block below: trimmed **−0.600**, body −0.207, every period better (Jan −0.33, Jul −0.69, Feb–Mar −1.70,
  Apr–May −0.37, Jun+Aug −0.46, Sep–Oct −0.43, Nov–Dec −0.20), every airport better (LTFM −1.79, LIRF −1.44,
  LFPG −0.41, LEMD −0.19, EGLL −0.15, EDDF / LEBL −0.12, EHAM −0.09, LSZH −0.08, EDDM −0.07);
- its parts alone: destination flow −0.371, region flow −0.249, schedule displacement −0.215;
- operator ±30 min windows −0.069 (two periods worse). **Dropped; not part of this test.**
- For scale: block 12 screened −1.02 in the same kind of test, then gate −0.72, board −0.38.

## Arm

Block 13 (`CTX_FLOW=1`, `tests/context_corrector_test.py` → `src/post/flow_feats.build`), 21 label-free inputs from
the movement file of the frame (2025 training frame or the ranking frame; all ten origins are in each):
- **destination flow, other origins** (`xo_*`): departures from the other nine airports to the same `ADES_mvt` with
  |T_j − T| ≤ 60 min: count `xo_n`, median T − SOBT `xo_moff`, median T − LOBT `xo_mdl`, share with T − SOBT > 30 min `xo_f30`;
- **destination flow, same origin** (`so_*`): the same four over our own airport's other departures to that ADES within ±90 min;
- **region flow** (`rso_*`, `rxo_*`): departures to the same two-letter ICAO prefix of ADES but a different ADES, ±60 min,
  same origin and other origins, the same four statistics each;
- **schedule-order displacement** (`sq_*`): our airport's departures with |SOBT_j − SOBT| ≤ 20 min: count `sq_n`,
  overtakers `sq_over` (scheduled later, took off earlier), overtaken `sq_under` (scheduled earlier, took off later),
  median T_j − T `sq_mdT`, `sq_net` = (over − under) / n.
Rows without neighbours get null; null ADES never joins.

Both members refitted with blocks 11 + 12 + 13 (`PRC_ARMC_TAG=_rf`):
- `ctxc1r_rf` = `ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_FLOW=1`
- `ctxc1rb_rf` = the same + `CTX_ADSB=1`
- **`ctxc1sel_rf`** = the `sel` rule (rb at EHAM / LSZH / EDDF / LEBL, r elsewhere).
Nothing else changes (rows, parameters, early stopping, ensemble2).

## Gate

Control = v65's production corrector `ctxc1sel_rh2` (holdout chain `cache/v65/gate_rh2/chain_all.parquet`, OOF
`cache/oof/ctxc1sel_rh2`), same machine, same code path.
1. **Holdout chain** (`tests/v50_chain_gate.py`, `ARMS=all`, `RO_FILE=reocc_feats.parquet`, `CORR=ctxc1sel_rf`,
   `SCR=cache/v66/gate_rf`).
2. **OOF months** (corrector level), labelled rows, y ≤ 5 h for trimmed.
Run: `CTL=rh2 CTL_DIR=cache/v65 TAG=rf GATE_DIR=cache/v66 BOARD=237.0694 tests/ctx_hour_gate.py`.

**Pass iff** (as §135 / §136): (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul;
(c) holdout chain full ≤ 0; (d) holdout chain body ≤ +0.10; (e) OOF trimmed better in February and in June and in
≥ 7 of 10 OOF months; and (f) no block-13 input with pooled |SMD| > 0.25 between the 2025 training rows and the 2026
ranking rows (`tests/ctx_flow_drift.py`). If (a)–(e) pass and only (f) fails, nothing is built until the team decides.

## Build if it passes

As v65: production fits (`ctx_armc.py prod` for both members, then `sel`); production chain for `ctxc1sel_rf`
(`scripts/run_v66_chain.sh`) with identity-arm deltas; delta = (chain_rf + id deltas) − (chain_ctxc1sel_ro + id deltas)
(`src/post/v64_delta.py`, the control chain v64 and v65 were built against); p = v54 + delta (`FREEZE_LIRF=1
v50_assemble.py`); then v63's recipe (`day_alloc_build.py`: OPDI deltas, G2 at 8 airports, LSZH / LIRF rows from v62).
**v66. Keep iff board < 237.0694.** The four-month version is built on the last day with the rest (user's plan).

## Amendments

**A1 (2026-10-11, before any holdout or OOF gate number; after condition (f)'s drift script had run once).** `sq_net`
was computed as (`sq_over` − `sq_under`) on unsigned counts, so every negative value wrapped to ≈ 4.3e9 / n (the drift
log showed a mean of 9.8e7). The §137 screen had the same bug. Fixed to signed arithmetic, as the definition above
states; the run was stopped during member r's first fit and restarted from the start, drift check included. Nothing
else changes.

## Outcome (2026-10-11; `logs/v66/ctx_flow_gate.log`, `logs/v66/ctx_flow_drift.log`, `logs/v66/run_v66.log`)

| holdout chain vs `ctxc1sel_rh2` | trimmed | full | body | trimmed Jan / Jul | body Jan / Jul | OOF trimmed (10/10 months better) | pass |
|---|---|---|---|---|---|---|---|
| **ctxc1sel_rf** | **−0.351** | −0.261 | −0.278 | −0.731 / −0.137 | −0.510 / −0.117 | −1.43 (Feb −0.46, Jun −2.13) | **yes** |

- Non-LIRF: trimmed −0.707, full −0.499, body −0.437. Per airport trimmed: LTFM −1.36, LFPG −1.00, LSZH −0.94, EDDF −0.82,
  EHAM −0.75, EDDM −0.60, EGLL −0.36, LEMD −0.24, LEBL 0.00, LIRF +0.98 (LIRF and LSZH rows are not changed by the build).
- Member r alone, corrector level, 12 months vs `ctxc1r_rh2`: −1.69, 12/12 months; non-LIRF only −0.21 (11/12). Most of the
  corrector-level LIRF gain is on NM-unmatched rows, where production's v48 hedges are far better anyway (holdout RMSE 3,233
  vs rf 3,933 / rh2 4,088), so nothing there is used.
- (f) drift: max pooled |SMD| 0.158 (`rxo_f30`), PASS.
- 2026 projection on the rows production changes: **−0.47 s** (LIRF frozen −0.51).
**PASS → v66 built.** `data/submissions/smart-jigsaw_v66.parquet`, md5 27cdb451694f6bbd1a384dbbea19f367. vs v65: 284,290 rows
changed, RMS 24.9 s, mean +0.09, max |Δ| 683; LIRF and LSZH identical; per airport RMS 19.7 (LEBL) to 35.1 (LTFM).
**Keep iff board < 237.0694.**
