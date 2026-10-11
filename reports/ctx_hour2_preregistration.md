# Runway / multi-scale window block in the context corrector (CTX_HOUR2) -- pre-registration

Written 2026-10-11 **before any gate number below was computed**. Base: v64 = 237.4538 (§135, board −0.86 vs v63).

## Why

After v64 the non-LIRF holdout chain (2025 Jan + Jul, `cache/v64/gate_rohb/chain_all`, trimmed y ≤ 5 h) still carries
shared within-day error. Leave-one-out group-mean oracle (shrunk n/(n+3)), Δ trimmed RMSE (v64 corrector / its control):

| group removed | control (`_ro`) | v64 (`_rohb`) |
|---|---|---|
| airport-day | −1.73 | −1.66 |
| airport-day × hour | −4.81 | −4.50 |
| airport-day × 30 min | −5.20 | −4.80 |
| airport-day × hour × runway | −5.99 | −5.78 |
| **airport-day × 30 min × runway** | **−6.89** | **−6.62** |
| airport-day × hour × stand prefix | −2.41 | −2.17 |
| airport-day × operator | +0.09 | +0.12 |

The runway is the strongest remaining dimension (LTFM −17.0 with runway vs −11.4 without, on its rows); v64's block 11
was airport-wide at ±30 min only.

**Disclosed screens on the 2025 data** (scratch, before this note): residual LightGBM on top of `ctxc1sel_rohb`'s own
OOF + holdout output (6 periods: holdout + the five OOF blocks, leave-period-out, 250 rounds, paired with vs without):
- block 12 below: trimmed **−1.02**, every period better (−0.80 to −1.25), body −1.10; LTFM −3.76, LFPG −1.13,
  LEMD −0.95, EDDF −0.82, EDDM −0.59, LIRF −0.60, LEBL −0.42, EGLL −0.27, LSZH −0.25, EHAM +0.01.
  (For scale: v64's block 11 screened −2.12 in the same kind of test, gate holdout chain −0.82, board −0.86.)
- a stand-prefix block (arrivals' taxi-in excess into the same stand prefix, departures from it, runway taxi-in excess)
  on top of block 12: +0.07, worse in 5 of 6 periods. **Dropped; not part of this test.**

## Arm

Block 12 (`CTX_HOUR2=1`, `tests/context_corrector_test.py` → `src/post/hour_feats.build_rwy`), 17 label-free inputs,
neighbours' p = the frame's stack prediction (as block 11):
`h_ra15`, `h_rl15`, `h_ra60`, `h_rl60` (block 11's T − AOBT_3 − p / T − LOBT − p means at ±15 and ±60 min),
`h_rac`, `h_rlc` (±30 min, each neighbour clipped to ±1,800 s), `bf_ra`, `af_ra` (clipped, before / after only),
per departure runway: `rw_ra30`, `rw_rl30`, `rw_rs30` (T − SOBT − p clipped ±3,600), `rw_ra15`, `rw_rl15`, `rw_bf_ra`,
`rw_af_ra`, `rw_p30` (mean p), `rw_n30` (count).

Both members refitted with blocks 11 + 12 (`PRC_ARMC_TAG=_rh2`):
- `ctxc1r_rh2` = `ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1`
- `ctxc1rb_rh2` = the same + `CTX_ADSB=1`
- **`ctxc1sel_rh2`** = the `sel` rule (rb at EHAM / LSZH / EDDF / LEBL, r elsewhere).
Nothing else changes (rows, parameters, early stopping, ensemble2).

## Gate

Control = v64's production corrector `ctxc1sel_rohb` (holdout chain `cache/v64/gate_rohb/chain_all.parquet`, OOF
`cache/oof/ctxc1sel_rohb`), same machine, same code path.
1. **Holdout chain** (`tests/v50_chain_gate.py`, `ARMS=all`, `RO_FILE=reocc_feats.parquet`, `CORR=ctxc1sel_rh2`).
2. **OOF months** (corrector level), labelled rows, y ≤ 5 h for trimmed.

**Pass iff** (as §135): (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul;
(c) holdout chain full ≤ 0; (d) holdout chain body ≤ +0.10; (e) OOF trimmed better in February and in June and in
≥ 7 of 10 OOF months; and (f) no block-12 input with |SMD| > 0.25 between the 2025 training rows and the 2026 ranking
rows (pooled; v64's block 11 had ≤ 0.09). If (a)–(e) pass and only (f) fails, nothing is built until the team decides.

## Build if it passes

As v64: production fits (`ctx_armc.py prod` for both members, then `sel`); production chain for `ctxc1sel_rh2`
(`cache/v65/run_chain.sh`) with identity-arm deltas; delta = (chain_rh2 + id deltas) − (chain_ctxc1sel_ro + id deltas)
(`src/post/v64_delta.py`, the control chain that v64 was built against); p = v54 + delta (`FREEZE_LIRF=1
v50_assemble.py`); then v63's recipe (`day_alloc_build.py`: OPDI deltas, G2 at 8 airports, LSZH / LIRF rows from v62).
**v65. Keep iff board < 237.4538.** Blind board: the same on the four-month chain (`run_v65_final4m.sh`) once the
four-month files are on this machine (or run on the other machine).

## Outcome (2026-10-11; `logs/v65/ctx_hour2_gate_early.log`, `logs/v65/ctx_hour2_drift_early.log`)

The holdout `sel`, chain and gate were run once while the production fits ran (`*_early.log`); the pipeline repeats them
(`logs/v65/ctx_hour2_gate.log`).

| holdout chain vs `ctxc1sel_rohb` | trimmed | full | body | trimmed Jan / Jul | body Jan / Jul | OOF trimmed (10/10 months better) | pass |
|---|---|---|---|---|---|---|---|
| **ctxc1sel_rh2** | **−0.718** | −0.610 | −1.050 | −0.048 / −1.123 | −0.308 / −1.573 | −1.29 (Feb −1.13, Jun −1.33) | **yes** |

- Per airport trimmed: LTFM −5.06, LIRF −0.72, EDDF −0.71, LEMD −0.56, LEBL −0.11, EDDM +0.04, EGLL +0.24, LFPG +0.43,
  EHAM +0.53, LSZH +1.15 (LSZH and LIRF rows are not changed by the build). Body: LTFM −4.65, LIRF −1.43, EDDF −0.77; no
  airport above +0.06.
- Corrector level, member r only, vs `ctxc1r_roh`: OOF −1.33 (10/10 months), holdout −1.12 (Jan −0.58, Jul −1.46).
- 2026 projection: −0.79 (all airports), −0.65 (LIRF frozen), **−0.71 on the rows production changes** (no LIRF, no LSZH).
- (f) drift: max pooled |SMD| 0.204 (`rw_rs30`: more departure delay in 2026), PASS. Per airport the AOBT_3-based inputs
  shift most at LFPG (+0.43 to +0.59 sd); block 11's `h_ra` carried the same LFPG shift in v64.
**PASS → v65 built.**
