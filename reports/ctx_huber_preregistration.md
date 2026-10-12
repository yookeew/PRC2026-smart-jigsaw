# Huber-loss corrector on NM-matched rows (CTX_HUBER) -- pre-registration

Written 2026-10-11 **before any holdout-chain (Jan / Jul 2025) or OOF-gate number of this arm was computed**. Base: v66 =
236.8 (team build).

**Why.** The corrector (ensemble2 LightGBM on y − stack) is fitted with L2. Its training residuals are heavy-tailed (y up
to 5 h on NM-matched rows), so a few long taxis set many leaf values. A Huber loss with α = 1,000 s caps the gradient of
residuals beyond ~17 min, so ordinary rows keep L2 behaviour. On NM-unmatched rows the tail *is* the signal (LTFM snow-day
rows, EHAM storm rows), and Huber is worse there, so the hybrid keeps L2 for them.

**Disclosed (2025 data already seen; session scratchpad `scr_run.py`, `hyb.py`).** Member `r`, ensemble2, fitted on
8 OOF months, scored on two held-out month pairs, L2 vs Huber α = 1,000 (the only α tried at corrector level):

| inputs | held out | plain Huber trimmed | **hybrid** trimmed | hybrid non-LIRF trimmed | hybrid body | hybrid full | hybrid per month (non-LIRF trimmed) |
|---|---|---|---|---|---|---|---|
| rh2 (v65's blocks 11 + 12) | Feb + Aug | +1.23 | **−0.41** | −0.36 | −1.35 | −0.40 | Feb −0.10, Aug −0.61 |
| rh2 | May + Nov | −0.49 | **−0.70** | −0.35 | −1.13 | −0.52 | May −0.39, Nov −0.30 |
| roh (v64's inputs, older) | Feb + Aug | +1.61 | **+0.10** | +0.25 | −1.37 | +0.09 | Feb +0.84 (LTFM +1.84), Aug −0.29 |

- Plain Huber loses on NM-unmatched rows (Feb + Aug non-LIRF NM-unmatched +36 s RMSE), hence the hybrid.
- A 50/50 blend and an |L2 correction| threshold (150–600 s) were also looked at on the rh2 screens; neither is clearly
  better, and the plain pick (w = 1, no threshold) is fixed here. Earlier today (§144, roh inputs, Feb + Aug) a variant
  gated on small |L2 correction| (all rows) gave non-LIRF trimmed −0.19, body −0.37.
- The roh row is the counter-evidence: February LTFM (snow days) can flip the sign. Body improves in all three (partly a
  truncation effect: body selects y ≤ 1 h and Huber predicts slightly lower), so the gate leans on trimmed and full.

## Arm

- `CTX_HUBER=1000` in `tests/context_corrector_test.py`: `objective="huber", alpha=1000` for the global and the
  per-airport models (early stopping on the Huber metric, as LightGBM's default; 3,000-round cap unchanged). Nothing else
  changes (rows, inputs, folds, ensemble2).
- Members `ctxc1r_rh2hu`, `ctxc1rb_rh2hu` (`ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1 CTX_HUBER=1000
  PRC_ARMC_TAG=_rh2hu`, rb adds `CTX_ADSB=1`), `sel` → `ctxc1sel_rh2hu`.
- **Treatment** `ctxc1sel_rh2hy` = `ctxc1sel_rh2hu` on NM-matched rows (`FLIGHT_ID_mvt` present), `ctxc1sel_rh2` on
  NM-unmatched rows (`src/post/ctx_huber_hybrid.py`), on holdout, OOF folds and ranking rows.
- **Control** `ctxc1sel_rh2` (refitted on this machine for the v67 gate; its holdout chain is `cache/v66/gate_rh2`). The
  Huber members are separate processes from the control, as any new fit.

## Gate (conditions of reports/ctx_hour_preregistration.md, as v64 / v65 / v67)

1. Holdout chain (`tests/v50_chain_gate.py`, `ARMS=all`, `RO_FILE=reocc_feats.parquet`) of the treatment vs
   `cache/v66/gate_rh2`, LIRF NM-unmatched rows as v41 in both.
2. OOF months, corrector level (`tests/ctx_hour_gate.py`, `CTL=rh2`, `CTL_DIR=cache/v66`, `TAG=rh2hy`).

**Pass iff:** (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul; (c) full ≤ 0;
(d) body ≤ +0.10; (e) OOF trimmed better in February and in June and in ≥ 7 of 10 OOF months.

**Output-spread stop (label-free; the v67 lesson, a condition, not a report).** `tests/ctx_huber_drift.py`: RMS of
(treatment − control) on NM-matched rows, per month pooled over airports, 2026 production files vs the 2025 holdout files.
**Stop if the January 2026 or the July 2026 value exceeds 1.25 × the same month's 2025 holdout value** (v67's MLP: 86 vs
66 s in January, ratio 1.30). Per-airport values and p99 are reported, not conditions.

## Build if it passes both

Production members (`ctx_armc.py prod`, tag `_rh2hu`), `sel`, hybrid with the existing production `ctxc1sel_rh2`;
production chain (`run_v66_chain.sh` steps) for `ctxc1sel_rh2hy`; full build B(rh2hy) as B(rh2) was built for v67
(`v64_delta.py`, `FREEZE_LIRF=1 v50_assemble.py`, `day_alloc_build.py`, LSZH / LIRF rows from v65).
**v68 = v66 + (B(rh2hy) − B(rh2))**, LOBT window, clip at 0, round; LIRF and LSZH rows unchanged. B(rh2) is
`cache/v66/B_rh2.parquet` (built for v67, same machine). **Keep iff board < v66's score (236.8…).**

## Outcome (2026-10-11; `logs/v68/ctx_huber_gate.log`, `logs/v68/gate_rh2hy.log`) -- FAIL, not built

Members `ctxc1r_rh2hu` / `ctxc1rb_rh2hu` (best iterations 2,010–3,000; the 3,000 cap bound in three of twelve fits).
Holdout chain vs `cache/v66/gate_rh2` (change RMS 27.0 s, mean −0.40):

| | trimmed | full | body | trimmed Jan / Jul | body Jan / Jul | non-LIRF trimmed / body |
|---|---|---|---|---|---|---|
| hybrid − control | **+0.236** | **+0.161** | −1.157 | +0.045 / **+0.351** | −0.235 / −1.806 | −0.136 / −0.383 |

- Per airport trimmed: LIRF **+1.91** (body −6.64), EGLL +0.54, LFPG +0.43, EDDM +0.13; LSZH −1.36, EHAM −0.62, LTFM −0.45,
  LEBL −0.43, LEMD −0.26, EDDF −0.22. 2026 projection on production rows (no LIRF, no LSZH) −0.07 s.
- OOF months, corrector level: trimmed −0.415, 9 / 10 months better (Feb −0.14, Jun −0.16; Aug +0.05).
- Conditions: (a) no, (b) no (July), (c) no, (d) yes, (e) yes. **FAIL.** The production fits were stopped; no drift check, no
  build. The conditional LIRF unfreeze (reports/lirf_matched_unfreeze_rh2hy_preregistration.md) is not run.
- Reading: Huber lowers the long-tail predictions. That is good for body (selected on y ≤ 1 h) and bad for trimmed where
  the tail is real, worst at LIRF (echo hedges) and in July. The corrector-level screens (member r only, no post-stages)
  did not show the LIRF tail cost.
