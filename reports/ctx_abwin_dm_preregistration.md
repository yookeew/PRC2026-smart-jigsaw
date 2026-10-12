# Demeaned ADS-B neighbour window (block 13d, CTX_ABWIN_DM) -- pre-registration

Written 2026-10-11 **before any holdout-chain or OOF-gate number of this arm was computed**. Base: v66 = 236.8.

**Why.** Block 13 (reports/ctx_abwin_preregistration.md) stopped at its drift check on two things: the neighbour count
moves with ADS-B coverage, and the tier means move in level from 2025 to 2026 (LSZH dwell −0.70, EHAM appear +0.36). Both
are removed by construction here: no count, and each tier mean minus its airport-day mean. What is left is the within-day
variation of how long the tracked neighbours actually taxi relative to their predictions. That is the hour-level error the
day stage cannot see (§134) and that blocks 11 / 12 partly took (v64 transferred at twice its projection).

**Disclosed (2025 data already seen; session scratchpad `scr_run.py` variant `t2d`).** Member `r` (rh2 inputs), ensemble2,
8 OOF months, demeaned window at EDDF / EDDM / EHAM / LEBL / LSZH vs without:

| held out | non-LIRF trimmed | non-LIRF body | per month | per airport trimmed (in scope) |
|---|---|---|---|---|
| Feb + Aug | −0.48 | −0.44 | Feb −0.43, Aug −0.53 | EDDM −2.16, LEBL −1.39, LSZH −0.60, EHAM −0.58, EDDF −0.34 |
| May + Nov | −0.19 | −0.16 | May −0.16, Nov −0.23 | LSZH −1.00, LEBL −0.45, EDDM −0.29, EDDF −0.28, EHAM −0.16 |

The non-demeaned window with the count (§144) was −0.96 / −0.38 on the same splits.

**Scope, fixed label-free** (`tests/ctx_abwin_dm_drift.py`, `logs/v68/ctx_abwin_dm_drift.log`, run before this report):
SD ratio 2026 / 2025 of the two demeaned inputs and 2026 non-null share:

| airport | SD ratio appear / dwell | non-null 2026 | in scope |
|---|---|---|---|
| EDDM | 1.07 / 0.93 | 0.99 / 1.00 | **yes** |
| EHAM | 1.12 / 1.07 | 0.99 / 1.00 | **yes** |
| LEBL | 0.91 / 0.94 | 0.95 / 0.94 | **yes** |
| EDDF | 1.45 / 1.46 | 0.60 / 0.42 | no (coverage collapse: means from 1–2 neighbours) |
| LSZH | 0.92 / 1.31 | 0.96 / 0.93 | no (dwell spread; LSZH rows are frozen in the build anyway) |

Means are 0 by construction in both years. The spread stop (SD ratio outside [0.8, 1.25] at an in-scope airport) is
therefore already measured and passes at the three airports.

## Arm

`CTX_ABWIN_DM=1` (`abwin_dm_block` in `tests/context_corrector_test.py`): block 13's appear and dwell means at **EDDM, EHAM,
LEBL** minus their airport-day mean (day of T), named `h3d_ab_app`, `h3d_ab_dw`; null elsewhere; no count. Members refitted
with everything else unchanged. Two branches, fixed now:
- **If the Huber gate passes** (reports/ctx_huber_preregistration.md): members `ctxc1r_rh2hud`, `ctxc1rb_rh2hud`
  (`CTX_HUBER=1000 CTX_ABWIN_DM=1`, tag `_rh2hud`), `sel` → `ctxc1sel_rh2hud`; treatment `ctxc1sel_rh2hdy` = hybrid of
  `ctxc1sel_rh2` (NM-unmatched rows) and `ctxc1sel_rh2hud` (NM-matched rows). Control: `ctxc1sel_rh2hy`.
- **Otherwise:** members `ctxc1r_rh2d`, `ctxc1rb_rh2d` (L2, `CTX_ABWIN_DM=1`, tag `_rh2d`), `sel` → treatment
  `ctxc1sel_rh2d`. Control: `ctxc1sel_rh2`.

## Gate (conditions of reports/ctx_hour_preregistration.md)

Holdout chain (`tests/v50_chain_gate.py`, `ARMS=all`) and OOF months (`tests/ctx_hour_gate.py`) vs the branch's control.
**Pass iff:** (a) trimmed ≤ −0.10; (b) Jan and Jul trimmed better; (c) full ≤ 0; (d) body ≤ +0.10; (e) OOF Feb and Jun
better and ≥ 7 / 10 months. Reported: per airport, and the trimmed change on EDDM + EHAM + LEBL rows alone.

## Build if it passes

As the branch's control was built (production members, sel, hybrid if any, production chain, B(·)), then
**v = current best + (B(treatment) − B(control))**, LOBT window, clip, round; LIRF and LSZH rows unchanged. Keep iff board
< the best at that time.

## Outcome (2026-10-12; L2 branch, `logs/v68/ctx_abwin_dm_gate.log`, `logs/v68/gate_rh2d.log`) -- FAIL, not built

The Huber gate failed, so the L2 branch ran: `ctxc1sel_rh2d` vs `ctxc1sel_rh2`. Holdout chain (change RMS 21.4 s, mean −0.10):
trimmed −0.033, full −0.055, body −0.072; trimmed Jan **+0.229** / Jul −0.186; non-LIRF trimmed −0.040. Per airport trimmed:
LEBL −0.83, EDDM −0.21, EHAM **+0.60** (in scope); out of scope LFPG +0.61, LSZH +0.46, EGLL −0.58 (chain interactions and
refit noise). OOF months: trimmed −0.080, June **+0.30**, 6 / 10 months better. Conditions (a), (b), (e) fail. **FAIL.** The
three-airport share of the screen gain (−0.31 / −0.08) was already near the refit-noise floor; Jan / Jul show nothing.
