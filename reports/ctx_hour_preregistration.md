# Hour-window block in the context corrector (CTX_HOUR) -- pre-registration

Written 2026-10-11 **before any number below was computed**. Base: v63 = 238.3173 (§133/§134).
**Disclosed:** §134's scratch screens already used the 2025 data: hour-window inputs on top of the corrector's OOF output
(10 OOF months, leave-period-out, paired) gave trimmed −2.12, body −0.36; on the Jan ↔ Jul holdout chain −0.57 trimmed.
This test puts the block inside the corrector, where it can interact with the other inputs.

## Arm

Member `r` of the stand-aware corrector (`ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1`) plus **block 11** (`CTX_HOUR=1`,
`tests/context_corrector_test.py` → `src/post/hour_feats.py`): `h_rl`, `h_ra`, `h_rs`, `h_p`, `h_n`, `rwy30`, `rwy15`,
`bk_lobt`, `bk_sobt`, `h_tinx`, with the frame's stack prediction as the neighbours' p. Name **ctxc1r_roh**. Nothing
else changes (rows, parameters, early stopping, ensemble2). Only `r` is refitted (compute): **ctxc1sel_roh** = ctxc1r_roh at
the non-STABLE airports + the **exact** production `ctxc1rb_ro` at EHAM / LSZH / EDDF / LEBL (the `sel` rule).

## Gate

Control = the production files `ctxc1sel_ro` (holdout and OOF, byte copies from the machine that built v54).
1. **Holdout chain** (`tests/v50_chain_gate.py`, `ARMS=all`, one process per corrector, same machine, `RO_FILE=reocc_feats.parquet`):
   chain(ctxc1sel_roh) vs chain(ctxc1sel_ro), LIRF NM-unmatched rows as v41 in both.
2. **OOF months** (corrector level): ctxc1sel_roh vs ctxc1sel_ro OOF folds, labelled rows, y ≤ 5 h for trimmed.

**Pass iff:** (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul; (c) holdout chain full ≤ 0;
(d) holdout chain body ≤ +0.10; (e) OOF trimmed better in February and in June (the blind board's extra months) and in
≥ 7 of 10 OOF months. Reported: per airport, body by month, 2026 projection (per-airport ΔMSE × 2026 counts).

## Build if it passes

Production fit of ctxc1r_roh (`ctx_armc.py prod`), sel with the exact `ctxc1rb_ro`; production chain for ctxc1sel_roh and
for ctxc1sel_ro (the control chain must reproduce `chain_ctxc1sel_ro_final` from the other machine); identity-arm deltas
on both (`reocc_feats6`); delta = (new chain + id deltas) − (control chain + id deltas); p = v54 + delta (LIRF and NM-line
rows frozen, `assemble.py`); then v63's recipe on p (`src/post/day_alloc_build.py`: OPDI deltas, G2 at 8 airports, LSZH /
LIRF from v62). **v64. Keep iff board < 238.3173.** Blind board: the same on the four-month chain once its files exist.

## Amendment A1 (2026-10-11, written before any gate number; spare compute)

Second candidate **ctxc1sel_rohb** = ctxc1r_roh at the non-STABLE airports + **ctxc1rb_rohb** (member `rb`,
`CTX_ADSB=1`, with block 11) at EHAM / LSZH / EDDF / LEBL. Same gate and pass rule, against the same control.
**Pick:** if both pass, the one with the lower holdout-chain trimmed RMSE; if only one passes, that one.

## Outcome (2026-10-11; `logs/v64/ctx_hour_gate_roh.log`, `logs/v64/ctx_hour_gate_rohb.log`)

Control reproduces: holdout chain on `ctxc1sel_ro` trimmed 241.84 (§124's −1.29 vs ctxc1sel), production chain byte-identical
to the other machine's `chain_ctxc1sel_ro_final` (md5 98099262…), identity deltas 23.74 / 24.34 s RMS (v54's numbers).

| holdout chain vs control | trimmed | full | body | trimmed Jan / Jul | body Jan / Jul | OOF trimmed (10/10 months better) | pass |
|---|---|---|---|---|---|---|---|
| ctxc1sel_roh (candidate 1) | −0.655 | −0.423 | −0.683 | −0.011 / −1.041 | +0.053 / −1.195 | −2.51 (Feb −11.46, Jun −0.95) | yes |
| **ctxc1sel_rohb (A1)** | **−0.819** | **−0.578** | **−0.824** | −0.114 / −1.244 | +0.056 / −1.438 | −2.72 (Feb −11.54, Jun −1.05) | **yes** |

**Both pass; A1's pick rule takes ctxc1sel_rohb** (lower holdout-chain trimmed). Per airport trimmed (rohb): EHAM −1.70,
LSZH −1.58, LIRF −1.38, LTFM −1.20, LEMD −0.82, EDDF −0.70, LEBL −0.48, LFPG −0.48, EGLL −0.47, EDDM +0.32.
2026 projection −0.735 (all airports), −0.485 (LIRF frozen), **−0.413 on the rows production changes (no LIRF, no LSZH:
the build takes LSZH and LIRF rows from v62).** Hour inputs show no 2025 → 2026 drift (|SMD| ≤ 0.09).
Built as **v64** (`data/submissions/smart-jigsaw_v64.parquet`, md5 6f482bdb3dfd62f842c5885bc3f39d38).
