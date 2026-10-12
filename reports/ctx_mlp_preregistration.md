# MLP member in the context corrector (CTX_MLP) -- pre-registration

Written 2026-10-11 **before any holdout (Jan / Jul 2025) number of this arm was computed**. Base: v66 = 236.8 (team
build; its code is not on `main` yet). Team decision (2026-10-11): try a neural-network member despite DESIGN.md's
"no neural networks" line; PyTorch 2.14.1 (CPU wheel, BSD) installed in `.venv`, to be declared in `pyproject.toml` only
if this passes.

**Disclosed (2025 data already used, scratch screens, session scratchpad `mlp_corr.py` / `scr_run.py`):** member `r` with
blocks 11 + 12 (v65's inputs), fitted on 8 OOF months, scored on two held-out month pairs; tree corrector (ensemble2)
vs 0.7 × tree + 0.3 × MLP, non-LIRF rows:

| held out | trimmed | body | per month (trimmed) |
|---|---|---|---|
| Feb + Aug | −3.80 | −0.27 | Feb −7.56 (LTFM snow-day NM-unmatched rows), Aug −0.40 |
| May + Nov | −0.68 | −0.33 | May −0.51, Nov −0.85 |

The MLP alone is worse than the trees (body +2.2 to +3.7); error correlation with the trees 0.95–0.96. The weight
0.3 was chosen on those screens (0.2 / 0.3 / 0.5 tried) and is fixed here.

## Arm

- **MLP member** (`CTX_MLP=1`, `src/models/ctx_mlp.py`, through `src/post/ctx_armc.py`): member `r`'s inputs, rows and
  folds (`ARMC_ARM=c1 CTX_ENS=2 CTX_RWY=1 CTX_RO=1 CTX_HOUR=1 CTX_HOUR2=1`, tag `_rh2mlp` → `ctxc1r_rh2mlp`); target
  y − s; embeddings for apt_c / stand_pfx; 512-256-128 SiLU MLP, Huber δ = 1,000 s, AdamW one-cycle, 12 epochs, best
  epoch on the inner days (held out of training); seed 0; one fit per fold, as the tree corrector.
- **Tree part:** v65's members refitted here (`run_v66_ctl.sh`, tag `_rh2`) → `ctxc1sel_rh2`. The same file is the
  control and the tree part of the treatment, so the gate measures the MLP blend only (no refit noise between arms).
- **Treatment A** `ctxc1sel_rh2m` = 0.7 × ctxc1sel_rh2 + 0.3 × ctxc1r_rh2mlp at every airport
  (`src/post/ctx_mlp_blend.py`).
- **Treatment B** `ctxc1sel_rh2mb` = the same blend only at the airports served by member `r` (not EHAM / LSZH / EDDF /
  LEBL, where the tree part is member `rb` with own-row ADS-B inputs the MLP does not see); tree part elsewhere.

## Gate (conditions of reports/ctx_hour_preregistration.md, as v64 / v65)

1. Holdout chain (`tests/v50_chain_gate.py`, `ARMS=all`, `RO_FILE=reocc_feats.parquet`), each treatment vs control,
   LIRF NM-unmatched rows as v41 in all.
2. OOF months, corrector level, each treatment vs control (`tests/ctx_hour_gate.py`, `CTL=rh2`, `CTL_DIR=cache/v66`).

**Pass iff:** (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul; (c) full ≤ 0;
(d) body ≤ +0.10; (e) OOF trimmed better in February and in June and in ≥ 7 of 10 OOF months.
**Pick:** if both pass, the lower holdout-chain trimmed RMSE; if one passes, that one.

## Build if it passes

Production members: tree (`ctx_armc.py prod`, r and rb, tag `_rh2`, then `sel`), MLP (`CTX_MLP=1 ... prod`), blend.
Production chain (`run_v65_chain.sh` steps) for the picked treatment and for `ctxc1sel_rh2`; full builds B(·) = v63's day
recipe on v54 + delta (`v64_delta.py`, `FREEZE_LIRF=1 v50_assemble.py`, `day_alloc_build.py`; LSZH / LIRF rows from
v65, which equal v62's). **v67 = v66 + (B(treatment) − B(ctxc1sel_rh2))**, then clip at 0 and round; LIRF and LSZH rows
unchanged. Keep iff board < v66's score (236.8…).

## Outcome (2026-10-11; `logs/v66/ctx_mlp_gate_rh2m.log`, `logs/v66/ctx_mlp_gate_rh2mb.log`, `logs/v66/run_v66_mlp_seq.out`)

Control `ctxc1sel_rh2` refitted here (`run_v66_ctl.sh`, early stopping 1,082–1,950 rounds); MLP member `ctxc1r_rh2mlp`
best epochs 5–9 of 12. Holdout chain control: trimmed 240.23, full 300.10, body 189.96.

| vs control | trimmed | full | body | trimmed Jan / Jul | body Jan / Jul | OOF trimmed | pass |
|---|---|---|---|---|---|---|---|
| A `ctxc1sel_rh2m` (all airports) | −0.877 | −0.692 | −0.663 | −0.550 / −1.085 | −0.239 / −0.962 | −0.93; Feb −4.29, **Jun +0.13**, Aug +0.32 (8/10) | no: (e) |
| **B `ctxc1sel_rh2mb` (member-r airports)** | **−0.849** | **−0.745** | **−0.637** | −0.834 / −0.874 | −0.724 / −0.580 | **−1.12; Feb −4.56, Jun −0.14 (9/10; Aug +0.15)** | **yes** |

- A fails (e) at the airports where the tree part is member `rb` (OOF EHAM +1.21, LEBL +0.86, LSZH +0.47): the MLP does not
  see the own-row ADS-B inputs there, as anticipated when B was registered.
- B per airport, holdout chain trimmed: EGLL −2.39, LIRF −1.67, LTFM −1.34, LFPG −1.11, LEMD −0.34, LEBL −0.12, LSZH −0.02,
  EDDF +0.17, EHAM +0.27, EDDM +0.69 (EHAM / LSZH / EDDF / LEBL are unchanged by B; their deltas are chain interactions).
- 2026 projection on the rows production changes (no LIRF, no LSZH): **−0.649 s** (all airports −0.954).
- **Pick: B** (the only arm that passes). Build: `run_v67_build.sh rh2mb`.

## Build (2026-10-11; `run_v67_build.sh rh2mb`, `logs/v66/run_v67_build.out`, `logs/v66/v67_build.log`)

- Production chain of `ctxc1sel_ro` rebuilt here: byte-identical to the other machine's (md5 98099262…).
- Full builds B(ctxc1sel_rh2) and B(ctxc1sel_rh2mb); B(ctxc1sel_rh2) differs from v65 by 22.3 s RMS (cross-machine refit
  noise of the tree part), which cancels in B(treatment) − B(control).
- Label-free (`tests/ctx_mlp_drift.py`, `logs/v66/ctx_mlp_drift.log`): MLP − tree on 2025 rows vs 2026 rows, mean ≈ 0 in
  both years (|SMD| ≤ 0.07 at the blended airports); spread larger in 2026 (EDDM sd 48 → 64, LFPG 65 → 86, LTFM 75 → 93).
- **v67** = v66 + (B(rh2mb) − B(rh2)): 281,821 rows changed, RMS 24.6 s, mean −0.22, max |Δ| 1,055; LIRF and LSZH identical
  to v66. Per airport RMS: LTFM 36.9, LFPG 33.9, EGLL 29.3, EDDM 22.2, LEMD 20.2, EDDF 20.1, EHAM 19.0, LEBL 15.4 (EDDF /
  EHAM / LEBL move through the post-stages' refit on the new corrector, as in the gate's holdout chain).
- `data/submissions/smart-jigsaw_v67.parquet`, md5 e22ac7a3933478c4887bedce85ff42ba. **Keep iff board < v66's score (236.8…).**

## Board (2026-10-11): v67 = 237.9138 vs v66 = 236.8 → +1.1. **REJECTED; v66 stays best.**

- v67's change vs v66 had RMS 24.6 s; a change uncorrelated with the truth would have cost ≈ +1.3 s, so on 2026 rows the
  change carried almost no signal, against −0.85 on the 2025 holdout chain.
- Most likely cause (label-free evidence only; no row-level inference from the score): MLP extrapolation on January 2026.
  RMS of (MLP − tree) on the blended airports, NM-matched: Jan 2025 66 s, Jul 2025 71 s, **Jan 2026 86 s** (p99 372 vs
  265–279), Jul 2026 53 s; NM-unmatched Jan 2026 329 s (p99 1,302). January 2026 (storm month) lies outside the 2025 input
  range, where a tree corrector saturates and the MLP does not. The holdout months are in-distribution and could not show it.
- The drift check was registered as "reported, not a gate condition"; it showed the spread growing (sd +30%, p99 +50%).
  **Lesson:** any member that extrapolates (non-tree) needs a stop rule on its 2026 output spread, as block 13 had for its
  inputs.
- Not done, on purpose: a January-only or July-only upload to locate the loss (that would read month-level labels from the
  board, §82 / §115). The MLP line is closed for the live board.

*(Reverted 2026-10-11 after the board result, team decision: the MLP code (`src/models/ctx_mlp.py`, the `CTX_MLP` hook in `src/post/ctx_armc.py`, `src/post/ctx_mlp_blend.py`, `src/post/v67_build.py`, `tests/ctx_mlp_drift.py`, the `run_v66_*.sh` / `run_v67_build.sh` scripts) was removed and PyTorch uninstalled from `.venv`. The record is `reports/ctx_mlp_preregistration.md`, this section and `logs/v66/`; `smart-jigsaw_v67.parquet` is kept as a rejected upload. `tests/ctx_hour_gate.py`'s `CTL_DIR` switch stays (default unchanged).)*
