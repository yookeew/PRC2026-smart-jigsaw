# v66's context corrector averaged over three LightGBM seeds (`ctxc1sel_rf3`) -- pre-registration

Written 2026-10-12 (≈ 01:00) **before any number below was computed**. Base: v66 = 236.8136 (best; §138, board −0.26 vs
v65). Focus (user, 2026-10-12): one candidate, the largest projected gain on the Jan/Jul board, general (not Jan/Jul-specific),
no board fitting.

## Why

- **The new-signal screens are dry** (PROGRESS §140, `tests/flow_screen/dir_screen.py`, `logs/dir_screen.log`, run on top of
  `ctxc1sel_rf`'s OOF + holdout, the §137 paired harness): departure-direction (30° bearing sector) flow −0.108 (body −0.013,
  mostly LIRF); LOBT / AOBT_3-order overtaking −0.186 (body −0.166, January +0.03); runway-configuration change ahead −0.100;
  all three together −0.306 (body −0.179; Jan −0.12, Jul −0.64, Nov–Dec 0.00; LTFM / LIRF −0.81, EDDM +0.13).
  Block 14 screened −0.32 and failed its gate (§139), so none of these is worth tonight's one refit; D+L+C is kept as a
  candidate block for a later round (it stacks on top of this change).
- **The corrector's refit variance is large and untouched.** Fits that differ only in their random draws give corrector
  outputs 11.7–17.0 s RMS apart (§105e, §108); v66's delta itself is one draw. Averaging K draws keeps the signal and
  divides that variance by K.
- **Evidence on an earlier corrector (§108, `reports/ctx_seedavg_preregistration.md`):** 3 seeds vs v42's chain: trimmed −0.33,
  full −0.33, body −0.19 (Jan −0.51, Jul +0.03), 8/10 airports better. It failed only that rule's July-body condition; under
  the §135–§138 gate below it would have passed (a)–(d). Expected size now: 0.1–0.25 s of board RMSE.
- Orthogonal to every corrector block, so a pass stacks with later blocks.

## Change

- Two more fits of **both v66 members, unchanged** (inputs, rows, parameters, early stopping, ensemble2), with LightGBM seeds
  43 and 44: `PRC_CTX_SEEDS=43,44 PRC_ARMC_TAG=_rf` + v66's member flags → `ctxc1r_rf_s2`, `ctxc1rb_rf_s2` (each the in-process
  mean of the two seeds, `src/post/ctx_armc.py`'s existing switch).
- **Average with v66's own seed-42 fits** (the files v66 was built from, not refits): member = (1 · `ctxc1{r,rb}_rf` +
  2 · `ctxc1{r,rb}_rf_s2`) / 3, for the holdout file, the five OOF folds and the ranking rows (from the unrounded
  `armc_rank_*` output, then rounded as `prod` does) → `ctxc1r_rf3`, `ctxc1rb_rf3` (`src/post/ctx_seedavg_merge.py`).
- `sel` as always (rb at EHAM / LSZH / EDDF / LEBL, r elsewhere) → **`ctxc1sel_rf3`**.
- Feature-subsampling (feature_fraction 0.9) is the source of diversity; within one process a seed is exactly reproducible (§107).

## Gate

Control = v66's production corrector `ctxc1sel_rf` (holdout chain `cache/v66/gate_rf/chain_all.parquet`, OOF `cache/oof/ctxc1sel_rf`).
1. **Holdout chain** (`tests/v50_chain_gate.py`, `ARMS=all`, `RO_FILE=reocc_feats.parquet`, `CORR=ctxc1sel_rf3`, `SCR=cache/v67s/gate_rf3`).
2. **OOF months** (corrector level), labelled rows, y ≤ 5 h for trimmed.
Run: `CTL=rf CTL_DIR=cache/v66 TAG=rf3 GATE_DIR=cache/v67s BOARD=236.8136 tests/ctx_hour_gate.py`.

**Pass iff** (as §135–§138): (a) holdout chain trimmed pooled ≤ −0.10 s; (b) trimmed better in Jan and in Jul; (c) holdout
chain full ≤ 0; (d) holdout chain body ≤ +0.10; (e) OOF trimmed better in February and in June and in ≥ 7 of 10 OOF months.
(f) (input drift) does not apply: no input changes.

Also reported: RMS between the seed-42 fit and the seed-43/44 mean (holdout, ranking rows), per-airport deltas, 2026 projection.

## Build if it passes

As v66: production chain for `ctxc1sel_rf3` (`scripts/run_v67s_chain.sh`, identity-arm deltas in `cache/v67s/id_ctxc1sel_rf3`);
delta = (chain_rf3 + id deltas) − (chain_ctxc1sel_ro + id deltas) (`src/post/v64_delta.py`); p = v54 + delta (`FREEZE_LIRF=1
v50_assemble.py`); then v63's recipe (`day_alloc_build.py`: OPDI deltas, G2 at 8 airports, LSZH / LIRF rows from v62).
**v67** (`data/submissions/smart-jigsaw_v67.parquet`). **Keep iff board < 236.8136.** If the gate fails, nothing is built and
v67 stays free. The four-month version is built on the last day with the rest.

## Amendments

(none)

## Outcome (2026-10-12; `logs/v67s/ctx_seedavg_gate.log`, `logs/v67s/merge_*.log`, `logs/run_v67s.log`)

Run note (not an amendment; nothing in the design changed): the laptop was on battery for the first ≈ 40 min (CPU throttled), then AC.

| holdout chain vs `ctxc1sel_rf` | trimmed | full | body | trimmed Jan / Jul | body Jan / Jul | OOF trimmed | pass |
|---|---|---|---|---|---|---|---|
| **ctxc1sel_rf3** | **−0.412** | −0.320 | −0.338 | −0.344 / −0.460 | −0.344 / −0.335 | −0.292, 10/10 months (Feb −0.45, Jun −0.23) | **yes** |

- Non-LIRF: trimmed −0.404, full −0.281, body −0.328. Per airport trimmed: LFPG −1.07, EGLL −0.64, LIRF −0.64, LTFM −0.44, LSZH −0.31,
  EDDM / LEMD −0.17, LEBL −0.11, EHAM −0.04, EDDF +0.01.
- RMS seed 42 vs the seed-43/44 mean: holdout 18.7 (r) / 18.9 (rb) s, ranking rows 19.7 / 19.4 s.
- 2026 projection on the rows production changes: **−0.28 s.**
**PASS → v67 built.** `data/submissions/smart-jigsaw_v67.parquet`, md5 4cbe2902013a011db5c8bf395e19ef50. vs v66: 278,940 rows changed,
RMS 17.1 s, mean −0.01, max |Δ| 620; LIRF and LSZH identical. **Keep iff board < 236.8136.**
