# LIRF NM-unmatched delayed departures: the airline's own echo record (v72) -- pre-registration

Written 2026-10-12 before the v72 build. Base: v71 (235.493, current best). Analysis and checks in PROGRESS.md §157–§158
(screens disclosed below; every number here was computed before this file). Script: `src/post/lirf_airline_record.py`.

## Evidence

**1. Echo on v48-scope rows is an airline practice.** Scope: LIRF NM-unmatched departures, 1 h ≤ T − S < 6 h, stack p < 0.99, no
arrival twin. The label is the echo (block = SCHED) or a late pushback with a stale LOBT. In 2025:
- always echo: ETH 21/21, TKJ 20/20, ISR 56/56, LAV 22/22, TWB 19/19;
- almost never: ITY 3/61, KLM 0/17, EWG 0/14, ELY 0/12, EZY 1/24.
- q2 (v48's echo probability) prices this only partly: ETH rows q2 0.26–0.91, EWG rows 0.03–0.83.
- Stand reuse and irregular flight numbers, which raise q2, do not override the record. Low-echo airlines: stand reuse 0/9 echo,
  irregular well-formed flight numbers 0/2.
- The record is persistent within 2025: ETH 9/9 → 12/12, ISR 24/24 → 32/32, ITY 2/21 → 1/40, KLM 0/8 → 0/9.

**2. Validation (2025, records and the combination below re-estimated without the evaluated month):**

| check | result |
|---|---|
| leave-one-month-out | −4.13e8, 12/12 months, 230 rows helped / 19 hurt |
| production holdout chain | Jan −3.1e7 (11 / 5), Jul −8.0e7 (40 / 8) |
| past-only records, gap ≥ 1 month / ≥ 4 months | −7.3e7 (6/9 months) / −7.2e7 (6/6) |
| per airline | all 11 airlines better; without any one airline −2.9e8 to −4.1e8 |
| airline-cluster bootstrap of the total | 5–95% −6.3e8 … −2.2e8 |
| calibration of π on the changed rows | within each bin |

**3. 2026 persistence (label-free):**
- LIRF early-arrival echo rate per airline, 2025 vs 2026: corr 0.91 over 55 airlines. The decisive airlines keep their level, except
  KMM (0.79 → 0.33), which is excluded.
- DLA discipline (NM-matched share of departures ≥ 1 h late): ITY 0.976 → 0.974, EWG 0.935 → 0.949, EZY 0.911 → 0.878.
- ELY's 2026 rows are the same flights, stands and pattern as its 12 non-echo 2025 rows.

## Candidate

**v72 = v71 with v48-scope rows of decisive airlines changed** (`src/post/lirf_airline_record.py`; constants from all of 2025,
printed by the build):
- **Decisive airline:** n ≥ 12 scope rows in 2025 and (e + 1)/(n + 2) ≤ 0.1 (low echo) or ≥ 0.9 (high echo).
- **Echo probability:** π = σ(0.132 + 0.899·logit q2 + 0.728·logit((e + 1)/(n + 2))). This is a logistic fitted on 2025 with
  out-of-month records and out-of-month q2; LOMO log-loss 0.343 vs q2's 0.387 on all rows with a record.
- **Value:** max(π·offc + (1 − π)·m(ub), 600), where m(ub) is the mean of the 2025 late-branch labels ≤ ub = T − S − 3,606.
- **Rows:**
  - the input value equals the replayed chain P0 → v45 → v47 → v48 (±1 s) or v71's 600 s floor of it;
  - no twin (exact or normalised);
  - low-echo airlines: well-formed flight number only (null / doubled / odd / ADES = LIRF are key failures);
  - low-echo airlines ≥ 4 h late (beyond every 2025 low-echo row): only if q2 ≤ 0.2 agrees.
- **Persistence guard:** skip an airline whose 2026 early-arrival echo rate moved by > 0.15 and > 3 standard errors from the same
  2025 months (≥ 20 arrivals).

**Disclosed choices made after seeing 2025 or 2026 results:**
- n ≥ 12 (n ≥ 8 admitted RAM 0/8, whose one echo cost March +1.5e8; no 2026 row changes because of it);
- m(ub) instead of the chain's implied non-echo branch (≈ 0 on 22 of the 57 candidate rows; 2025 LOMO equal, holdout −1.11e8 vs
  −1.26e8);
- ADES = LIRF as a key failure (ITY614 dropped);
- the guard's 0.15 materiality (ITY's 3.1% → 2.2% was a false alarm at n = 3,531);
- the 4 h support rule (drops ITY8015 only; it changes no 2025 row).

## Stop rules (label-free)

1. Only LIRF NM-unmatched v48-scope rows change; each passes the guard; IDs exactly the template's; no nulls; no negative values.
2. Live board: 54 rows change (scratch dry run). Any other count stops the build until it is explained.
3. Values within [600, max(offc, 600)].
4. Σd² ≤ 2.0e8 (dry run: 1.04e8).

## Upload rule

**Keep iff board < 235.493.**

Projection (labels drawn from π, i.e. 2025 behaviour persists): **−0.64 s**, sd 0.41, worse in 11%, 5–95% −1.06 … +0.19.

| alternative truth | effect |
|---|---|
| the record carries nothing in 2026 (labels from q2) | +0.13 s |
| midway | −0.25 s |
| every move wrong (bound) | +0.64 s |

Largest moves:

| row | v71 → v72 (s) |
|---|---|
| EWG70X (Jan) | 9,566 → 3,438 |
| ELY384 | 6,260 → 3,051 |
| ETH713 | 1,608 → 4,720 |
| ITY108 (Jan) | 6,057 → 1,222 |
| KLM76R (Jan) | 4,684 → 2,197 |

**Left out:** ITY8015 (July, 5.15 h late, q2 0.80 against ITA's 3/61). Its own projection, with the bootstrap π range 0.21–0.50
against break-even 0.62, is about −0.5 s ± 1 s. It is a single-row bet beyond the validated delay range, so not part of this candidate.

**Blind board:** the same script with `PRC_SCOPE=final4m` on the blind file after the v71 stage. P0 is
`data/submissions/final4m/f4m_a.parquet`, and the persistence guard runs over Jan/Feb/Jun/Jul (KMM skipped). Jan/Jul rows then
come from the live v72 through `final4m_compose.py`. Scratch dry run on the v71 blind dry run: 37 Feb/Jun rows change.

## Build (2026-10-12; `logs/v72/v72_build.log`)

- `smart-jigsaw_v72.parquet`, md5 8e394b0c1352867d47affbbad1fe0465: 344,841 rows, template ID order, same schema as v71, no nulls.
- 54 rows changed (Jan 10, Jul 44), Σd² 1.04e8; identical to the scratch dry run. Stop rules 1–4 pass.

## Next upload: include ITY8015 (team decision, 2026-10-12)

**The next upload after v72 adds ITY8015** (MVT_ID 213150756, July, off 18,538 s, q2 0.80, ITA record 3/61) at the lever's own
value: 16,231 → **7,463** (π 0.365, the same formula as the other rows). Build it by running the script with the 4 h support rule
disabled for that row only, or simply set that one row on top of v72; nothing else changes.
- Expected ≈ −0.5 s (π 0.21–0.50 against break-even 0.62), swing about ±1 s on that row alone.
- Keep rule: keep iff board < the v72 score (or < 235.493 if v72 is rejected).

## Outcome (2026-10-12)

- **Board: v72 = 234.5516 (−0.941 vs v71 = 235.493). Kept; new best.** Inside the projected range (central −0.64, 5–95% −1.06 … +0.19).
- ITY8015 stays scheduled for the next upload (2026-10-13): v73 = v72 + that row; keep iff board < 234.5516.
- **Board (2026-10-13): v73 = v72 + ITY8015 = 235.2757 (+0.724). Rejected by its keep rule (< 234.5516); v72 stays.** The 4 h support
  rule that left ITY8015 out of v72 was fixed before any board result and is unchanged; nothing is refit from this score (§82).

## v74 (written 2026-10-13, before upload)

Base: v72 (234.5516). Script: `ALR_V74=1 src/post/lirf_airline_record.py smart-jigsaw_v72.parquet smart-jigsaw_v74.parquet`.
Analysis in PROGRESS.md §160. ITY8015 is frozen at its v72 value.

| rule | rows | 2025 evidence |
|---|---|---|
| E: record n ≥ 8, ≤ 0.2 / ≥ 0.8 (null flight excluded; guard skips UAL, KMM) | 24 (AEZ 8 up) | LOMO −394e6, 12/12 months; holdout Jul −138e6; past-only gap ≥ 4 −110e6 |
| R: L3 cap, months 6–10, q2 ≥ 0.3, rows not moved by v72 / E | 41 | June–October each better; July ratio −1.8 (LOMO) / −3.0 (holdout) |
| D: departures 0–60 min early → 600 s | 5 | 11 of 13 labels 591–612 s (others 351, 1,131) |

- **Disclosed choices made after seeing 2025 results:** the 0.2 / 0.8 and n ≥ 8 thresholds; the months 6–10 window; excluding null
  flight numbers; the −60 min lower edge of D (outside it, three 2025 rows ≥ 6 h early sit at 1,200–1,700 s).
- **Stop rules:** 70 rows exactly; only LIRF NM-unmatched rows; values within [600, max(offc, 600)]; the frozen row is unchanged.
- **Build:** md5 5b18d6a07ce0c4c9ec708bf984a67864; 70 rows; Σd² 4.62e7.
- **Projection:** −0.34 … −0.41 s if the records persist; −0.20 … −0.27 midway; −0.06 … −0.13 if the records carry nothing.
- **Upload rule: keep iff board < 234.5516.**
- **Board (2026-10-13): v74 = 234.6769 (+0.125 vs 234.5516). Rejected; v72 stays.** Outside the projected range (best case
  −0.41, worst projected −0.06). Nothing is attributed to the three parts and no part is re-uploaded (§82). The blind file does not
  use `ALR_V74`.
