# LIRF NM-unmatched hedge repair and shifted block times (v71) -- pre-registration

Written 2026-10-12 before the v71 build. Base: v70 (236.2290, current best). Analysis in PROGRESS.md §155 (screens
disclosed below; every number here was computed before this file).

## Evidence

**1. The hedges' non-echo branch is broken.** v45 / v48 hedge as q·offc + (1 − q)·P_ne, with the non-echo branch recovered
from the chain: P_ne = (P − p·offc) / (1 − p). v47 uses the same P_ne on twin rows (clip(P_ne, 0, up)).
- In 2026, P_ne < 0 on 60 of 287 v48-scope rows (2025 holdout: 23 of 328). The hedge then clips to 0 s: 16 LIRF NM-unmatched
  rows 1–3.5 h late sit at 0–120 s. In 2025 only 2 of 1,373 such labels are below 120 s.
- On 2025 non-echo scope rows, P_ne predicts the taxi worse than a constant: RMSE 1,524 vs 1,020 for 1,000 s.

**2. Twin rows** (the departure's arrival twin is NM-matched and gives its LOBT; tl = T − LOBT_twin, up = tl − 3,606, lo = tl + 3,606).
- When the join window excludes the echo (tl ≥ 3,726 and off > up), every 2025 label is an ordinary taxi: 76/76 ≤ up,
  median 903 s. v47's value has RMSE 817 there; min(E[y | twin non-echo, y ≤ up], up) has 0.14 of its SSE. In 2026, six
  such rows sit at exactly 0 s.
- Forced twins (tl < 3,726: the block must lie before LOBT − 3,606) with the echo still possible (off ≥ lo): 21 of 23 echo
  in 2025.
- Normalising doubled flight-number prefixes (ITYTY672 → ITY672) finds 9 more 2025 twins (all obey the window) and one in
  2026 (RYRR88TM).

**3. Shifted block times at LIRF.** Some handlers record the block 1 h or 2 h early (the local UTC offset), so the NM join
fails on time. 2025 LIRF NM-unmatched departures with 0 < T − S < 1 h:

| flight-number format | shifted (y ≈ taxi + 1 h / 2 h) | echo | normal |
|---|---|---|---|
| null / doubled prefix / odd (the join fails on the key) | 0 / 43 | 33 | 10 |
| well-formed (the join fails on time) | 33 / 53 | 13 | 1 |

- Shifting airlines: BEL 14/14 (all summer cases +2 h), AEA, AEZ (4/6), BAW, UAL, CTN, MSR, QTR, LZB, GFA.
- Well-formed echo cases: ITY, RYR, WMT, VLG, LOT, NOS.
- Forced twins with the echo impossible (off < lo): 6/7 shifted (+1 h or +2 h).
- The mode is LIRF-only: none of the other nine airports has it. The frozen LIRF values do not price it (2025 holdout:
  frozen 1,100–2,400 s vs labels 7,800–8,600 s).

**4. Validation, labels hidden.**
- 2025 holdout (Jan + Jul; every statistic re-estimated from the other ten months; production-recipe holdout chain):

| component | rows | ΔSSE | helped / hurt |
|---|---|---|---|
| N1 | 8 | −2.1e6 | 7 / 1 |
| T1 | 21 | −2.06e7 | 17 / 4 |
| T2 | 6 | −2.97e7 | 3 / 3 |
| L2 key | 6 | −2.6e6 | 4 / 2 |
| L2 shift | 6 | −1.96e8 | 6 / 0 |
| L2 twin | 2 | −4.9e7 | 2 / 0 |

- Leave-one-month-out over 12 months:
  - L2 vs the frozen proxy (stack Pb; correlation 0.88 with the frozen chain on the holdout): −6.35e8, better in 12/12 months.
  - N1: 26/26 rows helped, 8/8 months.

## Candidate

**v71 = v70 with LIRF NM-unmatched rows changed only as below** (`src/post/lirf_hedge_repair.py`). Constants are estimated
from all of 2025 by the build and printed. Values from the analysis are given for reference.

- **L1 -- N1:** v48-scope rows (non-twin, 1 h ≤ off < 6 h, p < 0.99): floor at **600 s** (2025 late-branch q05 = 601).
- **L1 -- T1:** twin rows with the echo excluded: min(mean of 2025 twin non-echo labels ≤ up, up). If fewer than 10 such
  labels exist, use min(mean of the 10 smallest, up).
- **L1 -- T2:** forced twin rows with off ≥ lo: P_E·offc + (1 − P_E)·(lo + 900), with P_E = (echo + 1) / (n + 2) over 2025
  forced twins with off ≥ lo (≈ 0.88).
- **L2 -- key:** non-twin rows with 0 < off < 1 h and a key-failure flight number: PE_KEY·off + (1 − PE_KEY)·M_N.
  - PE_KEY = echo share among 2025 key-failure rows (≈ 0.767).
  - M_N = mean of 2025 "normal" labels in this population (≈ 958 s).
- **L2 -- shift:** non-twin rows with 0 < off < 1 h, a well-formed flight number, and an airline with **≥ 2 shifted 2025
  rows**: psh·(tzo + M_SH) + (1 − psh)·(PE_TM·off + (1 − PE_TM)·M_N).
  - psh = (s + 2·P0T) / (n + 2), from the airline's 2025 counts.
  - P0T = pooled shift rate of well-formed rows (≈ 0.67); PE_TM = echo share of their non-shifted rows (≈ 0.93).
  - M_SH = median taxi of shifted rows (≈ 838 s); tzo = Rome's UTC offset at T.
- **L2 -- twin:** forced twin rows with off < lo and a well-formed flight number.
  - Summer (UTC+2): 0.5·max(lo, 3,600 + M_SH) + 0.5·max(lo, 7,200 + M_SH).
  - Winter: max(lo, 3,600 + M_SH).
- **Guard:** a row changes only if its input value equals (±1 s) the value the replayed hedge chain gives it. The chain is
  P0 → v45 (lirf_signals) → v47 (lirf_twin_bounds) → v48 (lirf_hedge2), with P0 = v42 on the live board and f4m_a on the blind
  board. So no row touched by a later stage changes.

**Not changed:**
- L3 (the cap on the non-echo branch, §155). Its sign depends on 2026 echo calibration; it is in the script behind
  `LHR_L3=1`, default off.
- Rows ≥ 6 h late, NM-matched rows, other airports.

## Stop rules (label-free)

1. Only LIRF NM-unmatched rows change; every changed row passes the guard and carries one tag; IDs exactly the template's;
   no nulls; no negative values.
2. Live board, rows per tag as in the scratch build: N1 29, T1 22, T2 13, L2 key 16, L2 shift 3, L2 twin 2. Any difference
   stops the build until it is explained.
3. Value checks: N1 rows end at exactly 600; T1 ≤ up; T2 between lo + 900 and offc; L2 key between min(off, M_N) and
   max(off, M_N); L2 shift upward and ≤ tzo + 3,600; L2 twin ≥ lo.
4. Σd² ≤ 1.5e8 (scratch build: 1.33e8).

## Upload rule

**Keep iff board < 236.2290.**
- Projection −0.93 s: L1 −0.35 (sd 0.07), L2 −0.59 (sd 0.25). This is a simulation with labels drawn from the 2025 class
  mixtures; 5–95% ≈ −1.35 … −0.5.
- If every move were uncorrelated with the truth, the cost would be about +0.8 s.

**Blind board:** the same script with `PRC_SCOPE=final4m` on the blind file after the v65–v70 ports (input = that file,
P0 = `data/submissions/final4m/f4m_a.parquet`); Jan/Jul rows then come from the live v71 through `final4m_compose.py`.
Scratch run on `blind_v5_lszh`: 105 Feb/Jun rows change.

## Outcome (2026-10-12; `logs/v71/`)

- Build as registered: constants P_E 0.880, PE_KEY 0.767, P0T 0.667, PE_TM 0.929, M_N 958, M_SH 838. 85 rows (N1 29, T1 22,
  T2 13, L2 key 16, L2 shift 3, L2 twin 2), Σd² 1.33e8, stop rules 1–4 pass.
  `smart-jigsaw_v71.parquet`, md5 7a9508ba999d6c56efe3c33978e8fd7f.
- **Board (2026-10-12): v71 = 235.493 (−0.736 vs v70 = 236.2290). Kept; new best.** Inside the projected range (−0.5 … −1.35,
  central −0.93).
