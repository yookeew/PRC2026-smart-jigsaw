# H: intra-day level from ADS-B-tracked departures -- pre-registration

Written 2026-10-10 **before any number below was computed** except two scratch checks (PROGRESS §136/§137 (§134/§135 before the merge)): on the
Jan + Jul 2025 holdout chain after G2, the intra-day deviation (3 h block minus day) of tracked rows' residual correlates
+0.07 to +0.49 per airport with the same deviation of untracked rows' residual (EDDF 0.48, EHAM 0.47, LEBL 0.49,
LEMD 0.43, EGLL 0.32, EDDM 0.07; LFPG and LTFM have no tracked rows). That check used labels for tracked rows; the
method below uses only the ADS-B pushback. Base: v63 = 238.3173 (G2x8).

## Method

Rows: departures at the 8 G2 airports. q = G2 output (2025: leave-period-out G2 on the 2025 frame of `tests/v57_dsm.py`).
Tracked = ADS-B tier in (dwell, pass, fb_dwell) (`cache/adsb_pushback_v4`, as `pru_recon2.TRACKED`).
- w_i = clip((T_i − adsb_pushback_i) − q_i, −1200, 1200) on tracked rows (ADS-B taxi minus our prediction).
- For each untracked row j: m_j = mean of w_i over tracked rows of the same airport with |T_i − T_j| ≤ 90 min, n_j their count;
  d_j = mean of w_i over tracked rows of the same airport and takeoff UTC date.
- **H1** z_j = (m_j − d_j) · n_j / (n_j + 10) (intra-day deviation only; G2 owns the day level).
- **H2** z_j = m_j · n_j / (n_j + 10) (local level, day part included).
- q'_j = max(q_j + β_a · z_j, 0) on untracked rows (n_j ≥ 1); tracked rows unchanged. β_a per airport: OLS through
  the origin of (y − q) on z over untracked rows with y ≤ 18,000 s.

## Decision

Primary population: the Jan + Jul 2025 holdout chain (period P01, the production chain's post-stages), β cross-month
(fit on January apply to July and vice versa). Secondary: the five OOF periods (corrector-level p, no ADS-B stages),
β leave-period-out among them.
A variant passes iff: (1) chain months pooled full ≤ −0.05 s on the 8 airports' rows; (2) chain body ≤ 0; (3) January and
July each full ≤ 0; (4) OOF periods pooled full ≤ 0; (5) February and June (OOF) each full ≤ 0.
Pick the passing variant with the lowest chain full. Report per airport, per month, the 2026 projection onto 238.3173.

## Build if it passes

β_a from both chain months (Jan + Jul 2025). On v63 (and the four-month file on its G2 output), untracked rows at the
8 airports shifted by β_a · z_j computed from the 2026 tracked rows; LOBT window last; every other row as v63.
**Keep iff board < 238.3173.**

## Outcome H1/H2 (2026-10-10; `logs/intraday_trk.log`): FAIL

H1 chain full +0.004, H2 +0.031; betas ≈ 0 everywhere. Cause: the "tracked" set (dwell, pass, fb_dwell, copied from the
day stage) has ADS-B taxi − label RMS 912 s (median |d| 191). Per tier (scratch `adsb_err.py`): fb_dwell / fb_appear are
off by +230 to +1,200 s at the median with < 3% within ±60 s of it; `appear` (not in that set) is the cleanest (70–92%
within ±60 s of its per-airport median), `dwell` next (32–82%).

## Amendment A1 (written after the H1/H2 failure, before H3 was run)

**H3a / H3b** = H1 / H2 with the signal rows redefined: tiers `appear` and `dwell` only;
w_i = clip((T_i − pb_i) − b_{airport, tier} − q_i, −300, 300), b = median of ADS-B taxi − label per (airport, tier)
from the 2025 periods other than the evaluated one (production: all of 2025). Rows shifted: every other row at the 8
airports (with ≥ 1 signal row within ±90 min). Same decision rule; pick the passing one with the lowest chain full.

## Outcome H3a/H3b (2026-10-10; `logs/intraday_trk_h3.log`): FAIL

Signal rows now ADS-B taxi − label RMS 166 s (median |d| 42). H3a chain full −0.001 (needs ≤ −0.05), Jan −0.002 /
Jul +0.002, OOF −0.036 (Feb −0.013, Jun −0.042); H3b chain +0.059. Betas are now positive and consistent (H3a 0.1–0.5),
but on the production chain there is nothing left: the neighbour stage already uses this information. Not built.
