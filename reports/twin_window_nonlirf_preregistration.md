# Arrival-twin join window at the non-LIRF airports -- pre-registration

Written 2026-10-11 **before the 2025 test below was computed**. Base: v64 = 237.4538 (§140).
**Disclosed (PROGRESS §139, §140, scratch `twin_window.py`, `twin2.py`):** on 2025 labels, NM-unmatched departures whose
flight number has an NM-matched arrival at another of the 11 airports (same ADEP in NM, landing 15 min – 15 h after
takeoff; "twin") pushed back more than 3,606 s from the twin's LOBT in 78% (EHAM) to 100% of cases per airport
(905 non-LIRF twins). On 2026 Jan/Jul, 16 of v64's non-LIRF twin rows lie inside that window. LIRF is not touched
(team rule; its twin rows are handled by v45/v47/v48).

## Rule (non-LIRF airports only)

For a twin row with g = T − LOBT_twin, the excluded window for the taxi y is (g − 3,606, g + 3,606). If the prediction p
lies inside it, move p to the nearest allowed value: g − 3,606 if that is ≥ 0 and nearer, else g + 3,606. Rows outside
the window and all other rows are unchanged. Applied after the LOBT-window stage, i.e. on the final file.

## Test on 2025 (the chain's 2025 predictions, non-LIRF rows: `tests/v57_dsm.py::frame25` p after G2 leave-period-out)

Pass iff (1) pooled ΔSSE < 0 over the 2025 twin rows; (2) ΔSSE ≤ 0 at ≥ 7 of the 9 non-LIRF airports;
(3) ΔSSE ≤ 0 on Jan + Jul 2025 pooled. Reported: rows moved, per airport, the 2026 projection.

## Build if it passes

v65 = v64 with the rule (Jan/Jul). **Keep iff board < 237.4538.** Blind board: the same rule on `blind_v5` (all four
months; Jan/Jul rows then equal v65).

## Outcome (2026-10-11; scratch `twin_gate.py`): FAIL (1) and (3); not built

2025 non-LIRF twin rows with a chain prediction: 799; 83 inside the window and moved (mean |move| 1,382 s).
Pooled ΔSSE **+1.87e8** (≈ +0.20 s on the 2025 frame); Jan + Jul ΔSSE +6.4e7 (22 rows). Per airport ΔSSE (helped / hurt):
LTFM −9.1e7 (26 / 0), EGLL −2.1e7 (11 / 0), EDDM −4.9e5 (5 / 0), LEMD −1.3e5 (5 / 0); EHAM +1.05e8 (1 / 8),
LEBL +1.10e8 (4 / 8), LFPG +4.9e7 (7 / 3), EDDF +2.9e7 (2 / 2), LSZH +7.1e6 (0 / 1).
The window itself holds, but "nearest boundary" puts the prediction on the wrong side for EHAM / LEBL / LFPG, where the
true off-block is far beyond the other edge. LTFM is clean (26 / 0), as in gracious-tornado's README (LIRF and LTFM
only), but an LTFM-only rule would be chosen on these same rows, and on 2026 it moves 3 rows (≤ 0.04 s even if all are
right). Not pursued.
