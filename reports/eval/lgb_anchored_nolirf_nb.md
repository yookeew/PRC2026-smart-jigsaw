# Holdout eval — lgb_anchored_nolirf_nb

- engine: **lgb**   |   generated: 2026-10-06 23:51
- target anchored_nolirf  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at anchored_nolirf=2083 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| nan | nan | +nan |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LEMD | nan | +nan | 1011 | 35,328 |
| EDDF | nan | +nan | 875 | 36,830 |
| LFPG | nan | +nan | 1052 | 39,590 |
| LIRF | nan | +nan | 1274 | 26,528 |
| LSZH | nan | +nan | 759 | 22,289 |
| EGLL | nan | +nan | 1382 | 40,210 |
| LEBL | nan | +nan | 957 | 28,985 |
| EHAM | nan | +nan | 798 | 40,949 |
| EDDM | nan | +nan | 832 | 27,091 |
| LTFM | nan | +nan | 1070 | 46,539 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | nan | 153,660 |
| 2025-07 | nan | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 262.2 | 339,016 | nan% |
| False | nan | 5,323 | nan% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | nan | 2,452 | nan% |
| False | nan | 341,887 | nan% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | nan | +nan | 34,434 |
| d1 | 569–671 | nan | +nan | 34,434 |
| d2 | 671–770 | nan | +nan | 34,434 |
| d3 | 770–842 | nan | +nan | 34,434 |
| d4 | 842–923 | nan | +nan | 34,434 |
| d5 | 923–1020 | nan | +nan | 34,434 |
| d6 | 1020–1134 | nan | +nan | 34,434 |
| d7 | 1134–1266 | nan | +nan | 34,434 |
| d8 | 1266–1505 | nan | +nan | 34,434 |
| d9 | 1505–88132 | nan | +nan | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 617 | 951 | 1486 | 32102 | 2 | 5324 |
