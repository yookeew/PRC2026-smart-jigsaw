# Holdout eval — lgb_direct_nolirf_nb

- engine: **lgb**   |   generated: 2026-10-06 16:59
- target direct_nolirf  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct_nolirf=2594 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 390.1 | 155.6 | -7.5 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 979.0 | -117.6 | 1274 | 26,528 |
| LFPG | 578.5 | +1.8 | 1052 | 39,590 |
| EGLL | 284.9 | +5.3 | 1382 | 40,210 |
| LTFM | 258.0 | -6.0 | 1070 | 46,539 |
| LEBL | 220.8 | +4.4 | 957 | 28,985 |
| LSZH | 207.8 | -0.7 | 759 | 22,289 |
| EHAM | 199.3 | +1.1 | 798 | 40,949 |
| EDDF | 189.8 | +2.5 | 875 | 36,830 |
| EDDM | 184.1 | -1.1 | 832 | 27,091 |
| LEMD | 177.5 | +9.1 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 363.0 | 153,660 |
| 2025-07 | 410.6 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 264.4 | 339,016 | 45.2% |
| False | 2321.9 | 5,323 | 54.8% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1998.9 | 2,452 | 18.7% |
| False | 353.0 | 341,887 | 81.3% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 191.1 | +110.7 | 34,434 |
| d1 | 569–671 | 167.0 | +82.5 | 34,434 |
| d2 | 671–770 | 165.0 | +60.7 | 34,434 |
| d3 | 770–842 | 163.7 | +54.0 | 34,434 |
| d4 | 842–923 | 179.8 | +38.4 | 34,434 |
| d5 | 923–1020 | 177.1 | +19.5 | 34,434 |
| d6 | 1020–1134 | 189.8 | +2.2 | 34,434 |
| d7 | 1134–1266 | 216.3 | -22.7 | 34,434 |
| d8 | 1266–1505 | 240.9 | -76.6 | 34,434 |
| d9 | 1505–88132 | 1094.8 | -343.5 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 65 | 614 | 943 | 1441 | 68894 | 0 | 37 |
