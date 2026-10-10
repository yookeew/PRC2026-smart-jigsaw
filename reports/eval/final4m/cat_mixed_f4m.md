# Holdout eval — cat_mixed_f4m

- engine: **cat**   |   generated: 2026-10-09 22:03
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.05  |  rounds ceiling 8000  |  early-stopped at flip=7999/direct=7985 (inner-valid 2025-06)  |  full-refit rounds flip=8799/direct=8784
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 336.6 | 154.6 | +2.8 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 667.8 | +10.0 | 1274 | 26,528 |
| LFPG | 574.8 | +8.5 | 1052 | 39,590 |
| EGLL | 285.8 | +6.2 | 1382 | 40,210 |
| LTFM | 267.9 | -18.1 | 1070 | 46,539 |
| LEBL | 223.3 | +2.6 | 957 | 28,985 |
| LSZH | 208.9 | +2.3 | 759 | 22,289 |
| EHAM | 199.1 | +2.0 | 798 | 40,949 |
| EDDF | 190.5 | +5.8 | 875 | 36,830 |
| EDDM | 188.1 | -2.3 | 832 | 27,091 |
| LEMD | 179.8 | +16.8 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 356.9 | 153,660 |
| 2025-07 | 319.3 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 241.8 | 339,016 | 50.8% |
| False | 1898.4 | 5,323 | 49.2% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1203.2 | 2,093 | 7.8% |
| False | 324.2 | 342,246 | 92.2% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 196.5 | +117.0 | 34,434 |
| d1 | 569–671 | 176.1 | +88.7 | 34,434 |
| d2 | 671–770 | 214.7 | +66.9 | 34,434 |
| d3 | 770–842 | 166.4 | +60.2 | 34,434 |
| d4 | 842–923 | 189.9 | +45.0 | 34,434 |
| d5 | 923–1020 | 181.2 | +25.6 | 34,434 |
| d6 | 1020–1134 | 190.7 | +8.7 | 34,434 |
| d7 | 1134–1266 | 221.0 | -15.8 | 34,434 |
| d8 | 1266–1505 | 249.6 | -67.3 | 34,434 |
| d9 | 1505–88132 | 879.3 | -301.0 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 617 | 950 | 1454 | 90223 | 7 | 43 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 670,790 | 0 | 623 | 946 | 1013 | 1451 | 111354 | 7 | 63 |
