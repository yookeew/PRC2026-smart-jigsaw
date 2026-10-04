# Holdout eval — lgb_mixed_nb

- engine: **lgb**   |   generated: 2026-10-04 18:08
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7996/direct=2896 (inner-valid 2025-06)  |  full-refit rounds flip=8796/direct=3186
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 334.3 | 153.2 | +3.3 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 658.8 | +34.4 | 1274 | 26,528 |
| LFPG | 578.4 | +0.0 | 1052 | 39,590 |
| EGLL | 285.1 | +5.4 | 1382 | 40,210 |
| LTFM | 257.7 | -7.5 | 1070 | 46,539 |
| LEBL | 221.8 | +1.6 | 957 | 28,985 |
| LSZH | 207.6 | -0.4 | 759 | 22,289 |
| EHAM | 199.1 | -0.4 | 798 | 40,949 |
| EDDF | 189.8 | +1.7 | 875 | 36,830 |
| EDDM | 184.4 | -1.8 | 832 | 27,091 |
| LEMD | 177.7 | +8.7 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 353.3 | 153,660 |
| 2025-07 | 318.2 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 236.5 | 339,016 | 49.3% |
| False | 1915.5 | 5,323 | 50.7% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1455.5 | 2,452 | 13.5% |
| False | 312.1 | 341,887 | 86.5% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.2 | +110.6 | 34,434 |
| d1 | 569–671 | 178.9 | +83.2 | 34,434 |
| d2 | 671–770 | 177.2 | +61.4 | 34,434 |
| d3 | 770–842 | 177.4 | +56.7 | 34,434 |
| d4 | 842–923 | 192.5 | +41.8 | 34,434 |
| d5 | 923–1020 | 190.8 | +24.4 | 34,434 |
| d6 | 1020–1134 | 201.9 | +9.0 | 34,434 |
| d7 | 1134–1266 | 229.9 | -12.3 | 34,434 |
| d8 | 1266–1505 | 255.3 | -60.5 | 34,434 |
| d9 | 1505–88132 | 866.8 | -281.7 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 612 | 947 | 1467 | 90726 | 6 | 42 |

## Submission file (ranking, Jan+Jul 2026)

| rows | min | p10 | median | mean | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 344,841 | 0 | 632 | 958 | 1037 | 1495 | 109966 | 3 | 42 |
