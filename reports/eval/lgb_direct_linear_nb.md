# Holdout eval — lgb_direct_linear_nb

- engine: **lgb**   |   generated: 2026-10-06 22:28
- target direct  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at direct=1074 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 386.8 | 153.7 | +3.0 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 957.4 | +16.0 | 1274 | 26,528 |
| LFPG | 577.8 | +2.5 | 1052 | 39,590 |
| EGLL | 278.3 | +4.5 | 1382 | 40,210 |
| LTFM | 263.1 | -4.9 | 1070 | 46,539 |
| LEBL | 233.0 | +2.7 | 957 | 28,985 |
| LSZH | 207.5 | +0.2 | 759 | 22,289 |
| EHAM | 202.5 | +2.2 | 798 | 40,949 |
| EDDF | 194.0 | +3.0 | 875 | 36,830 |
| EDDM | 185.1 | -1.5 | 832 | 27,091 |
| LEMD | 178.6 | +9.2 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 361.9 | 153,660 |
| 2025-07 | 405.8 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 257.8 | 339,016 | 43.7% |
| False | 2334.1 | 5,323 | 56.3% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 2007.2 | 2,452 | 19.2% |
| False | 349.0 | 341,887 | 80.8% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 198.6 | +113.4 | 34,434 |
| d1 | 569–671 | 175.2 | +85.2 | 34,434 |
| d2 | 671–770 | 179.5 | +63.0 | 34,434 |
| d3 | 770–842 | 174.8 | +57.5 | 34,434 |
| d4 | 842–923 | 188.7 | +41.9 | 34,434 |
| d5 | 923–1020 | 189.4 | +24.7 | 34,434 |
| d6 | 1020–1134 | 199.6 | +8.4 | 34,434 |
| d7 | 1134–1266 | 226.5 | -12.6 | 34,434 |
| d8 | 1266–1505 | 319.6 | -59.3 | 34,434 |
| d9 | 1505–88132 | 1048.2 | -291.9 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 616 | 947 | 1466 | 68666 | 1 | 41 |
