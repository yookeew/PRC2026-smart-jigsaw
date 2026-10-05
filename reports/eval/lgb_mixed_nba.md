# Holdout eval — lgb_mixed_nba

- engine: **lgb**   |   generated: 2026-10-04 20:40
- target mixed  |  loss huber (alpha=800.0)  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=7997/direct=4040 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 327.5 | 141.5 | +2.8 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 651.5 | +36.2 | 1274 | 26,528 |
| LFPG | 578.2 | +0.5 | 1052 | 39,590 |
| EGLL | 282.0 | +3.0 | 1382 | 40,210 |
| LTFM | 258.0 | -6.8 | 1070 | 46,539 |
| LEBL | 202.4 | +0.2 | 957 | 28,985 |
| LSZH | 186.0 | -3.2 | 759 | 22,289 |
| EDDF | 179.2 | -1.8 | 875 | 36,830 |
| LEMD | 176.4 | +8.9 | 1011 | 35,328 |
| EHAM | 174.0 | -1.0 | 798 | 40,949 |
| EDDM | 154.9 | +1.5 | 832 | 27,091 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 347.2 | 153,660 |
| 2025-07 | 310.7 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 227.4 | 339,016 | 47.5% |
| False | 1908.8 | 5,323 | 52.5% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1447.8 | 2,448 | 13.9% |
| False | 304.9 | 341,891 | 86.1% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 184.3 | +88.6 | 34,434 |
| d1 | 569–671 | 167.9 | +69.2 | 34,434 |
| d2 | 671–770 | 165.7 | +52.2 | 34,434 |
| d3 | 770–842 | 169.4 | +53.7 | 34,434 |
| d4 | 842–923 | 184.5 | +42.6 | 34,434 |
| d5 | 923–1020 | 181.1 | +27.6 | 34,434 |
| d6 | 1020–1134 | 192.2 | +13.3 | 34,434 |
| d7 | 1134–1266 | 223.0 | -6.1 | 34,434 |
| d8 | 1266–1505 | 247.5 | -49.3 | 34,434 |
| d9 | 1505–88132 | 859.8 | -264.0 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 602 | 948 | 1474 | 91075 | 6 | 41 |
