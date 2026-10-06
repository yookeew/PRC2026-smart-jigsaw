# Holdout eval — lgb_mixed_nb_clipl2

- engine: **lgb**   |   generated: 2026-10-06 14:32
- target mixed  |  loss l2  |  eta 0.02  |  rounds ceiling 8000  |  early-stopped at flip=1049/direct=1516 (inner-valid 2025-06)  |  full-refit rounds ?
- holdout = 2025-01, 2025-07 (fit on the other 10 months of 2025)
- rows scored: 344,339 (all true taxi >= 0; no upper-bound exclusion -- the real board scores these rows too, see 'By AOBT_3_flt lane' below)
- target d clipped to [30, 7200] s in training; taxi predictions clipped to [0, 10800] s

## Overall

| RMSE (s) | MAE (s) | mean error (pred−true, s) |
| --- | --- | --- |
| 342.5 | 154.9 | +6.6 |

## Per airport

| airport | rmse | bias | mean taxi | n |
| --- | --- | --- | --- | --- |
| LIRF | 712.8 | +47.2 | 1274 | 26,528 |
| LFPG | 576.5 | +6.3 | 1052 | 39,590 |
| EGLL | 283.6 | +8.0 | 1382 | 40,210 |
| LTFM | 258.5 | -5.4 | 1070 | 46,539 |
| LEBL | 222.4 | +4.6 | 957 | 28,985 |
| LSZH | 208.0 | -0.3 | 759 | 22,289 |
| EHAM | 199.3 | +2.2 | 798 | 40,949 |
| EDDF | 189.6 | +3.7 | 875 | 36,830 |
| EDDM | 185.9 | +0.3 | 832 | 27,091 |
| LEMD | 178.0 | +9.1 | 1011 | 35,328 |

## Per month (both must improve, not just the pooled number)

| month | rmse | n |
| --- | --- | --- |
| 2025-01 | 355.3 | 153,660 |
| 2025-07 | 331.7 | 190,679 |

## By AOBT_3_flt lane

Rows missing an NM off-block time are a small, high-leverage population (echoes + schedule-default poison labels) that behaves very differently from the rest -- see PROGRESS.md.

| has_aobt3 | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 236.8 | 339,016 | 47.1% |
| False | 2004.2 | 5,323 | 52.9% |

## By echo classifier verdict (P(echo) > 0.5)

Supersedes the has_aobt3 lane above as the primary echo-lane diagnostic -- see fit_echo_classifier / PROGRESS.md.

| echo_pred | rmse | n | % of total squared error |
| --- | --- | --- | --- |
| True | 1604.0 | 2,452 | 15.6% |
| False | 315.7 | 341,887 | 84.4% |

## Per true-taxi decile

| decile | taxi range (s) | rmse | bias | n |
| --- | --- | --- | --- | --- |
| d0 | 0–569 | 203.4 | +114.3 | 34,434 |
| d1 | 569–671 | 190.5 | +87.2 | 34,434 |
| d2 | 671–770 | 189.0 | +65.1 | 34,434 |
| d3 | 770–842 | 189.0 | +60.5 | 34,434 |
| d4 | 842–923 | 196.0 | +45.3 | 34,434 |
| d5 | 923–1020 | 203.0 | +27.5 | 34,434 |
| d6 | 1020–1134 | 214.5 | +12.5 | 34,434 |
| d7 | 1134–1266 | 238.2 | -9.6 | 34,434 |
| d8 | 1266–1505 | 264.2 | -58.4 | 34,434 |
| d9 | 1505–88132 | 878.4 | -278.8 | 34,433 |

## Predicted taxi distribution (holdout)

| min | p10 | p50 | p90 | max | n@floor | n@ceil |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 615 | 950 | 1467 | 90411 | 5 | 43 |
