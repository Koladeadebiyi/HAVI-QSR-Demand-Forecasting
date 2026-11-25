# EDA Exercise 2 — Time-Based Demand Analysis

## Objective
Quantify weekday, weekend, monthly, quarterly, restaurant and category demand patterns relevant to daily forecasting.

## Aggregation
For weekday/weekend/month/quarter summaries, sum observed quantities across all restaurant-item records for each date, then average daily totals across dates. Missing quantities were not imputed; Pandas sum skips missing values, so daily totals may be understated. Restaurant and category heatmaps use average quantity per record within each entity and weekday.

## Weekday results
| Day | Average daily observed sales |
|---|---:|
| Monday | 17,380.73 |
| Tuesday | 16,633.18 |
| Wednesday | 17,361.08 |
| Thursday | 17,941.99 |
| Friday | 20,844.15 |
| Saturday | 22,689.95 |
| Sunday | 20,445.35 |

Saturday is highest; Tuesday is lowest.

## Weekday versus weekend
Weekday average: 18,032.84. Weekend average: 21,567.65. Weekend average daily observed sales are 19.61% higher. Median daily sales are 20,982.5 on weekends versus 17,502 on weekdays. Statistical significance has not been tested.

## Balanced monthly seasonality: 2021–2024
| Month | Average daily observed sales |
|---|---:|
| January | 13,944.15 |
| February | 14,575.88 |
| March | 16,101.98 |
| April | 17,717.06 |
| May | 20,826.45 |
| June | 23,586.44 |
| July | 24,754.70 |
| August | 23,009.13 |
| September | 20,847.00 |
| October | 18,186.37 |
| November | 16,458.36 |
| December | 15,079.33 |

July is highest and January lowest; July is approximately 77.5% higher than January.

## Balanced quarterly seasonality: 2021–2024
| Quarter | Average daily observed sales |
|---|---:|
| Q1 | 14,883.09 |
| Q2 | 20,711.26 |
| Q3 | 22,892.27 |
| Q4 | 16,575.95 |

Q3 is strongest and Q1 weakest.

## Entity weekday patterns
The supplied heatmaps show Saturday as the highest displayed weekday for all 15 restaurants and all eight categories. Demand levels differ by restaurant and category. Heatmap values are per-record averages, not network-level daily totals.

## Interpretation and forecasting implications
The results suggest weekly and annual demand structure and differing entity demand scales. Candidate features include weekday number, weekend indicator, month, quarter, restaurant ID, menu item ID and category. Past-only lags and rolling statistics should be investigated. Test all candidate features with time-based validation; the weekend flag overlaps with weekday number.

## Limitations
- Missing target quantities can understate daily observed totals.
- The original monthly summary mixed five years for January–September with four years for October–December. The 2021–2024 comparison provides a balanced seasonal view.
- Heatmap averages and network daily totals have different denominators.
- No significance tests, causal analysis or forecasting model were performed.

## Conclusion
Saturday is the strongest weekday, July the strongest month and Q3 the strongest quarter in these descriptive summaries. These are candidate modelling signals, not proven accuracy improvements.
