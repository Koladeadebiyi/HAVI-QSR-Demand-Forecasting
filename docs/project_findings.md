# Project Findings and Decision Log

All findings below are descriptive. They do not establish causation or prove forecasting improvement.

## A. Dataset inspection
| Measure | Result |
|---|---:|
| Rows | 1,369,500 |
| Columns | 23 |
| Date range | 2021-01-01 to 2025-12-31 |
| Unique dates | 1,826 |
| Restaurants | 15 |
| Menu items | 50 |
| Categories | 8 |
| Cities | 14 |
| States | 9 |
| Duplicate rows | 0 |
| Records per date | 750 |

Missing values: `quantity` 5,475 (0.40%); `precip_type` 904,350 (66.04%); `holiday_name` 1,322,250 (96.55%); `special_event_name` 1,322,250 (96.55%). Missing quantities by year: 2021 1,122; 2022 1,091; 2023 1,062; 2024 1,053; 2025 1,147. Each restaurant has 365 missing target values. This pattern warrants investigation but does not identify the missingness mechanism.

Full-file target summary: 1,364,025 observed quantities; mean approximately 25.38; median 21; standard deviation approximately 17.36; minimum 0; maximum 1,050. Observed zero is distinct from missing.

## B. EDA Exercise 1 — Overall demand
The analysis excluded 2025-10-01 onward and did not impute or delete missing targets.
- Quantity is strongly right-skewed, with a long upper tail and maximum 1,050.
- Average demand generally rises toward June/July and declines later in the year.
- Average quantity per record differs materially by restaurant.
- Soft Drink Regular led the displayed top-15 menu items by average quantity per record.
- Drinks led category average quantity per record; Specials were lowest.
- Daily average demand showed short-period fluctuations and larger annual waves.

Chart readings were approximate. No causal conclusions or model-performance claims were established.

## C. EDA Exercise 2 — Time-based demand

### Calculation basis
For weekday/weekend/month/quarter summaries, observed `quantity` was summed across restaurant-item records for each date, then daily totals were averaged across dates. Pandas sum skips missing values; these are observed totals and may be understated on dates with missing quantities. Observed/missing record counts were also retained.

Restaurant and category heatmaps show average quantity per record within each restaurant/category and weekday, not network-level daily totals.

### Weekday demand
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

### Weekday versus weekend
| Day type | Average daily observed sales |
|---|---:|
| Weekday | 18,032.84 |
| Weekend | 21,567.65 |

Weekend average daily observed sales are 19.61% higher. Median daily sales are also higher on weekends (20,982.5 versus 17,502). Statistical significance has not been tested.

### Balanced monthly comparison: 2021–2024
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

July is highest and January lowest. July's average is approximately 77.5% higher than January's.

### Balanced quarterly comparison: 2021–2024
| Quarter | Average daily observed sales |
|---|---:|
| Q1 | 14,883.09 |
| Q2 | 20,711.26 |
| Q3 | 22,892.27 |
| Q4 | 16,575.95 |

Q3 is strongest and Q1 weakest.

### Restaurant and category heatmaps
Saturday is the highest displayed weekday for all 15 restaurants and all eight categories. Restaurant and category demand levels differ. These heatmaps use average quantity per record.

### Available-period versus balanced-year comparison
Original available-period quarterly averages: Q1 15,029.74; Q2 20,913.71; Q3 23,104.60; Q4 16,575.95. Balanced 2021–2024 values are shown above. Q4 is identical because the original analysis already excluded Q4 2025. Q1–Q3 averages are slightly higher when 2025 is included, suggesting January–September 2025 observed demand was somewhat higher than the 2021–2024 average. This is descriptive, not proof of a lasting trend.

## D. Modelling implications
Candidate features: weekday number, weekend indicator, month, quarter, restaurant ID, menu item ID, category, and past-only lag/rolling statistics. Weather, holiday, event and promotion features require confirmation that future values are available and permitted. `day_of_week_num` and `is_weekend` overlap; test incremental value rather than assuming both help.

## E. Decisions
1. Keep the original combined CSV unchanged.
2. Do not fill missing quantities with zero.
3. Do not delete missing-target rows during EDA.
4. Treat all Q4 2025 as unseen.
5. Use time-based validation only.
6. Use wMAPE as primary metric, subject to official evaluator definition.
7. Distinguish average demand from total sales volume.
8. Use balanced 2021–2024 comparisons for month/quarter seasonality.

## F. Open questions
- Are weekday effects statistically robust across years?
- Do weekday effects differ by item within restaurant?
- Is there a trend beyond annual seasonality?
- What explains missing targets and missing precipitation types?
- Are future weather, promotion and event values known/allowed?
- Do lags and rolling features improve validation wMAPE?

## G. Status
Completed: data inspection, EDA 1, EDA 2 and balanced seasonal comparison. Pending: further EDA, executed validation, feature engineering, baselines, model training, model selection and final predictions. No model score has been established.
