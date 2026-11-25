# Methodology and Modelling Guardrails

## Forecasting unit
One row represents date × restaurant × menu item. Target: `quantity`. Forecast horizon: 2025-10-01 to 2025-12-31 (92 × 15 × 50 = 69,000 predictions).

## Leakage prevention
- Treat all Q4 2025 rows as forecast rows, regardless of whether `quantity` is populated.
- Do not use Q4 2025 quantities for training, imputation, target encoding, target-derived aggregates, lag/rolling features, feature selection or model selection.
- Calendar features may be derived from dates.
- Use weather, promotion and event values for the horizon only if genuinely known in advance and allowed by the competition.
- Fit learned preprocessing on training data only.
- Never use actual future sales as lag values in a multi-step forecast.

## Proposed first validation
Train: 2021-01-01 through 2024-09-30. Validate: 2024-10-01 through 2024-12-31. This is a proposal, not yet executed. Consider additional rolling-origin splits.

## Missing target values
The full file has 5,475 missing `quantity` values (0.40%). Do not replace them with zero or delete them during EDA. Exclude missing targets from supervised fitting and actual-versus-predicted scoring. Preserve observed zero sales as distinct from missing. Investigate missingness before modelling.

## Metric
Primary metric: `wMAPE = sum(abs(actual - predicted)) / sum(abs(actual)) * 100`. Use known actuals only for validation. Follow the official evaluator if it specifies a different implementation.

## Candidate features from EDA
Candidate signals include weekday number, weekend indicator, month, quarter, restaurant ID, menu item ID and category. Past-only lags and rolling statistics should also be investigated. Weather, holiday, event and promotion features require confirmation of future availability. These are candidates, not selected features; test incremental value using time-based validation.

## Aggregation distinction
Network-level weekday/month/quarter results use daily sums of observed quantities, averaged over dates. Restaurant and category heatmaps use average quantity per record within each restaurant/category and weekday. These measures have different denominators.

## Submission checks
Exactly 69,000 rows and four required columns; all 92 dates, 15 restaurants and 50 items; no duplicate keys; no missing/non-finite predictions; non-negative predictions; CSV reads successfully.


# Forecasting Methodology – HAVI QSR Demand Forecasting

## Evaluation Metric

The competition evaluates submissions using Weighted Mean Absolute Percentage Error (wMAPE), weighted by menu item volume.

The objective is to minimise the total absolute forecasting error relative to actual demand.

**Formula:**

\[
wMAPE = \frac{\sum |Actual - Predicted|}{\sum |Actual|}\times100
\]

A lower wMAPE indicates better forecasting accuracy.

## Modelling Implications

- Prioritise accurate forecasts for high-volume menu items.
- Evaluate performance across all 750 restaurant-item combinations.
- Use wMAPE as the primary metric for model comparison and selection.
- Track RMSE and MAE as supplementary diagnostic metrics.
- Avoid optimising only for average performance across individual combinations without considering their contribution to total menu item volume.

## Forecasting Constraint

October–December 2025 is the forecast period. Target values from this period must not be used for model training, lag construction, rolling statistics, or other target-derived features.

All validation must use chronological splits that reproduce the intended forecasting conditions.