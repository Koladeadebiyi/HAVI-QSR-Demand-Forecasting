
# Feature Engineering – HAVI QSR Demand Forecasting

## 1. Overview

Feature engineering transforms the raw QSR demand dataset into structured variables that can be used by machine learning models to predict daily menu-item demand.

The objective is to capture historical demand behaviour, calendar patterns and relevant external factors while ensuring that no information from the forecast period enters model training.

The implementation is maintained in:

`src/feature_engineering.py`

## 2. Historical and Forecast Data Separation

The dataset was separated chronologically using October 1, 2025 as the forecast boundary.

### Historical dataset

- Start date: January 1, 2021
- End date: September 30, 2025
- Rows: 1,300,500

### Forecast dataset

- Start date: October 1, 2025
- End date: December 31, 2025
- Rows: 69,000

The forecast dataset contains 92 dates, 15 restaurants and 50 menu items.

The target variable, `quantity`, was removed from the forecast dataset to prevent accidental use of future demand observations.

## 3. Calendar Features

Calendar variables were extracted from the date column.

| Feature | Description |
|---|---|
| day_of_week | Day of week |
| day_name | Weekday name |
| day_of_week_num | Numeric weekday representation |
| is_weekend | Weekend indicator |
| month | Calendar month |
| quarter | Calendar quarter |
| day_of_month | Day within month |
| day_of_year | Day within year |
| week_of_year | Calendar week |

These features help the model capture recurring demand patterns.

## 4. Historical Rolling Features

Rolling statistics were created separately for each restaurant-item combination.

The features include:

| Feature | Description |
|---|---|
| mean_7d | Average quantity over the preceding 7 days |
| mean_28d | Average quantity over the preceding 28 days |
| mean_56d | Average quantity over the preceding 56 days |
| mean_90d | Average quantity over the preceding 90 days |
| mean_365d | Average quantity over the preceding 365 days |
| std_28d | Demand variability over the preceding 28 days |
| std_90d | Demand variability over the preceding 90 days |
| observed_7d | Number of observed quantities in the preceding 7 days |
| observed_28d | Number of observed quantities in the preceding 28 days |

### Missing target observations

Missing quantities are treated as unknown observations, not zero demand.

Rolling averages and standard deviations therefore use available observed quantities. Observation-count features help identify how many actual observations contributed to each statistic.

## 5. Preventing Target Leakage

A major consideration in time-series forecasting is ensuring that features do not contain information from the target date.

For historical training features, the quantity series is shifted forward by one calendar day at the index level before rolling calculations are performed.

This ensures that the feature values for a given date only use quantities available before that date.

For example:

| Target date | Target quantity | Previous 7-day mean |
|---|---:|---:|
| January 1 | 75 | NaN |
| January 2 | 49 | 75 |
| January 3 | 55 | 62 |

The January 1 target is not used to calculate its own rolling features.

The initial leakage check confirmed this behaviour after correcting the function call to use `calculate_training_features()`.

## 6. Forecast-Origin Reference Features

A separate reference feature table is created for the forecast period.

Forecast origin: September 30, 2025.

The reference table contains one record for each of the 750 restaurant-item combinations.

It includes the same rolling statistics, calculated using historical observations available at the forecast origin.

These reference features are merged into the forecast dataset using:

- restaurant_id
- menu_item_id

The merge uses a many-to-one validation to ensure that each forecast row receives the correct restaurant-item reference statistics.

### Forecast merge validation

| Check | Result |
|---|---:|
| Original forecast rows | 69,000 |
| Merged forecast rows | 69,000 |
| Unique combinations | 750 |
| Missing reference feature values | 0 |
| Forecast quantity column | Absent |

## 7. Important Forecasting Consideration

The forecast covers 92 days.

Forecast-origin rolling features are based on information available through September 30, 2025. They are static reference features when merged across the forecast horizon.

They do not automatically incorporate future actual quantities or update recursively with predicted quantities.

The final modelling strategy must therefore be validated using a historical holdout that reproduces the full 92-day forecasting problem.

Actual target quantities from the validation horizon must not be used to update forecast features unless the evaluation design explicitly allows those observations to become available.

## 8. Current Limitations and Next Steps

The following tasks remain before model training:

1. Complete missing-value checks across all model features.
2. Replace missing rolling observation counts with zero where appropriate.
3. Review categorical and numerical feature types.
4. Confirm that forecast-origin features are calculated independently of lagged training features.
5. Design a leakage-safe 92-day validation strategy.
6. Establish a baseline model and evaluate using the competition's official wMAPE definition.

No final model performance has been established at this stage.
