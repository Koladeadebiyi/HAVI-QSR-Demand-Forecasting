# EDA Exercise 1 — Overall Demand Analysis

## Objective
Understand broad demand characteristics across the QSR network before engineering features or selecting models.

## Analysis population
Rows dated before 2025-10-01. The competition forecast period is excluded. Missing `quantity` values are not filled or deleted.

## Findings
- Quantity is strongly right-skewed, with a long upper tail and full-file maximum of 1,050.
- Average demand generally rises toward June/July and declines later in the year.
- Average quantity per record differs materially by restaurant.
- Soft Drink Regular led the displayed top-15 items by average quantity per record.
- Drinks led category average quantity per record; Specials were lowest.
- Daily average demand showed recurring short-period fluctuations and larger annual waves.

## Forecasting implications
Investigate weekday, month, restaurant and menu-item features, plus past-only lags and rolling statistics. Investigate extreme values rather than automatically removing them.

## Limitations
Chart values were approximate visual readings. Averages do not show total contribution to sales volume. Aggregated analysis can hide item-level and restaurant-level patterns. No causal conclusions or model-performance claims are established.
