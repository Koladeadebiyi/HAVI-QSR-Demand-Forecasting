# EDA Exercise 5: Feature Availability Assessment

## 1. Objective

The objective of this exercise was to assess the availability, completeness, consistency, and suitability of explanatory variables within the October–December 2025 forecast dataset.

The assessment focused on determining:

- Which features are available throughout the forecast period.
- Whether missing values indicate genuine data-quality issues or expected conditional values.
- Whether features are consistent across restaurant-item records for the same date.
- Whether historical and forecast-period feature distributions provide useful information for modelling.
- Which features can be considered candidates for the forecasting model.

The forecast period covers October 1 to December 31, 2025.

## 2. Dataset Overview

| Metric | Result |
|---|---:|
| Total dataset records | 1,369,500 |
| Total columns | 23 |
| Historical and forecast date range | Jan 1, 2021 – Dec 31, 2025 |
| Forecast records | 69,000 |
| Forecast dates | 92 |
| Restaurants | 15 |
| Menu items | 50 |
| Restaurant-item combinations | 750 |

The forecast dataset contains the expected 69,000 records:

92 dates × 15 restaurants × 50 menu items.

This confirms the expected forecast-period dataset structure.

## 3. Forecast Feature Availability

The following table summarises missing values and unique values for selected features in the forecast dataset.

| Feature | Missing Values | Missing % | Unique Values |
|---|---:|---:|---:|
| unit_price | 0 | 0.00% | 25 |
| avg_temp_f | 0 | 0.00% | 408 |
| precip_inches | 0 | 0.00% | 84 |
| precip_type | 47,200 | 68.41% | 2 |
| is_holiday | 0 | 0.00% | 2 |
| holiday_name | 66,000 | 95.65% | 4 |
| is_special_event | 0 | 0.00% | 2 |
| special_event_name | 67,400 | 97.68% | 12 |
| is_promotion | 0 | 0.00% | 2 |

### Observations

Most selected explanatory variables are complete in the forecast dataset.

The high missing percentages for `precip_type`, `holiday_name`, and `special_event_name` require interpretation rather than immediate imputation.

Subsequent consistency checks established that these missing values are associated with the absence of the corresponding weather condition or calendar event.

## 4. Calendar and Event Feature Distribution

### 4.1 Holiday Distribution

| Holiday Flag | Records |
|---|---:|
| No holiday | 66,000 |
| Holiday | 3,000 |

The holiday indicator is complete. The `holiday_name` field is populated only when `is_holiday` equals 1.

The consistency check showed that all 66,000 non-holiday records had no holiday name, while all 3,000 holiday records had a name.

### 4.2 Special Event Distribution

| Special Event Flag | Records |
|---|---:|
| No special event | 67,400 |
| Special event | 1,600 |

The `special_event_name` field is populated only when `is_special_event` equals 1.

All 67,400 non-event records had no event name, while all 1,600 event records had a name.

### Interpretation

The missing holiday and event names represent conditional missingness. They do not indicate that the corresponding binary indicators are incomplete.

Both binary indicators can be retained as candidate model features. The categorical names can be evaluated separately to determine whether they provide additional predictive value.

## 5. Weather Feature Assessment

### 5.1 Precipitation Consistency

The relationship between precipitation amount and precipitation type was examined.

| Precipitation Amount | Precipitation Type | Records |
|---|---|---:|
| Zero | Missing | 21,800 |
| Greater than zero | Present | 47,200 |

There was an exact correspondence between precipitation amount and precipitation type.

- Records with zero precipitation had missing precipitation types.
- Records with positive precipitation amounts had a precipitation type.

### Interpretation

The missing `precip_type` values appear to represent days without precipitation rather than unknown weather conditions.

A binary feature, `has_precipitation`, can therefore be derived from `precip_inches`.

The continuous precipitation amount should also be retained as a candidate feature. The additional predictive contribution of precipitation type can be evaluated during modelling.

### 5.2 Weather Feature Consistency

Weather variables were checked for consistency across records belonging to the same restaurant and date.

| Feature | Inconsistent Restaurant-Days |
|---|---:|
| avg_temp_f | 0 |
| precip_inches | 0 |
| precip_type | 0 |

The weather values were consistent across all restaurant-item records for each restaurant-day.

This confirms that the weather variables are recorded consistently at the restaurant-day level.

**Forecasting consideration:** Although weather variables are present for all 92 forecast dates, their presence in the dataset does not establish whether actual observed weather is permitted as a forecast input. Their eligibility must be checked against the competition's forecasting rules and intended information availability.

## 6. Unit Price Stability

Unit price was examined across all 750 restaurant-item combinations in both the forecast period and historical dataset.

| Metric | Result |
|---|---:|
| Restaurant-item combinations | 750 |
| Combinations with multiple prices | 0 |
| Maximum distinct prices per combination | 1 |

No price changes were detected across the historical dataset or forecast period.

### Interpretation

`unit_price` is stable within each restaurant-item combination in the supplied data.

This makes it a potential explanatory feature for distinguishing demand patterns across products and locations.

However, its predictive usefulness and availability at the forecast origin should still be considered during model development.

## 7. Promotion Assessment

### 7.1 Historical Promotion Frequency by Year

| Year | Total Records | Promotional Records | Promotion Rate |
|---|---:|---:|---:|
| 2021 | 273,750 | 2,905 | 1.06% |
| 2022 | 273,750 | 3,108 | 1.14% |
| 2023 | 273,750 | 3,325 | 1.21% |
| 2024 | 274,500 | 3,213 | 1.17% |
| 2025 | 273,750 | 3,031 | 1.11% |

Promotion frequency remained relatively consistent across the five years, ranging from approximately 1.06% to 1.21%.

2023 recorded the highest promotion rate at 1.21%, while 2021 recorded the lowest at 1.06%.

There is no indication of a major structural change in promotion frequency based on these annual figures.

### 7.2 Promotion Date Coverage

| Metric | Result |
|---|---:|
| Total dates | 1,826 |
| Dates with promotions | 420 |
| Dates without promotions | 1,406 |
| Percentage of dates with promotions | 23.00% |
| Maximum promotional records on a date | 48 |

Although promotions appeared on approximately 23% of calendar dates, they accounted for a relatively small proportion of restaurant-item records.

### 7.3 Forecast Period Promotion Distribution

| Metric | Result |
|---|---:|
| Total forecast records | 69,000 |
| Promotional records | 672 |
| Non-promotional records | 68,328 |
| Promotion rate | 0.97% |
| Average promotional records per date | 7.30 |
| Median promotional records per date | 0 |
| Maximum promotional records per date | 35 |

Promotions are particularly sparse during the forecast period.

### Interpretation

The historical promotion rate is relatively stable, but promotional records account for a small proportion of the overall dataset.

Previous EDA analysis also found a positive average promotional uplift across combinations with both promotional and non-promotional observations. However, promotion exposure was sparse and varied across combinations.

Consequently:

- Retain `is_promotion` as a candidate feature.
- Avoid applying a blanket promotional uplift to forecasts.
- Evaluate its predictive contribution during model validation.
- Investigate promotion concentration by restaurant and menu item before drawing conclusions about its distribution.

## 8. Feature Consistency Assessment

The following features were checked for consistency across records belonging to the same restaurant and date.

| Feature | Inconsistent Restaurant-Days |
|---|---:|
| avg_temp_f | 0 |
| precip_inches | 0 |
| precip_type | 0 |
| is_holiday | 0 |
| holiday_name | 0 |
| is_special_event | 0 |
| special_event_name | 0 |
| is_weekend | 0 |

All checked features were consistent within restaurant-day groups.

This indicates that the same date-level contextual information is repeated consistently across menu items within each restaurant.

The `is_weekend` indicator can also be derived directly from the date.

## 9. Preliminary Feature Selection Recommendations

Based on the assessment, the following features are candidates for the forecasting model.

| Feature | Recommendation | Reason |
|---|---|---|
| unit_price | Retain and evaluate | Complete and stable |
| avg_temp_f | Conditional | Complete, but forecast-time availability must be verified |
| precip_inches | Conditional | Complete, but forecast-time availability must be verified |
| precip_type | Evaluate | Conditional missingness; may provide additional weather information |
| has_precipitation | Create | Derived from precipitation amount |
| is_holiday | Retain | Complete binary indicator |
| holiday_name | Evaluate | Provides categorical event information |
| is_special_event | Retain | Complete binary indicator |
| special_event_name | Evaluate | Provides categorical event information |
| is_weekend | Derive | Deterministic calendar feature |
| is_promotion | Retain and evaluate | Complete but sparse |

These are preliminary recommendations. Final feature selection will depend on chronological validation performance and the availability of each variable at the intended forecast origin.

## 10. Key Findings and Modelling Implications

1. The forecast dataset contains the expected 69,000 records across 92 dates, 15 restaurants, and 50 menu items.
2. Most selected explanatory variables are complete.
3. Missing precipitation types correspond exactly to zero precipitation.
4. Holiday and special-event names are populated conditionally when their respective flags are active.
5. Weather and calendar variables are consistent across restaurant-item records for the same restaurant-day.
6. Unit prices are stable across all 750 restaurant-item combinations in the supplied historical and forecast data.
7. Promotion activity is relatively stable historically but sparse at the individual record level.
8. The forecast-period weather variables require an explicit forecast-time availability check before use.

## 11. Conclusion

The feature availability assessment establishes that the supplied forecast dataset contains several potentially useful explanatory variables, with no major structural inconsistencies identified in the checks performed.

Calendar indicators, unit price, promotions, and event information are promising candidates for feature engineering. Weather information may also be useful, subject to confirmation that the forecast-period values are eligible inputs.

The next stage is feature engineering, where calendar, historical demand, lag, rolling, and eligible external features will be constructed using strict chronological controls to prevent target leakage.

All final feature decisions will be assessed using the competition's primary evaluation metric, wMAPE.