# EDA Exercise 4 – External Factors and Demand Relationships

## 1. Objective

Investigate the relationship between external factors and restaurant-item demand to identify potentially useful predictors for the demand forecasting model.

The analysis covers January 2021 to September 2025. October–December 2025 was excluded to preserve the competition forecast period.

## 2. Variables Investigated

The following variables were examined:

- Temperature (`avg_temp_f`)
- Precipitation (`precip_inches`)
- Holidays (`is_holiday`)
- Special events (`is_special_event`)
- Promotions (`is_promotion`)
- Weekend patterns (`is_weekend`)

## 3. Methodology and Data Preparation

The analysis followed these steps:

1. Weather, holiday and event variables were aggregated to restaurant-day level to prevent repeated menu-item observations from being treated as independent external-factor measurements.
2. Temperature and demand were centred around restaurant-specific averages to reduce differences in baseline demand and temperature between restaurants.
3. Demand distributions were compared between dry and precipitation days.
4. Holiday and special-event demand were compared against ordinary days.
5. Promotional and non-promotional average quantities were compared within restaurant-item combinations.
6. Weekend demand was examined alongside external conditions.

### Data Preparation and Validation

The original dataset was restricted to January 2021–September 2025.

Missing target quantities were preserved as missing, while observed zero-sales records were retained.

Precipitation amount was used to distinguish dry days from days with precipitation. Missing precipitation values were not classified as dry days.

The dataset was aggregated from restaurant-menu-item observations to one record per restaurant per date.

**Aggregation validation results:**

- Resulting dataset: 26,010 rows and 14 columns.
- Restaurants: 15.
- Analysis period: 1,734 days.
- Zero inconsistencies were identified across temperature, precipitation, holiday, special event, weekend, month and year variables within restaurant-day groups.
- Zero restaurant-days had missing precipitation values.

The resulting observation count was consistent with the expected structure:

1,734 days × 15 restaurants = 26,010 restaurant-day records.

These checks support the use of the `first` aggregation method for contextual variables because their values were consistent across menu-item records within each restaurant-day.

### Analytical Safeguards

- Repeated restaurant-day weather observations across menu items were not treated as independent weather measurements.
- Correlations and promotion differences were interpreted as descriptive associations, not proof of causality.
- Observation counts were retained alongside comparisons, particularly for less frequent events and promotions.
- Missing target quantities were not replaced with zero.

### Expected Outputs

**Charts**
- Temperature versus demand.
- Precipitation versus demand.
- Holiday demand comparison.
- Special-event demand comparison.
- Promotion uplift distribution.

**Supporting datasets**
- Restaurant-day external factors dataset.
- Precipitation summary.
- Holiday summary.
- Special-event summary.
- Promotion comparison.
- Weekend external factors.
- Holiday-event interaction summary.

## 4. Key Findings

### 4.1 Temperature and Demand

The temperature-demand analysis was conducted using within-restaurant centred temperature and demand values.

**Correlation results:**

| Metric | Correlation |
|---|---:|
| Pearson | 0.7207 |
| Spearman | 0.7949 |

Both results indicate a strong positive association between temperature deviations and demand deviations within restaurants.

This suggests that days with temperatures above a restaurant's average temperature tend to have higher-than-average demand.

The stronger Spearman correlation indicates a consistent positive ranking relationship, even where the relationship may not be perfectly linear.

**Interpretation and modelling implication:**

Temperature demonstrates a strong historical relationship with demand and will be retained as a candidate forecasting feature.

However, seasonality and other time-related factors may contribute to the observed relationship. The correlation does not establish that temperature independently causes demand to increase.

### 4.2 Precipitation and Demand

Demand was compared between dry days and days with recorded precipitation.

| Metric | Dry days | Precipitation days |
|---|---:|---:|
| Average quantity | 25.98 | 24.55 |
| Median quantity | 24.82 | 23.28 |
| Restaurant-days | 17,143 | 8,867 |

Restaurant-days with precipitation recorded approximately 5.5% lower average demand than dry restaurant-days.

The median demand was also lower on precipitation days.

**Interpretation and modelling implication:**

Precipitation demonstrates a negative historical association with average demand.

The `precip_inches` variable will be retained as a candidate forecasting feature. Its predictive contribution will be assessed during chronological validation.

### 4.3 Holidays and Demand

Average demand was compared between holiday and non-holiday restaurant-days.

| Metric | Non-holiday | Holiday |
|---|---:|---:|
| Average quantity | 25.42 | 27.46 |
| Median quantity | 24.34 | 24.50 |
| Restaurant-days | 25,125 | 885 |

Average demand was approximately 8.0% higher on holidays than on non-holidays.

However, the median values were relatively close, suggesting that some higher-demand holiday observations may contribute to the difference in average demand.

**Interpretation and modelling implication:**

Holidays demonstrate a modest positive association with average demand.

The `is_holiday` variable will be retained as a candidate forecasting feature, with its contribution evaluated alongside weekday, weekend and other calendar effects.

### 4.4 Special Events and Demand

Demand was compared between restaurant-days with and without special events.

| Metric | No special event | Special event |
|---|---:|---:|
| Average quantity | 24.95 | 40.44 |
| Median quantity | 23.98 | 38.10 |
| Restaurant-days | 25,097 | 913 |

Special-event restaurant-days recorded approximately 62.1% higher average demand than days without special events.

The median demand was also substantially higher on special-event days.

**Interpretation and modelling implication:**

Special events demonstrate a strong positive historical association with restaurant demand.

The `is_special_event` variable will be retained as a candidate forecasting feature.

Further investigation may examine differences across restaurants, event types and menu categories.

### 4.5 Holiday and Special-Event Interaction

The analysis also examined restaurant-days where holidays and special events occurred independently or together.

| Holiday | Special event | Average quantity | Restaurant-days |
|---|---|---:|---:|
| No | No | 24.93 | 24,277 |
| No | Yes | 39.54 | 848 |
| Yes | No | 25.50 | 820 |
| Yes | Yes | 52.17 | 65 |

The highest average demand occurred when holidays and special events coincided.

Special-event days also recorded higher average demand when they did not coincide with holidays.

However, the combined holiday and special-event group contained only 65 restaurant-days.

**Interpretation and modelling implication:**

The results suggest that the relationship between special events and demand may differ depending on whether an event coincides with a holiday.

A holiday-event interaction feature may be worth testing during model development. The limited number of combined observations means its predictive contribution must be validated carefully.

### 4.6 Promotion and Demand Relationship

The analysis compared average observed demand across promotion and non-promotion days for each restaurant-menu item combination.

**Key findings:**

- 714 out of 750 restaurant-menu item combinations had observations under both promotion and non-promotion conditions.
- 657 combinations (92.0%) recorded positive promotional uplift, while 57 (8.0%) recorded negative uplift.
- Median promotional uplift was 23.10%, while mean uplift was 24.25%.
- Promotional observations were considerably fewer than non-promotional observations.
- The median was 21 promotional observations compared with 1,708 non-promotional observations per combination.
- 130 combinations had fewer than 10 promotional observations, while 305 had fewer than 20.
- No comparable combination had zero baseline demand.

**Promotion Uplift Distribution**

| Metric | Result |
|---|---:|
| Mean uplift | 24.25% |
| Median uplift | 23.10% |
| Standard deviation | 20.83% |
| Minimum uplift | -58.54% |
| Maximum uplift | 173.70% |
| Positive uplift combinations | 657 |
| Negative uplift combinations | 57 |

Approximately 92% of comparable combinations recorded higher average demand during promotional observations.

**Interpretation and modelling implication:**

Promotional activity is associated with higher average observed demand for most restaurant-menu item combinations.

However, the limited number of promotional observations for many combinations means the estimated uplift should be interpreted cautiously.

The `is_promotion` variable will be retained as a candidate model feature. Its predictive contribution will be assessed through chronological validation rather than applying a fixed promotional uplift to forecasts.

Promotion-related demand differences should not be interpreted as causal effects.

### 4.7 Weekday versus Weekend Demand

Demand was compared between weekdays and weekends alongside average weather conditions.

| Metric | Weekday | Weekend |
|---|---:|---:|
| Average quantity | 24.14 | 28.87 |
| Average temperature | 52.05°F | 52.02°F |
| Average precipitation | 0.0974 inches | 0.0945 inches |
| Holiday restaurant-days | 765 | 120 |
| Event restaurant-days | 649 | 264 |
| Total restaurant-days | 18,570 | 7,440 |

Weekend demand was approximately 19.6% higher than weekday demand.

Average temperature and precipitation were very similar across both groups.

The proportion of restaurant-days with special events was also similar between weekdays and weekends.

**Interpretation and modelling implication:**

Weekend demand demonstrates a recurring historical pattern that is not simply explained by differences in average temperature or precipitation.

The `is_weekend` variable will be retained as a candidate forecasting feature.

## 5. Preliminary Forecasting Implications

The exploratory results suggest retaining the following variables for further evaluation:

- `avg_temp_f`
- `precip_inches`
- `is_holiday`
- `is_special_event`
- `is_promotion`
- `is_weekend`

A potential holiday-event interaction feature may also be evaluated.

Temperature, special events, promotions and weekend indicators demonstrate particularly interesting historical relationships with demand.

These priorities remain provisional. Their predictive value must be assessed through chronological validation.

All external variables must also be available at the time forecasts are generated. Historical observed weather, for example, must not be substituted for future weather information that would not be available at prediction time.

## 6. Limitations and Further Investigation

- Correlation does not establish causation.
- Temperature and demand may share seasonal patterns.
- Special-event and holiday effects may overlap with calendar effects.
- Promotion averages are based on unequal observation counts.
- Extreme demand observations may influence average comparisons.
- Weather relationships may differ by restaurant and menu category.
- Rare events may provide insufficient observations for reliable estimation.
- Future availability of external variables must be established before feature engineering.
- No feature has yet been confirmed to improve out-of-sample forecasting performance.

## 7. Conclusion

EDA Exercise 4 demonstrates that external factors are associated with meaningful differences in restaurant demand.

Temperature, special events and weekends show particularly strong historical relationships with demand. Promotions also show positive associations across most comparable restaurant-menu item combinations, although promotional observations are relatively limited.

Precipitation and holidays provide additional potential forecasting information.

These findings provide a basis for feature engineering and subsequent model evaluation.

No forecasting feature has yet been confirmed as predictive through out-of-sample validation.