## Exercise 3 – Stage 3: Time-Series Behaviour

### Objective
Investigate the historical demand patterns of selected restaurant-item combinations to identify seasonality, trend, variability and differences across restaurant locations.

### Selected combinations
- R01 – Soft Drink Regular
- R13 – Brownie
- R12 – Oatmeal Cup
- R03 – Oatmeal Cup
- R01 – French Fries Regular

### Key findings

1. **Seasonality:** Soft Drink Regular and French Fries Regular exhibit recurring annual demand patterns, with higher demand around June and July and lower demand towards the end of the year.

2. **Demand variability:** Brownie at R13 demonstrates considerable relative variability, with frequent zero-sales observations and occasional demand spikes.

3. **Stable demand:** Oatmeal Cup at R12 exhibits relatively stable, low-volume demand, making it a useful candidate for simple baseline forecasting methods.

4. **Restaurant differences:** Oatmeal Cup at R03 has a substantially higher demand level than the same item at R12. This confirms the importance of retaining restaurant-level information.

5. **Long-term trend:** Comparable January–September averages indicate gradual growth in Soft Drink Regular and French Fries Regular demand between 2021 and 2025.

6. **Model implications:** Future forecasting models should consider seasonal calendar variables, lagged demand, rolling averages, restaurant identity, menu item identity, promotions and weather-related features.

### Important observations
- Missing target quantities were excluded from demand averages.
- Zero sales were retained as valid observations.
- October–December 2025 were excluded from historical analysis.
- 2025 year-over-year comparisons were restricted to January–September to ensure comparable periods.
- Extreme demand spikes were identified but not automatically removed.

### Conclusion
Demand behaviour varies significantly across menu items and restaurant locations. Some combinations exhibit strong seasonal patterns, while others are relatively stable or experience intermittent zero-sales observations.

The findings support the use of restaurant-item-level forecasting with carefully engineered temporal and contextual features. Multiple forecasting approaches should be evaluated using chronological validation and weighted Mean Absolute Percentage Error (wMAPE).