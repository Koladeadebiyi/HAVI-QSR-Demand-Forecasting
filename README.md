# HAVI QSR Demand Forecasting

## 1. Project Overview

This project develops a daily demand forecasting solution for 50 menu
items across 15 quick-service restaurants (QSRs), using historical
sales, calendar information, promotions, and other business variables.

The forecasting target was **1 October--31 December 2025**, covering a
92-day horizon and 69,000 restaurant-item-date combinations.

The project follows a leakage-aware modelling workflow, evaluates
candidate models using weighted Mean Absolute Percentage Error (wMAPE),
selects a preferred model, generates Q4 forecasts, and evaluates those
forecasts against the actual quantities available in the source dataset.

## 2. Business Problem

Forecast daily quantities for 50 menu items across 15 QSR restaurants
over a 92-day horizon.

The forecast grid contains:

-   69,000 prediction records (92 days × 15 restaurants × 50 menu
    items).
-   Daily restaurant-item predictions.
-   Non-negative predicted quantities.

The competition metric is wMAPE; lower values indicate better
forecasting performance.

\[ wMAPE =
`\frac{\sum |Actual - Predicted|}{\sum |Actual|}`{=tex}`\times100`{=tex}
\]

Only observations with known actual quantities are included when
calculating evaluation metrics.

## 3. Dataset Overview

  Attribute                          Description
  ------------------- --------------------------
  Total records                        1,369,500
  Total columns                               23
  Historical period     2021-01-01 to 2025-09-30
  Forecast period       2025-10-01 to 2025-12-31
  Unique dates                             1,826
  Restaurants                                 15
  Menu items                                  50
  Categories                                   8
  Cities                                      14
  States                                       9
  Records per date                           750

The dataset contains 750 restaurant-item records per date. The original
dataset is maintained locally and is not included in this documentation
bundle.

## 4. Modelling and Data Guardrails

The project applied the following principles:

-   Q4 target quantities were not used to train the models or construct
    forecast features.
-   Validation followed chronological ordering rather than random
    train-test splitting.
-   Missing target quantities were not automatically treated as zero
    sales.
-   Historical features used only information available before the
    relevant target date or forecast origin.
-   Fixed-origin training examples were constructed to simulate a
    multi-day forecast made from information available at a single
    origin.
-   Forecast features were calculated using information available
    through 30 September 2025.
-   All 69,000 required forecast combinations were retained in the
    generated forecast.
-   Q4 actual quantities were used only for post-forecast evaluation,
    not for model training or forecast feature construction.

## 5. Exploratory Data Analysis

Five major areas of exploratory analysis were completed.

### 5.1 Overall and time-based demand

The analysis examined demand distributions, restaurant and product
contributions, weekday and weekend patterns, and monthly and quarterly
seasonality.

Key findings:

-   Weekend average daily demand was approximately 19.6% higher than
    weekday demand.
-   Q3 recorded the strongest average daily demand; Q1 was the weakest.
-   Demand generally increased from January towards a July peak before
    declining.
-   Soft Drink Regular featured prominently among the highest-demand
    items.

### 5.2 Restaurant and menu-item behaviour

Demand varied considerably across restaurant-item combinations. Some
low-volume products showed greater relative variability, and zero-sales
behaviour differed across products and restaurants.

### 5.3 External factors

Exploratory analysis identified:

-   A strong positive association between temperature and demand.
-   Higher average demand on special-event days than on non-event days.
-   Positive observed promotion uplift in 657 of 714 comparable
    combinations.

These are exploratory associations and should not be interpreted as
proof of causation. Weather variables were excluded from the initial
model feature set because their availability at forecast time required
consideration.

### 5.4 Feature availability

The forecast dataset contained 69,000 records. Some descriptive fields,
including precipitation type, holiday names, and special-event names,
had substantial conditional missingness. The model therefore prioritised
features with reliable forecast-time availability.

Detailed EDA findings are maintained in the `docs/` directory.

## 6. Feature Engineering

The initial feature-engineering implementation is in
`src/feature_engineering.py`. The workflow created calendar, business,
and historical demand features.

### 6.1 Rolling demand features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `mean_7d`                           Seven-day rolling mean

  `mean_28d`                          28-day rolling mean

  `mean_56d`                          56-day rolling mean

  `mean_90d`                          90-day rolling mean

  `mean_365d`                         365-day rolling mean

  `std_28d`                           28-day rolling standard deviation

  `std_90d`                           90-day rolling standard deviation

  `observed_7d`                       Number of observed quantities in
                                      the seven-day window

  `observed_28d`                      Number of observed quantities in
                                      the 28-day window
  -----------------------------------------------------------------------

Other features included restaurant and menu-item identifiers, category,
unit price, calendar attributes, and promotion, holiday, and
special-event indicators.

### 6.2 Fixed-origin training

An important modelling improvement was made after the first model
iteration.

CatBoost v1 used rolling features that were updated daily during
training, while the validation and Q4 forecast features represented a
fixed forecast origin. This created a mismatch between the training
setup and the multi-day forecasting task.

For CatBoost v2, historical training examples were reconstructed at
multiple forecast origins. Each example used only information available
at its origin, with labels drawn from the subsequent forecast window.
This more closely simulated the fixed-origin forecasting scenario.

The resulting forecast-origin training dataset contained:

-   961,468 training rows.
-   750 restaurant-item combinations.
-   14 historical forecast origins.
-   Forecast horizons extending up to 92 days.
-   No missing target values or selected rolling-demand features.

The July--September 2025 validation period remained separate from this
training dataset.

## 7. Validation Strategy

The primary model-development validation period was 1 July--30 September
2025.

  Dataset                Period
  ---------------------- --------------------------
  CatBoost v1 training   2021-01-01 to 2025-06-30
  Validation             2025-07-01 to 2025-09-30
  Q4 forecast            2025-10-01 to 2025-12-31

There were 68,721 validation observations with known targets; 279 rows
with missing targets were excluded from validation scoring.

The same validation observations and wMAPE definition were used to
compare CatBoost v1, v2, and v3.

## 8. Model Development

Three CatBoost approaches were developed.

### 8.1 CatBoost v1 --- Initial regression model

CatBoost v1 established the first machine-learning benchmark using 24
features and historical rolling-demand statistics.

  Metric                              Result
  ------------------------------ -----------
  Training observations            1,226,617
  Validation observations             68,721
  Features                                24
  Baseline wMAPE (28-day mean)      21.8397%
  CatBoost v1 wMAPE                 15.0948%
  MAE                                 4.8406
  RMSE                                9.4464

The model reduced wMAPE by 6.7449 percentage points compared with the
baseline. Subsequent diagnostics identified a tendency to underpredict
combinations experiencing stronger demand growth.

### 8.2 CatBoost v2 --- Fixed-origin regression model

CatBoost v2 addressed the training/forecast feature mismatch by training
on historical fixed-origin examples.

  Parameter                                Value
  ------------------------- --------------------
  Model                       CatBoost Regressor
  Training observations                  961,468
  Validation observations                 68,721
  Features                                    24
  Iterations configured                    1,500
  Best iteration                           1,110
  Learning rate                             0.05
  Tree depth                                   8
  Loss function                             RMSE
  L2 regularisation                            5
  Random seed                                 42

Validation performance:

  Metric                   Result
  ---------------- --------------
  wMAPE              **14.6904%**
  MAE                  **4.7109**
  RMSE                 **9.3312**
  Aggregate bias     **-0.0758%**

### 8.3 CatBoost v3 --- Two-stage classifier/regressor

CatBoost v3 tested a two-stage approach to account for zero-sales
observations:

1.  A classifier estimated the probability of a positive sale.
2.  A regressor estimated quantity conditional on a positive sale.
3.  Expected demand was calculated as the probability of a positive sale
    multiplied by the conditional quantity prediction.

The approach achieved 14.7388% validation wMAPE, slightly behind
CatBoost v2.

## 9. Final Model Selection

All three models were evaluated on the same 68,721 validation
observations.

  Metric             CatBoost v1    CatBoost v2   CatBoost v3
  ---------------- ------------- -------------- -------------
  wMAPE                 15.0948%   **14.6904%**      14.7388%
  MAE                     4.8406     **4.7109**        4.7265
  RMSE                    9.4464     **9.3312**        9.3321
  Aggregate bias        -1.6530%       -0.0758%      +0.0524%

CatBoost v2 achieved the lowest overall wMAPE, MAE, and RMSE. It also
recorded the lowest wMAPE in seven of eight product categories and four
of five forecast-horizon bands.

**CatBoost v2 was selected as the preferred development model.**

Its wMAPE improved by 0.4044 percentage points compared with v1,
equivalent to a 2.68% relative improvement. The difference between v2
and v3 was small, so the selection should be understood as the best
result on the chosen validation period, not proof of statistically
significant superiority.

## 10. Q4 2025 Forecast Generation

The selected CatBoost v2 model was used to generate daily predictions
for 1 October--31 December 2025.

Forecast generation used the forecast feature dataset constructed from
information available through 30 September 2025.

Quality checks confirmed:

-   69,000 forecast records generated.
-   15 restaurants and 50 menu items represented.
-   No duplicate date-restaurant-item keys.
-   No missing predictions.
-   No negative predictions after the non-negative output check.

The total forecast quantity across the complete Q4 forecast grid was
**1,623,657.97**.

### 10.1 Monthly forecast

  Month             Forecast quantity   Average daily forecast
  --------------- ------------------- ------------------------
  October 2025             596,212.11                19,232.65
  November 2025            533,936.03                17,797.87
  December 2025            493,509.83                15,919.67
  **Q4 total**       **1,623,657.97**                      ---

### 10.2 Forecast by category

  Category      Q4 forecast quantity
  ----------- ----------------------
  Drinks                  321,669.95
  Burgers                 261,955.35
  Sides                   248,587.71
  Chicken                 237,865.72
  Breakfast               187,432.95
  Combos                  182,952.10
  Desserts                119,542.57
  Specials                 63,651.62

## 11. Q4 Forecast Evaluation Against Available Actuals

After generating and preserving the Q4 forecasts, the original source
dataset was checked for Q4 actual quantities.

Actual quantities were available for 68,687 of the 69,000 forecast
records. The remaining 313 records had missing actual quantities and
were excluded from scoring rather than treated as zero.

  Evaluation item                                  Result
  ---------------------------------------- --------------
  Q4 forecast records                              69,000
  Records with known actual quantities             68,687
  Records with missing actual quantities              313
  Actual-data coverage                             99.55%
  Actual demand on scored records            1,593,979.00
  Predicted demand on scored records         1,615,859.82
  Q4 wMAPE                                    **14.866%**
  MAE                                              3.4499
  RMSE                                             6.7166
  Aggregate forecast bias                        +1.3727%

The positive aggregate bias indicates that the model overforecast total
demand by approximately 1.37% across the scored records.

### 11.1 Monthly evaluation

  -----------------------------------------------------------------------
  Month               Actual      Predicted       Forecast   Difference %
                    quantity       quantity     difference 
  ----------- -------------- -------------- -------------- --------------
  October         586,492.00     593,411.39       6,919.39         +1.18%

  November        521,896.00     531,567.92       9,671.92         +1.85%

  December        485,591.00     490,880.52       5,289.52         +1.09%
  -----------------------------------------------------------------------

Forecast difference is predicted quantity minus actual quantity.
Positive values therefore indicate overforecasting.

### 11.2 Segment-level findings

At category level, Chicken and Combos recorded the lowest wMAPE, while
Specials recorded the highest.

  Category      Q4 wMAPE
  ----------- ----------
  Chicken         13.42%
  Combos          13.82%
  Drinks          13.83%
  Burgers         14.90%
  Sides           15.24%
  Desserts        16.55%
  Breakfast       16.79%
  Specials        18.02%

Restaurant-level wMAPE ranged from 14.14% to 15.91%. Menu-item and
restaurant-item diagnostics are available in the corresponding
evaluation outputs.

### 11.3 Interpretation and limitations

The Q4 evaluation is a post-forecast comparison using actual quantities
present in the original source dataset. The 313 records with missing
actual quantities are not scored.

The Q4 wMAPE of 14.866% is an evaluation result on the available Q4
records. It should not be described as an independently verified
official competition score unless the competition evaluator and
actual-data provenance are confirmed.

## 12. Final Deliverables

The project produces the following main deliverables:

  -----------------------------------------------------------------------------------------------------
  Deliverable                         Location / file
  ----------------------------------- -----------------------------------------------------------------
  Raw dataset                         `data/qsr_demand_dataset.csv`

  Feature engineering                 `src/feature_engineering.py`

  CatBoost v1 model                   `models/catboost_v1.cbm`

  CatBoost v2 model                   `models/catboost_v2.cbm`

  CatBoost v3 models                  `models/`

  Validation predictions and          `outputs/model_evaluation/`
  comparisons                         

  Q4 daily forecast                   `outputs/forecasts/q4_2025_daily_forecast.csv`

  Q4 summary outputs                  `outputs/forecasts/`

  Final Excel workbook                `outputs/forecasts/HAVI_Q4_2025_Forecast_Submission_Final.xlsx`
  -----------------------------------------------------------------------------------------------------

The final Excel workbook contains:

1.  Raw Uploaded Data --- the original 69,000-row Q4 dataset, preserved
    in a separate sheet.
2.  Executive Summary.
3.  Monthly Summary.
4.  Restaurant Summary.
5.  Menu Item Summary.
6.  Category Summary.
7.  Restaurant-Item Summary.
8.  Daily Forecast Detail.
9.  Missing Actuals.

The workbook combines the original uploaded data with forecast
evaluation summaries and detailed outputs.

## 13. Project Structure

The project structure is organised around data, documentation, scripts,
models, and outputs. The following is a representative structure; retain
any additional scripts and files already present in the repository.

``` text
HAVI_DEMAND_FORECASTING/
├── data/
│   ├── raw/
│   └── processed/
│       └── model_data/
├── notebooks/
├── docs/
│   ├── methodology.md
│   ├── project_findings.md
│   └── eda_*.md
├── src/
│   ├── inspect_data.py
│   ├── forecast_diagnostics.py
│   ├── feature_engineering.py
│   ├── train_model_v2.py
│   ├── compare_catboost_v1_v2.py
│   └── ...
├── outputs/
│   ├── eda/
│   ├── model_evaluation/
│   └── forecasts/
│       └── HAVI_Q4_2025_Forecast_Submission_Final.xlsx
├── models/
├── submissions/
├── requirements.txt
└── README.md
```

## 14. Reproducibility

Run scripts from the project root so relative paths resolve correctly.
For example:

``` powershell
python src/inspect_data.py
python src/feature_engineering.py
python src/train_model_v2.py
python src/compare_catboost_v1_v2.py
python src/21_generate_q4_forecast.py
python src/23_analyse_q4_forecast_accuracy.py
```

The script names above identify key workflow stages; check the actual
files in `src/` before running them. The final workbook is an output
artifact and does not need to be regenerated to inspect the results.

## 15. Final Project Status

  Stage                                       Status
  ------------------------------------------- -----------------
  Dataset inspection and quality assessment   Completed
  EDA exercises 1--5                          Completed
  Feature engineering                         Completed
  Leakage and forecast-origin design          Completed
  Baseline forecasting                        Completed
  CatBoost v1 development and diagnostics     Completed
  CatBoost v2 fixed-origin development        Completed
  CatBoost v3 two-stage experiment            Completed
  Model comparison and selection              Completed
  Q4 2025 forecast generation                 Completed
  Q4 forecast quality checks                  Completed
  Q4 evaluation against available actuals     Completed
  Final Excel workbook                        Completed
  README and repository housekeeping          Final close-out

## 16. Conclusion

This project developed and evaluated a daily demand forecasting workflow
for 15 QSR restaurants and 50 menu items.

The modelling process progressed from an initial CatBoost regression
model to a fixed-origin training approach and a two-stage zero-sales
experiment. CatBoost v2 was selected based on its performance across the
common July--September 2025 validation period.

The selected model generated 69,000 Q4 2025 forecasts. On the 68,687
records with known actual quantities in the source dataset, the forecast
achieved **14.866% wMAPE**, with an aggregate overforecast bias of
**1.3727%**.

The final Excel workbook consolidates the original uploaded Q4 data,
forecast details, summary analyses, and missing-actual records into one
deliverable.

**Project outcome:** The Q4 2025 demand forecasting and evaluation
workflow is complete. Further work, such as additional rolling-origin
backtesting, model monitoring, and operational deployment, would be a
separate extension rather than a requirement for closing this project.
