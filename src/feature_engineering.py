
from tokenize import group

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = Path("data/qsr_demand_dataset.csv")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


print("\nORIGINAL DATASET IDENTIFIER CHECK")

print(
    df[["restaurant_id", "restaurant_name",
        "menu_item_id", "menu_item_name"]].head()
)

print("\nOriginal columns:")
print(df.columns.tolist())

# ==========================================
# 2. SEPARATE HISTORICAL AND FORECAST DATA
# ==========================================

historical_df = df[
    df["date"] < "2025-10-01"
].copy()

forecast_df = df[
    (df["date"] >= "2025-10-01") &
    (df["date"] <= "2025-12-31")
].copy()

# VALIDATION PERIOD SPLIT
validation_start = "2025-07-01"
validation_end = "2025-09-30"

validation_df = historical_df[
    (historical_df["date"] >= validation_start) &
    (historical_df["date"] <= validation_end)
].copy()

validation_training_df = historical_df[
    historical_df["date"] < validation_start
].copy()

print("\nVALIDATION DATASET SPLIT")
print("Validation training records:", len(validation_training_df))
print("Validation records:", len(validation_df))
print(
    "Validation training period:",
    validation_training_df["date"].min(),
    "to",
    validation_training_df["date"].max()
)
print(
    "Validation period:",
    validation_df["date"].min(),
    "to",
    validation_df["date"].max()
)


print("\nDATASET SPLIT")

print("Historical records:", len(historical_df))
print("Forecast records:", len(forecast_df))

print(
    "Historical period:",
    historical_df["date"].min(),
    "to",
    historical_df["date"].max()
)

print(
    "Forecast period:",
    forecast_df["date"].min(),
    "to",
    forecast_df["date"].max()
)


# ==========================================
# 3. CHECK TARGET AVAILABILITY
# ==========================================

print("\nTARGET AVAILABILITY")

print(
    "Missing historical quantities:",
    historical_df["quantity"].isna().sum()
)

print(
    "Missing forecast quantities:",
    forecast_df["quantity"].isna().sum()
)


# ==========================================
# 4. CALENDAR FEATURE ENGINEERING
# ==========================================

def create_calendar_features(data):

    data = data.copy()

    # Day of week: Monday = 0, Sunday = 6
    data["day_of_week"] = data["date"].dt.day_of_week

    # Day name
    data["day_name"] = data["date"].dt.day_name()

    # Weekend indicator
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Month and quarter
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter

    # Additional date features
    data["day_of_month"] = data["date"].dt.day
    data["day_of_year"] = data["date"].dt.day_of_year

    # ISO week number
    data["week_of_year"] = (
        data["date"].dt.isocalendar().week.astype(int)
    )

    return data


# Apply to both datasets
historical_df = create_calendar_features(historical_df)
forecast_df = create_calendar_features(forecast_df)


# ==========================================
# 5. INSPECT CALENDAR FEATURES
# ==========================================

calendar_columns = [
    "date",
    "day_of_week",
    "day_name",
    "is_weekend",
    "month",
    "quarter",
    "day_of_month",
    "day_of_year",
    "week_of_year"
]

print("\nCALENDAR FEATURE PREVIEW")

print(
    historical_df[calendar_columns]
    .head(10)
    .to_string(index=False)
)


print("\nWEEKEND DISTRIBUTION")

print(
    historical_df.groupby("is_weekend")["date"]
    .nunique()
)


print("\nMONTH DISTRIBUTION")

print(
    historical_df.groupby("month")["date"]
    .nunique()
    .sort_index()
)


# ==========================================
# 6. INVESTIGATE FORECAST TARGET VALUES
# ==========================================

print("\nFORECAST TARGET INVESTIGATION")

print("Total forecast records:", len(forecast_df))

print(
    "Non-missing forecast quantities:",
    forecast_df["quantity"].notna().sum()
)

print(
    "Missing forecast quantities:",
    forecast_df["quantity"].isna().sum()
)

# Examine missing quantities by month
missing_forecast = forecast_df[
    forecast_df["quantity"].isna()
].copy()

print("\nMISSING FORECAST QUANTITIES BY MONTH")

print(
    missing_forecast.groupby(
        missing_forecast["date"].dt.month
    ).size()
)

# Examine non-missing quantities by month
available_forecast = forecast_df[
    forecast_df["quantity"].notna()
].copy()

print("\nNON-MISSING FORECAST QUANTITIES BY MONTH")

print(
    available_forecast.groupby(
        available_forecast["date"].dt.month
    ).size()
)

# Check whether missing values are concentrated
# in particular restaurants
print("\nMISSING QUANTITIES BY RESTAURANT")

print(
    missing_forecast.groupby("restaurant_id")
    .size()
    .sort_values(ascending=False)
)

# ==========================================
# 7. PREVENT FORECAST TARGET LEAKAGE
# ==========================================

# Preserve the historical target for model training.
# Remove future target values from the forecast dataset.
forecast_df = forecast_df.drop(columns=["quantity"])

print("\nTARGET LEAKAGE CHECK")

print("Historical quantity column available:",
      "quantity" in historical_df.columns)

print("Forecast quantity column available:",
      "quantity" in forecast_df.columns)

print("Historical records:", len(historical_df))
print("Forecast records:", len(forecast_df))


# ==========================================
# 8. CHECK HISTORICAL SERIES CONTINUITY
# ==========================================

print("\nHISTORICAL SERIES CONTINUITY CHECK")

# Count records for each restaurant-menu item combination
series_summary = (
    historical_df.groupby(["restaurant_id", "menu_item_id"])
    .agg(
        total_records=("date", "size"),
        unique_dates=("date", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max")
    )
    .reset_index()
)

print("Restaurant-item combinations:", len(series_summary))

print("\nRecords per combination:")
print(series_summary["total_records"].describe())

print("\nUnique dates per combination:")
print(series_summary["unique_dates"].describe())

# Check whether each combination has the expected number of historical dates
expected_dates = historical_df["date"].nunique()

series_summary["has_complete_date_coverage"] = (
    series_summary["unique_dates"] == expected_dates
)

print(
    "\nCombinations with complete date coverage:",
    series_summary["has_complete_date_coverage"].sum()
)

print(
    "Combinations with incomplete date coverage:",
    (~series_summary["has_complete_date_coverage"]).sum()
)

# Check duplicate dates within each restaurant-item combination
duplicate_count = historical_df.duplicated(
    subset=["restaurant_id", "menu_item_id", "date"]
).sum()

print("\nDuplicate restaurant-item-date records:", duplicate_count)


# ==========================================
# 9. HISTORICAL TARGET MISSINGNESS BY SERIES
# ==========================================

print("\nHISTORICAL TARGET MISSINGNESS CHECK")

missing_summary = (
    historical_df.groupby(["restaurant_id", "menu_item_id"])
    .agg(
        total_records=("quantity", "size"),
        observed_quantities=("quantity", "count"),
        missing_quantities=("quantity", lambda x: x.isna().sum())
    )
    .reset_index()
)

missing_summary["missing_rate"] = (
    missing_summary["missing_quantities"] /
    missing_summary["total_records"]
)

print("Total missing historical quantities:",
      historical_df["quantity"].isna().sum())

print("\nMissing quantities per combination:")
print(missing_summary["missing_quantities"].describe())

print("\nCombinations with missing quantities:",
      (missing_summary["missing_quantities"] > 0).sum())

print("\nCombinations with no missing quantities:",
      (missing_summary["missing_quantities"] == 0).sum())

print("\nHighest missing rates:")
print(
    missing_summary.sort_values(
        "missing_rate", ascending=False
    ).head(10)
)



# ==========================================
# 10. HISTORICAL MISSINGNESS OVER TIME
# ==========================================

print("\nHISTORICAL MISSINGNESS BY YEAR")

historical_df["year"] = historical_df["date"].dt.year
historical_df["month"] = historical_df["date"].dt.month

missing_by_year = (
    historical_df.groupby("year")["quantity"]
    .agg(
        total_records="size",
        missing_quantities=lambda x: x.isna().sum()
    )
    .reset_index()
)

missing_by_year["missing_rate"] = (
    missing_by_year["missing_quantities"] /
    missing_by_year["total_records"] * 100
)

print(missing_by_year.to_string(index=False))


print("\nHISTORICAL MISSINGNESS BY MONTH")

missing_by_month = (
    historical_df.groupby("month")["quantity"]
    .agg(
        total_records="size",
        missing_quantities=lambda x: x.isna().sum()
    )
    .reset_index()
)

missing_by_month["missing_rate"] = (
    missing_by_month["missing_quantities"] /
    missing_by_month["total_records"] * 100
)

print(missing_by_month.to_string(index=False))


print("\nMISSINGNESS BY QUARTER")

missing_by_quarter = (
    historical_df.assign(
        quarter=historical_df["date"].dt.quarter
    )
    .groupby("quarter")["quantity"]
    .agg(
        total_records="size",
        missing_quantities=lambda x: x.isna().sum()
    )
    .reset_index()
)

missing_by_quarter["missing_rate"] = (
    missing_by_quarter["missing_quantities"] /
    missing_by_quarter["total_records"] * 100
)

print(missing_by_quarter.to_string(index=False))

def calculate_reference_features(group):
    # Retrieve the restaurant and menu item identifiers
    # from the group name because pandas excludes grouping columns.
    restaurant_id, menu_item_id = group.name

    # Sort records chronologically
    group = group.sort_values("date").copy()

    # Create a date-indexed quantity series
    quantity_series = group.set_index("date")["quantity"]

    # Calculate rolling demand features
    group["mean_7d"] = (
        quantity_series.rolling("7D", min_periods=1).mean().values
    )

    group["mean_28d"] = (
        quantity_series.rolling("28D", min_periods=1).mean().values
    )

    group["mean_56d"] = (
        quantity_series.rolling("56D", min_periods=1).mean().values
    )

    group["mean_90d"] = (
        quantity_series.rolling("90D", min_periods=1).mean().values
    )

    group["mean_365d"] = (
        quantity_series.rolling("365D", min_periods=1).mean().values
    )

    group["std_28d"] = (
        quantity_series.rolling("28D", min_periods=2).std().values
    )

    group["std_90d"] = (
        quantity_series.rolling("90D", min_periods=2).std().values
    )

    # Count available historical observations in each window
    group["observed_7d"] = (
        quantity_series.notna().rolling("7D", min_periods=1).sum().values
    )

    group["observed_28d"] = (
        quantity_series.notna().rolling("28D", min_periods=1).sum().values
    )

    # Restore the grouping identifiers
    group["restaurant_id"] = restaurant_id
    group["menu_item_id"] = menu_item_id

    return group


# ==========================================
# 11. HISTORICAL REFERENCE FEATURES
# ==========================================

print("\nCREATING HISTORICAL REFERENCE FEATURES")


historical_reference_df = (
    historical_df.groupby(
        ["restaurant_id", "menu_item_id"],
        group_keys=False,
        sort=False
    )
    .apply(calculate_reference_features)
    .reset_index(drop=True)
)


# Sort records to ensure chronological calculations
historical_df = historical_df.sort_values(
    ["restaurant_id", "menu_item_id", "date"]
).copy()

# Store the last available historical date
forecast_origin = historical_df["date"].max()

print("Forecast origin:", forecast_origin)







def calculate_training_features(group):
    restaurant_id, menu_item_id = group.name

    group = group.sort_values("date").copy()
    group = group.set_index("date")

    # Move each observed quantity forward by one calendar day.
    # This makes the previous day's quantity available to the
    # current date, without including the current day's target.
    previous_quantity = group["quantity"].copy()
    previous_quantity.index = (
        previous_quantity.index + pd.Timedelta(days=1)
    )

    # Rolling averages
    group["mean_7d"] = (
        previous_quantity.rolling("7D", min_periods=1)
        .mean().reindex(group.index)
    )
    group["mean_28d"] = (
        previous_quantity.rolling("28D", min_periods=1)
        .mean().reindex(group.index)
    )
    group["mean_56d"] = (
        previous_quantity.rolling("56D", min_periods=1)
        .mean().reindex(group.index)
    )
    group["mean_90d"] = (
        previous_quantity.rolling("90D", min_periods=1)
        .mean().reindex(group.index)
    )
    group["mean_365d"] = (
        previous_quantity.rolling("365D", min_periods=1)
        .mean().reindex(group.index)
    )

    # Rolling standard deviations
    group["std_28d"] = (
        previous_quantity.rolling("28D", min_periods=2)
        .std().reindex(group.index)
    )
    group["std_90d"] = (
        previous_quantity.rolling("90D", min_periods=2)
        .std().reindex(group.index)
    )

    # Number of previously observed target values
    group["observed_7d"] = (
        previous_quantity.notna().rolling("7D", min_periods=1)
        .sum().reindex(group.index)
    )
    group["observed_28d"] = (
        previous_quantity.notna().rolling("28D", min_periods=1)
        .sum().reindex(group.index)
    )

    group["observed_7d"] = group["observed_7d"].fillna(0)
    group["observed_28d"] = group["observed_28d"].fillna(0)

    group["restaurant_id"] = restaurant_id
    group["menu_item_id"] = menu_item_id

    return group.reset_index()



print("\nCREATING LEAKAGE-SAFE HISTORICAL TRAINING FEATURES")

historical_df = (
    historical_df.groupby(
        ["restaurant_id", "menu_item_id"],
        group_keys=False,
        sort=False
    )
    .apply(calculate_training_features)
    .reset_index(drop=True)
)

print("\nHISTORICAL TRAINING FEATURE VALIDATION")

print("Historical rows:", len(historical_df))
print("Unique restaurants:", historical_df["restaurant_id"].nunique())
print("Unique menu items:", historical_df["menu_item_id"].nunique())

print("\nTraining feature preview:")

print(
    historical_df[
        [
            "date",
            "restaurant_id",
            "menu_item_id",
            "quantity",
            "mean_7d",
            "mean_28d",
            "mean_365d"
        ]
    ].head(10)
)

print("\nROLLING FEATURE LEAKAGE CHECK")

sample = historical_df[
    (historical_df["restaurant_id"] == "R01") &
    (historical_df["menu_item_id"] == "M01")
].sort_values("date").head(10)

print(
    sample[
        [
            "date",
            "quantity",
            "mean_7d",
            "mean_28d",
            "observed_7d",
            "observed_28d"
        ]
    ].to_string(index=False)
)

print(
    "\nFirst observation rolling mean:",
    sample.iloc[0]["mean_7d"]
)

# ==========================================
# DEBUG: CHECK HISTORICAL DATAFRAME STRUCTURE
# ==========================================

print("\nDATAFRAME STRUCTURE CHECK")

print("Columns:")
print(historical_df.columns.tolist())

print("\nIndex names:")
print(historical_df.index.names)

print("\nIndex type:")
print(type(historical_df.index))

print("\nDataFrame shape:")
print(historical_df.shape)

print("\nFirst five rows:")
print(historical_df.head())


# ==========================================
# 12. CREATE FORECAST-ORIGIN FEATURE TABLE
# ==========================================

print("\nCREATING FORECAST-ORIGIN FEATURE TABLE")

# Select the latest historical record for each
# restaurant-item combination.
forecast_reference = (
    historical_reference_df.sort_values("date")
    .groupby(
        ["restaurant_id", "menu_item_id"],
        as_index=False
    )
    .tail(1)
    .copy()
)

# VERIFY FORECAST REFERENCE FEATURES FOR R01 / M01

check_reference = historical_reference_df[
    (historical_reference_df["restaurant_id"] == "R01") &
    (historical_reference_df["menu_item_id"] == "M01")
].sort_values("date")

print("\nFORECAST REFERENCE VERIFICATION: R01 / M01")

print(
    check_reference[
        [
            "date",
            "quantity",
            "mean_7d",
            "mean_28d",
            "mean_365d"
        ]
    ].tail(10).to_string(index=False)
)

# Keep only the identifiers and reference features
reference_columns = [
    "restaurant_id",
    "menu_item_id",
    "mean_7d",
    "mean_28d",
    "mean_56d",
    "mean_90d",
    "mean_365d",
    "std_28d",
    "std_90d",
    "observed_7d",
    "observed_28d"
]

forecast_reference = forecast_reference[
    reference_columns
].copy()

print("Forecast reference shape:", forecast_reference.shape)

print(
    "Unique restaurant-item combinations:",
    forecast_reference[
        ["restaurant_id", "menu_item_id"]
    ].drop_duplicates().shape[0]
)

print("\nMissing values in reference features:")
print(forecast_reference.isna().sum())

print("\nForecast reference preview:")
print(forecast_reference.head())

print("\nMERGING REFERENCE FEATURES INTO FORECAST DATA")

reference_columns = [
    "restaurant_id",
    "menu_item_id",
    "mean_7d",
    "mean_28d",
    "mean_56d",
    "mean_90d",
    "mean_365d",
    "std_28d",
    "std_90d",
    "observed_7d",
    "observed_28d"
]

forecast_df= forecast_df.merge(
    forecast_reference[reference_columns],
    on=["restaurant_id", "menu_item_id"],
    how="left",
    validate="many_to_one"
)

print("\nFORECAST MERGE VALIDATION")
print("Original forecast rows:", len(forecast_df))
print("Forecast rows after merge:", len(forecast_df))

print(
    "Unique restaurant-item combinations:",
    forecast_df[["restaurant_id", "menu_item_id"]]
    .drop_duplicates()
    .shape[0]
)

print("\nMissing values in merged reference features:")
print(forecast_df[
    reference_columns[2:]
].isna().sum())

print("\nMerged forecast preview:")
print(
    forecast_df[
        [
            "date",
            "restaurant_id",
            "menu_item_id",
            "mean_7d",
            "mean_28d",
            "mean_365d"
        ]
    ].head()
)

print("\nForecast target leakage check:")
print("Quantity column available:", "quantity" in forecast_df.columns)


# ==========================================
# CREATE VALIDATION-ORIGIN REFERENCE FEATURES
# ==========================================

print("\nCREATING VALIDATION-ORIGIN FEATURE TABLE")

validation_reference_df = (
    validation_training_df.groupby(
        ["restaurant_id", "menu_item_id"],
        group_keys=False,
        sort=False
    )
    .apply(calculate_reference_features)
    .reset_index(drop=True)
)

# Keep the latest historical record for each combination
validation_reference = (
    validation_reference_df.sort_values("date")
    .groupby(
        ["restaurant_id", "menu_item_id"],
        as_index=False
    )
    .tail(1)
    .copy()
)

print("Validation reference shape:", validation_reference.shape)
print("Validation reference origin:", validation_reference["date"].max())
print(
    "Unique restaurant-item combinations:",
    validation_reference[
        ["restaurant_id", "menu_item_id"]
    ].drop_duplicates().shape[0]
)

# ==========================================
# MERGE FEATURES INTO VALIDATION DATA
# ==========================================

print("\nMERGING REFERENCE FEATURES INTO VALIDATION DATA")

validation_model_df = validation_df.merge(
    validation_reference[
        [
            "restaurant_id",
            "menu_item_id",
            "mean_7d",
            "mean_28d",
            "mean_56d",
            "mean_90d",
            "mean_365d",
            "std_28d",
            "std_90d",
            "observed_7d",
            "observed_28d"
        ]
    ],
    on=["restaurant_id", "menu_item_id"],
    how="left",
    validate="many_to_one"
)

print("\nVALIDATION MERGE CHECK")
print("Original validation rows:", len(validation_df))
print("Merged validation rows:", len(validation_model_df))
print("Validation target available:", "quantity" in validation_model_df.columns)

rolling_columns = [
    "mean_7d",
    "mean_28d",
    "mean_56d",
    "mean_90d",
    "mean_365d",
    "std_28d",
    "std_90d",
    "observed_7d",
    "observed_28d"
]

print("\nMissing validation reference features:")
print(validation_model_df[rolling_columns].isna().sum())

print("\nVALIDATION FEATURE PREVIEW: R01 / M01")

print(
    validation_model_df[
        (validation_model_df["restaurant_id"] == "R01") &
        (validation_model_df["menu_item_id"] == "M01")
    ][
        [
            "date",
            "quantity",
            "mean_7d",
            "mean_28d",
            "mean_365d"
        ]
    ].head(5).to_string(index=False)
)


# ============================================================
# CHECK VALIDATION AND FORECAST DATASETS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION AND FORECAST DATASET CHECK")
print("=" * 60)

# Check available columns
print("\nValidation columns:")
print(validation_model_df.columns.tolist())

print("\nForecast columns:")
print(forecast_df.columns.tolist())

# Check missing values in validation dataset
print("\nMissing values in validation dataset:")
print(
    validation_model_df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

# Check missing values in forecast dataset
print("\nMissing values in forecast dataset:")
print(
    forecast_df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

# Confirm target availability
print("\nTarget availability:")
print("Quantity in validation:", "quantity" in validation_model_df.columns)
print("Quantity in forecast:", "quantity" in forecast_df.columns)


# ============================================================
# ALIGN CALENDAR FEATURES
# ============================================================

for dataset in [validation_model_df, forecast_df]:
    dataset["date"] = pd.to_datetime(dataset["date"])

    dataset["quarter"] = dataset["date"].dt.quarter
    dataset["day_of_month"] = dataset["date"].dt.day
    dataset["day_of_year"] = dataset["date"].dt.day_of_year
    dataset["week_of_year"] = (
        dataset["date"].dt.isocalendar().week.astype(int)
    )

    dataset["day_name"] = dataset["date"].dt.day_name()

print("\nCalendar features aligned.")

# ============================================================
# DEFINE INITIAL MODEL FEATURES
# ============================================================

feature_columns = [
    # Restaurant and product information
    "restaurant_id",
    "menu_item_id",
    "category",
    "unit_price",

    # Calendar features
    "day_of_week_num",
    "is_weekend",
    "month",
    "quarter",
    "year",
    "day_of_month",
    "day_of_year",
    "week_of_year",

    # Commercial and event indicators
    "is_promotion",
    "is_holiday",
    "is_special_event",

    # Historical demand features
    "mean_7d",
    "mean_28d",
    "mean_56d",
    "mean_90d",
    "mean_365d",
    "std_28d",
    "std_90d",
    "observed_7d",
    "observed_28d"
]

print("\nNumber of selected features:", len(feature_columns))

# Confirm that every selected feature exists in both datasets
missing_validation_features = [
    col for col in feature_columns
    if col not in validation_model_df.columns
]

missing_forecast_features = [
    col for col in feature_columns
    if col not in forecast_df.columns
]

print("Missing validation features:", missing_validation_features)
print("Missing forecast features:", missing_forecast_features)

# ============================================================
# CHECK FEATURE COMPLETENESS AND PREPARE VALIDATION DATA
# ============================================================

# Check missing values in selected features
print("\nMissing selected features - validation:")
print(
    validation_model_df[feature_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nMissing selected features - forecast:")
print(
    forecast_df[feature_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

# Remove rows with missing target values from validation scoring
validation_scoring_df = validation_model_df.dropna(
    subset=["quantity"]
).copy()

# Separate validation predictors and actual target
X_validation = validation_scoring_df[feature_columns].copy()
y_validation = validation_scoring_df["quantity"].copy()

# Prepare forecast predictors
X_forecast = forecast_df[feature_columns].copy()

# Identify categorical features for the model
categorical_features = [
    "restaurant_id",
    "menu_item_id",
    "category"
]

# Convert categorical columns to strings
for col in categorical_features:
    X_validation[col] = X_validation[col].astype(str)
    X_forecast[col] = X_forecast[col].astype(str)

print("\nVALIDATION AND FORECAST PREPARATION")
print("=" * 50)

print("Validation rows before removing missing targets:",
      len(validation_model_df))

print("Validation rows available for scoring:",
      len(X_validation))

print("Validation rows excluded due to missing targets:",
      validation_model_df["quantity"].isna().sum())

print("Validation feature shape:", X_validation.shape)
print("Validation target shape:", y_validation.shape)
print("Forecast feature shape:", X_forecast.shape)

print("\nCategorical features:", categorical_features)

print("\nValidation target summary:")
print(y_validation.describe())



# Prepare model training dataset
model_training_df = historical_df[
    historical_df["date"] < validation_start
].copy()

# Exclude rows with missing target quantities
model_training_df = model_training_df.dropna(subset=["quantity"])

# Separate training features and target
X_train = model_training_df[feature_columns].copy()
y_train = model_training_df["quantity"].copy()

# Convert categorical features to strings
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)

print("\n--- MODEL TRAINING DATA ---")
print("Training period:", model_training_df["date"].min(),
      "to", model_training_df["date"].max())
print("Training rows:", len(X_train))
print("Training features:", X_train.shape[1])
print("Training target shape:", y_train.shape)
print("Missing training features:",
      X_train.isnull().sum().sum())
print("Missing training targets:", y_train.isnull().sum())


print("\n--- MISSING TRAINING VALUES BY FEATURE ---")
missing_features = X_train.isnull().sum()

print(missing_features[missing_features > 0])

print("\nTotal missing feature values:", X_train.isnull().sum().sum())

print("\n--- BASELINE FEATURE CHECK ---")

print("Validation mean_28d missing:",
      X_validation["mean_28d"].isnull().sum())

print("Forecast mean_28d missing:",
      X_forecast["mean_28d"].isnull().sum())


# ------------------------------------------
# BASELINE MODEL: 28-DAY HISTORICAL AVERAGE
# ------------------------------------------

import numpy as np

y_validation_array = y_validation.to_numpy()

# Use the 28-day historical average as prediction
baseline_predictions = X_validation["mean_28d"].to_numpy()

# Ensure predictions are non-negative
baseline_predictions = np.maximum(baseline_predictions, 0)

# Calculate wMAPE
baseline_wmape = (
    np.sum(np.abs(y_validation_array - baseline_predictions))
    / np.sum(np.abs(y_validation_array))
) * 100

print("\n--- BASELINE MODEL RESULTS ---")
print("Baseline method: 28-day historical average")
print("Validation observations:", len(y_validation_array))
print("Baseline wMAPE:", round(baseline_wmape, 4), "%")


# ------------------------------------------
# SAVE PREPARED MODEL DATASETS
# ------------------------------------------

from pathlib import Path

output_dir = Path("data/processed/model_data")
output_dir.mkdir(parents=True, exist_ok=True)

# Save training data
training_output = model_training_df[
    ["date", "quantity"] + feature_columns
].copy()

training_output.to_csv(
    output_dir / "training_data.csv",
    index=False
)

# Save validation data
validation_output = validation_model_df[
    validation_model_df["quantity"].notna()
][["date", "quantity"] + feature_columns].copy()

validation_output.to_csv(
    output_dir / "validation_data.csv",
    index=False
)

# Save forecast data (no target quantity)
forecast_output = forecast_df[
    ["date"] + feature_columns
].copy()

forecast_output.to_csv(
    output_dir / "forecast_data.csv",
    index=False
)

print("\n--- MODEL DATASETS SAVED ---")
print("Training:", training_output.shape)
print("Validation:", validation_output.shape)
print("Forecast:", forecast_output.shape)
print("Output directory:", output_dir)
