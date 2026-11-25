
# ============================================================
# HAVI QSR DEMAND FORECASTING
# CATBOOST V2 - HISTORICAL FORECAST-ORIGIN TRAINING DATA
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path


# ------------------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# ------------------------------------------------------------

DATA_PATH = Path("data/qsr_demand_dataset.csv")

OUTPUT_DIR = Path("data/processed/model_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUT = OUTPUT_DIR / "forecast_origin_training_data.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "forecast_origin_summary.csv"

VALIDATION_START = pd.Timestamp("2025-07-01")
HORIZON_DAYS = 92
MIN_HISTORY_DAYS = 365


# ------------------------------------------------------------
# 2. DEFINE MODEL FEATURES
# ------------------------------------------------------------

feature_columns = [
    "restaurant_id",
    "menu_item_id",
    "category",
    "unit_price",

    "day_of_week_num",
    "is_weekend",
    "month",
    "quarter",
    "year",
    "day_of_month",
    "day_of_year",
    "week_of_year",

    "is_promotion",
    "is_holiday",
    "is_special_event",

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

categorical_features = [
    "restaurant_id",
    "menu_item_id",
    "category"
]


# ------------------------------------------------------------
# 3. LOAD AND PREPARE HISTORICAL DATA
# ------------------------------------------------------------

print("\nLOADING HISTORICAL DATA")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

# Keep data strictly before the validation period.
df = df[df["date"] < VALIDATION_START].copy()

# Keep only rows with observed target values for training labels.
# Missing quantities are not interpreted as zero.
df = df.sort_values(
    ["restaurant_id", "menu_item_id", "date"]
).copy()

print("Historical dataset shape:", df.shape)
print("Historical start:", df["date"].min())
print("Historical end:", df["date"].max())
print("Missing target values:", df["quantity"].isna().sum())


# ------------------------------------------------------------
# 4. PREPARE CALENDAR FEATURES
# ------------------------------------------------------------

df["day_of_week_num"] = df["date"].dt.day_of_week
df["is_weekend"] = (
    df["day_of_week_num"] >= 5
).astype(int)

df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["year"] = df["date"].dt.year
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.day_of_year

df["week_of_year"] = (
    df["date"].dt.isocalendar().week.astype(int)
)


# ------------------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "date",
    "quantity",
    "restaurant_id",
    "menu_item_id",
    "category",
    "unit_price",
    "is_promotion",
    "is_holiday",
    "is_special_event"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

duplicate_count = df.duplicated(
    subset=["date", "restaurant_id", "menu_item_id"]
).sum()

if duplicate_count > 0:
    raise ValueError(
        f"Found {duplicate_count} duplicate restaurant-item-date rows."
    )


# ------------------------------------------------------------
# 6. DEFINE HISTORICAL FORECAST ORIGINS
# ------------------------------------------------------------

first_origin = (
    df["date"].min()
    + pd.Timedelta(days=MIN_HISTORY_DAYS - 1)
)

last_origin = (
    VALIDATION_START
    - pd.Timedelta(days=HORIZON_DAYS)
)

# Create origins separated by 92 days.
origins = list(
    pd.date_range(
        start=first_origin,
        end=last_origin,
        freq=f"{HORIZON_DAYS}D"
    )
)

# Add a final origin to ensure the training period reaches
# the end of June 2025. This may create a small overlap
# with the preceding historical forecast window.
if not origins or origins[-1] != last_origin:
    origins.append(last_origin)

origins = sorted(set(origins))

print("\nHISTORICAL FORECAST ORIGINS")
print("Number of origins:", len(origins))

for origin in origins:
    print(
        origin.date(),
        "to",
        (origin + pd.Timedelta(days=HORIZON_DAYS)).date()
    )


# ------------------------------------------------------------
# 7. PREPARE RESTAURANT-ITEM GROUPS
# ------------------------------------------------------------

grouped_data = {
    key: group.sort_values("date").copy()
    for key, group in df.groupby(
        ["restaurant_id", "menu_item_id"],
        sort=False
    )
}

print("\nRestaurant-item combinations:", len(grouped_data))


# ------------------------------------------------------------
# 8. CALCULATE DEMAND FEATURES AT EACH ORIGIN
# ------------------------------------------------------------

def calculate_origin_features(history):

    history = history.sort_values("date").copy()

    quantity_series = (
        history.set_index("date")["quantity"]
    )

    features = {}

    features["mean_7d"] = (
        quantity_series.rolling(
            "7D", min_periods=1
        ).mean().iloc[-1]
    )

    features["mean_28d"] = (
        quantity_series.rolling(
            "28D", min_periods=1
        ).mean().iloc[-1]
    )

    features["mean_56d"] = (
        quantity_series.rolling(
            "56D", min_periods=1
        ).mean().iloc[-1]
    )

    features["mean_90d"] = (
        quantity_series.rolling(
            "90D", min_periods=1
        ).mean().iloc[-1]
    )

    features["mean_365d"] = (
        quantity_series.rolling(
            "365D", min_periods=1
        ).mean().iloc[-1]
    )

    features["std_28d"] = (
        quantity_series.rolling(
            "28D", min_periods=2
        ).std().iloc[-1]
    )

    features["std_90d"] = (
        quantity_series.rolling(
            "90D", min_periods=2
        ).std().iloc[-1]
    )

    features["observed_7d"] = (
        quantity_series.notna()
        .rolling("7D", min_periods=1)
        .sum()
        .iloc[-1]
    )

    features["observed_28d"] = (
        quantity_series.notna()
        .rolling("28D", min_periods=1)
        .sum()
        .iloc[-1]
    )

    return features


# ------------------------------------------------------------
# 9. BUILD TRAINING EXAMPLES FOR EACH ORIGIN
# ------------------------------------------------------------

print("\nBUILDING FORECAST-ORIGIN TRAINING DATA")

training_parts = []
origin_summaries = []

for origin_number, origin in enumerate(origins, start=1):

    forecast_end = (
        origin + pd.Timedelta(days=HORIZON_DAYS)
    )

    print(
        f"\nProcessing origin {origin_number}/{len(origins)}:",
        origin.date()
    )

    origin_feature_rows = []

    for (restaurant_id, menu_item_id), group in grouped_data.items():

        # Only information available on or before the origin.
        history = group[
            group["date"] <= origin
        ].copy()

        if history.empty:
            continue

        # Calculate one fixed demand-feature snapshot
        # for this restaurant-item combination.
        demand_features = calculate_origin_features(history)

        origin_feature_rows.append({
            "restaurant_id": restaurant_id,
            "menu_item_id": menu_item_id,
            **demand_features
        })

    origin_features_df = pd.DataFrame(origin_feature_rows)

    # Select the next 92 days after this forecast origin.
    future_parts = []

    for key, group in grouped_data.items():

        future = group[
            (group["date"] > origin) &
            (group["date"] <= forecast_end)
        ].copy()

        if not future.empty:
            future_parts.append(future)

    if not future_parts:
        print("No future observations found for this origin.")
        continue

    future_df = pd.concat(
        future_parts,
        ignore_index=True
    )

    # Exclude rows where the actual target is unavailable.
    future_df = future_df.dropna(
        subset=["quantity"]
    ).copy()

    # Attach the fixed origin-level demand features.
    origin_training_df = future_df.merge(
        origin_features_df,
        on=["restaurant_id", "menu_item_id"],
        how="inner",
        validate="many_to_one"
    )

    # Record the origin and forecast horizon for diagnostics.
    origin_training_df["forecast_origin"] = origin
    origin_training_df["forecast_horizon_day"] = (
        origin_training_df["date"] - origin
    ).dt.days

    training_parts.append(origin_training_df)

    origin_summaries.append({
        "forecast_origin": origin,
        "forecast_start": origin + pd.Timedelta(days=1),
        "forecast_end": forecast_end,
        "training_rows": len(origin_training_df),
        "unique_dates": origin_training_df["date"].nunique(),
        "unique_combinations": (
            origin_training_df[
                ["restaurant_id", "menu_item_id"]
            ].drop_duplicates().shape[0]
        )
    })

    print("Training rows:", len(origin_training_df))
    print(
        "Unique forecast dates:",
        origin_training_df["date"].nunique()
    )


# ------------------------------------------------------------
# 10. COMBINE ALL HISTORICAL TRAINING WINDOWS
# ------------------------------------------------------------

if not training_parts:
    raise ValueError(
        "No historical forecast-origin training examples were created."
    )

training_origin_df = pd.concat(
    training_parts,
    ignore_index=True
)

origin_summary_df = pd.DataFrame(origin_summaries)

# Keep the exact model features, target, and diagnostic columns.
output_columns = [
    "date",
    "quantity",
    "forecast_origin",
    "forecast_horizon_day"
] + feature_columns

training_origin_df = training_origin_df[
    output_columns
].copy()


# ------------------------------------------------------------
# 11. VALIDATE THE NEW TRAINING DATASET
# ------------------------------------------------------------

print("\nTRAINING DATA VALIDATION")

print("Final shape:", training_origin_df.shape)

print(
    "Unique forecast origins:",
    training_origin_df["forecast_origin"].nunique()
)

print(
    "Unique restaurant-item combinations:",
    training_origin_df[
        ["restaurant_id", "menu_item_id"]
    ].drop_duplicates().shape[0]
)

print(
    "Training date range:",
    training_origin_df["date"].min(),
    "to",
    training_origin_df["date"].max()
)

print(
    "Forecast horizon range:",
    training_origin_df["forecast_horizon_day"].min(),
    "to",
    training_origin_df["forecast_horizon_day"].max()
)

print(
    "Missing target values:",
    training_origin_df["quantity"].isna().sum()
)

print("\nMissing demand features:")

demand_feature_columns = [
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

print(
    training_origin_df[demand_feature_columns]
    .isna()
    .sum()
)

print("\nRows by forecast origin:")
print(
    training_origin_df.groupby("forecast_origin")
    .size()
)


# ------------------------------------------------------------
# 12. SAVE OUTPUTS
# ------------------------------------------------------------

training_origin_df.to_csv(
    TRAINING_OUTPUT,
    index=False
)

origin_summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

print("\nFILES SAVED")
print("Training data:", TRAINING_OUTPUT)
print("Origin summary:", SUMMARY_OUTPUT)

print("\nFORECAST-ORIGIN TRAINING DATA BUILD COMPLETE")
