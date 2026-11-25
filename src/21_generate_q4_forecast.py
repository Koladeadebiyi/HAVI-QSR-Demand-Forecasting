
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "catboost_v2.cbm"

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_data"
    / "forecast_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD FORECAST DATA
# ============================================================

print("Loading Q4 forecast dataset...")

forecast_df = pd.read_csv(INPUT_PATH)

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"]
)

print("Forecast dataset shape:", forecast_df.shape)
print("Forecast period:", forecast_df["date"].min(),
      "to", forecast_df["date"].max())


# ============================================================
# 3. LOAD CATBOOST V2
# ============================================================

print("\nLoading CatBoost v2 model...")

model = CatBoostRegressor()

model.load_model(str(MODEL_PATH))

expected_features = model.feature_names_

print("Model loaded successfully.")
print("Expected features:", len(expected_features))


# ============================================================
# 4. VALIDATE FEATURE AVAILABILITY
# ============================================================

missing_features = [
    col for col in expected_features
    if col not in forecast_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

X_forecast = forecast_df[expected_features].copy()

if X_forecast.isna().any().any():
    missing_counts = X_forecast.isna().sum()

    raise ValueError(
        "Missing values detected in model features:\n"
        f"{missing_counts[missing_counts > 0]}"
    )

print("All 24 model features are available.")
print("No missing model feature values detected.")


# ============================================================
# 5. GENERATE FORECASTS
# ============================================================

print("\nGenerating Q4 demand predictions...")

predictions = model.predict(X_forecast)

forecast_df["predicted_quantity_raw"] = predictions

# Demand quantities cannot be negative.
# Preserve the raw model output for audit purposes.
forecast_df["predicted_quantity"] = np.maximum(
    forecast_df["predicted_quantity_raw"],
    0,
)

print("Predictions generated successfully.")


# ============================================================
# 6. CREATE FINAL FORECAST OUTPUT
# ============================================================

final_forecast = forecast_df[
    [
        "date",
        "restaurant_id",
        "menu_item_id",
        "category",
        "unit_price",
        "is_promotion",
        "is_holiday",
        "is_special_event",
        "predicted_quantity_raw",
        "predicted_quantity",
    ]
].copy()

final_forecast["forecast_month"] = (
    final_forecast["date"].dt.to_period("M").astype(str)
)

final_forecast["forecast_quarter"] = "Q4-2025"


# ============================================================
# 7. DATA QUALITY CHECKS
# ============================================================

print("\nRunning forecast quality checks...")

print("Total forecast rows:", len(final_forecast))

print(
    "Missing predictions:",
    final_forecast["predicted_quantity"].isna().sum()
)

print(
    "Negative raw predictions:",
    (final_forecast["predicted_quantity_raw"] < 0).sum()
)

print(
    "Negative final predictions:",
    (final_forecast["predicted_quantity"] < 0).sum()
)

print(
    "Duplicate forecast keys:",
    final_forecast.duplicated(
        ["date", "restaurant_id", "menu_item_id"]
    ).sum()
)

print(
    "Unique restaurants:",
    final_forecast["restaurant_id"].nunique()
)

print(
    "Unique menu items:",
    final_forecast["menu_item_id"].nunique()
)

print(
    "Total predicted demand:",
    round(final_forecast["predicted_quantity"].sum(), 2)
)

print(
    "Average daily predicted demand:",
    round(
        final_forecast.groupby("date")[
            "predicted_quantity"
        ].sum().mean(),
        2,
    )
)


# ============================================================
# 8. DAILY FORECAST SUMMARY
# ============================================================

daily_forecast = (
    final_forecast.groupby("date", as_index=False)
    .agg(
        predicted_demand=("predicted_quantity", "sum"),
        restaurant_count=("restaurant_id", "nunique"),
        menu_item_count=("menu_item_id", "nunique"),
    )
)


# ============================================================
# 9. MONTHLY FORECAST SUMMARY
# ============================================================

daily_forecast["forecast_month"] = (
    daily_forecast["date"].dt.to_period("M").astype(str)
)

monthly_forecast = (
    daily_forecast.groupby("forecast_month", as_index=False)
    .agg(
        predicted_demand=("predicted_demand", "sum"),
        average_daily_demand=("predicted_demand", "mean"),
        forecast_days=("date", "nunique"),
    )
)


# ============================================================
# 10. RESTAURANT-LEVEL FORECAST SUMMARY
# ============================================================

restaurant_daily = (
    final_forecast.groupby(
        ["date", "restaurant_id"],
        as_index=False,
    )
    .agg(
        daily_demand=("predicted_quantity", "sum")
    )
)

restaurant_forecast = (
    restaurant_daily.groupby("restaurant_id", as_index=False)
    .agg(
        predicted_demand=("daily_demand", "sum"),
        average_daily_demand=("daily_demand", "mean"),
    )
    .sort_values("predicted_demand", ascending=False)
)


# ============================================================
# 11. MENU ITEM FORECAST SUMMARY
# ============================================================

menu_item_daily = (
    final_forecast.groupby(
        ["date", "menu_item_id", "category"],
        as_index=False,
    )
    .agg(
        daily_demand=("predicted_quantity", "sum")
    )
)

menu_item_forecast = (
    menu_item_daily.groupby(
        ["menu_item_id", "category"],
        as_index=False,
    )
    .agg(
        predicted_demand=("daily_demand", "sum"),
        average_daily_demand_across_restaurants=(
            "daily_demand", "mean"
        ),
    )
    .sort_values("predicted_demand", ascending=False)
)


# ============================================================
# 12. CATEGORY-LEVEL FORECAST SUMMARY
# ============================================================

category_daily = (
    final_forecast.groupby(
        ["date", "category"],
        as_index=False,
    )
    .agg(
        daily_demand=("predicted_quantity", "sum")
    )
)

category_forecast = (
    category_daily.groupby("category", as_index=False)
    .agg(
        predicted_demand=("daily_demand", "sum"),
        average_daily_demand=("daily_demand", "mean"),
    )
    .sort_values("predicted_demand", ascending=False)
)


# ============================================================
# 13. RESTAURANT-MENU ITEM FORECAST SUMMARY
# ============================================================

restaurant_item_forecast = (
    final_forecast.groupby(
        ["restaurant_id", "menu_item_id", "category"],
        as_index=False,
    )
    .agg(
        predicted_demand=("predicted_quantity", "sum"),
        average_daily_demand=("predicted_quantity", "mean"),
    )
    .sort_values("predicted_demand", ascending=False)
)


# ============================================================
# 14. SAVE OUTPUT FILES
# ============================================================

outputs = {
    "q4_2025_daily_forecast.csv": final_forecast,
    "q4_2025_daily_summary.csv": daily_forecast,
    "q4_2025_monthly_summary.csv": monthly_forecast,
    "q4_2025_restaurant_summary.csv": restaurant_forecast,
    "q4_2025_menu_item_summary.csv": menu_item_forecast,
    "q4_2025_category_summary.csv": category_forecast,
    "q4_2025_restaurant_item_summary.csv": restaurant_item_forecast,
}

for filename, dataframe in outputs.items():

    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print("Saved:", output_path.name)


# ============================================================
# 15. DISPLAY KEY FORECAST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("Q4 2025 FORECAST SUMMARY")
print("=" * 60)

print("\nMONTHLY FORECAST")
print(monthly_forecast.round(2).to_string(index=False))

print("\nCATEGORY FORECAST")
print(category_forecast.round(2).to_string(index=False))

print("\nTOP 10 RESTAURANTS")
print(restaurant_forecast.head(10).round(2).to_string(index=False))

print("\nTOP 10 MENU ITEMS")
print(menu_item_forecast.head(10).round(2).to_string(index=False))

print("\nForecast generation completed successfully.")
