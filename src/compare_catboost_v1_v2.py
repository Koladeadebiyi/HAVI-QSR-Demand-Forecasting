
# ==========================================
# HAVI QSR DEMAND FORECASTING
# CATBOOST V1 VS V2
# COMPARATIVE ERROR ANALYSIS
# ==========================================

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------

DATA_DIR = Path("data/processed/model_data")
OUTPUT_DIR = Path("outputs/model_evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

V1_FILE = OUTPUT_DIR / "catboost_v1_validation_predictions.csv"
V2_FILE = OUTPUT_DIR / "catboost_v2_validation_predictions.csv"

VALIDATION_FILE = DATA_DIR / "validation_data.csv"


# ------------------------------------------
# 2. LOAD VALIDATION PREDICTIONS
# ------------------------------------------

v1_df = pd.read_csv(V1_FILE)
v2_df = pd.read_csv(V2_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("--- DATA LOADING ---")
print("V1 predictions:", v1_df.shape)
print("V2 predictions:", v2_df.shape)
print("Validation data:", validation_df.shape)


# ------------------------------------------
# 3. PREPARE AND CHECK DATA
# ------------------------------------------

key_columns = [
    "date",
    "restaurant_id",
    "menu_item_id"
]

v1_df["date"] = pd.to_datetime(v1_df["date"])
v2_df["date"] = pd.to_datetime(v2_df["date"])
validation_df["date"] = pd.to_datetime(validation_df["date"])

# Confirm unique prediction records.
for name, df in [("V1", v1_df), ("V2", v2_df)]:
    if df.duplicated(key_columns).any():
        raise ValueError(f"Duplicate prediction keys found in {name}.")

# Rename actual and predicted columns.
v1_df = v1_df[
    key_columns + ["quantity", "predicted_quantity"]
].rename(columns={
    "quantity": "actual_v1",
    "predicted_quantity": "predicted_v1"
})

v2_df = v2_df[
    key_columns + ["quantity", "predicted_quantity"]
].rename(columns={
    "quantity": "actual_v2",
    "predicted_quantity": "predicted_v2"
})

# Merge the two prediction datasets.
comparison_df = v1_df.merge(
    v2_df,
    on=key_columns,
    how="inner",
    validate="one_to_one"
)

# Confirm both models were evaluated on the same observations.
if len(v1_df) != len(v2_df) or len(comparison_df) != len(v1_df):
    raise ValueError(
        "V1 and V2 do not contain identical validation observations."
    )

if not np.allclose(
    comparison_df["actual_v1"],
    comparison_df["actual_v2"]
):
    raise ValueError(
        "Actual quantities differ between V1 and V2 predictions."
    )

comparison_df["actual"] = comparison_df["actual_v1"]

# Bring in category information from validation data.
category_df = validation_df[
    key_columns + ["category"]
].drop_duplicates()

comparison_df = comparison_df.merge(
    category_df,
    on=key_columns,
    how="left",
    validate="one_to_one"
)

if comparison_df["category"].isna().any():
    raise ValueError("Some validation records have no category.")

# Remove temporary actual columns.
comparison_df = comparison_df.drop(
    columns=["actual_v1", "actual_v2"]
)

print("\n--- VALIDATION ALIGNMENT ---")
print("Matched observations:", len(comparison_df))
print("Unique restaurants:", comparison_df["restaurant_id"].nunique())
print("Unique menu items:", comparison_df["menu_item_id"].nunique())
print("Categories:", comparison_df["category"].nunique())
print(
    "Validation period:",
    comparison_df["date"].min(),
    "to",
    comparison_df["date"].max()
)


# ------------------------------------------
# 4. CREATE ERROR AND HORIZON FEATURES
# ------------------------------------------

comparison_df["error_v1"] = (
    comparison_df["predicted_v1"] - comparison_df["actual"]
)

comparison_df["error_v2"] = (
    comparison_df["predicted_v2"] - comparison_df["actual"]
)

comparison_df["absolute_error_v1"] = (
    comparison_df["error_v1"].abs()
)

comparison_df["absolute_error_v2"] = (
    comparison_df["error_v2"].abs()
)

comparison_df["date"] = pd.to_datetime(comparison_df["date"])

# Validation forecast origin was June 30, 2025.
validation_origin = pd.Timestamp("2025-06-30")

comparison_df["forecast_horizon_day"] = (
    comparison_df["date"] - validation_origin
).dt.days

comparison_df["month"] = (
    comparison_df["date"].dt.strftime("%Y-%m")
)

# Group forecast days into useful horizon bands.
comparison_df["horizon_band"] = pd.cut(
    comparison_df["forecast_horizon_day"],
    bins=[0, 7, 14, 30, 60, 92],
    labels=[
        "Days 1-7",
        "Days 8-14",
        "Days 15-30",
        "Days 31-60",
        "Days 61-92"
    ],
    include_lowest=True
)


# ------------------------------------------
# 5. DEFINE PERFORMANCE METRICS
# ------------------------------------------

def calculate_metrics(group, model_name):

    actual = group["actual"].to_numpy()

    if model_name == "CatBoost v1":
        predicted = group["predicted_v1"].to_numpy()
    else:
        predicted = group["predicted_v2"].to_numpy()

    absolute_error = np.abs(actual - predicted)

    wmape = (
        absolute_error.sum() / np.abs(actual).sum()
    ) * 100

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # Positive bias means overprediction.
    # Negative bias means underprediction.
    bias = (
        (predicted - actual).sum() / actual.sum()
    ) * 100

    return {
        "model": model_name,
        "observations": len(group),
        "actual_demand": actual.sum(),
        "predicted_demand": predicted.sum(),
        "wMAPE": wmape,
        "MAE": mae,
        "RMSE": rmse,
        "bias_pct": bias
    }


# ------------------------------------------
# 6. OVERALL MODEL COMPARISON
# ------------------------------------------

overall_results = pd.DataFrame([
    calculate_metrics(comparison_df, "CatBoost v1"),
    calculate_metrics(comparison_df, "CatBoost v2")
])

print("\n--- OVERALL MODEL COMPARISON ---")
print(overall_results.round(4).to_string(index=False))

overall_results.to_csv(
    OUTPUT_DIR / "v1_v2_overall_comparison.csv",
    index=False
)


# ------------------------------------------
# 7. FUNCTION FOR GROUPED COMPARISONS
# ------------------------------------------

def grouped_comparison(df, group_column):

    results = []

    for group_value, group in df.groupby(
        group_column,
        observed=True,
        dropna=False
    ):

        for model_name in ["CatBoost v1", "CatBoost v2"]:

            metrics = calculate_metrics(group, model_name)

            metrics["group"] = group_value

            results.append(metrics)

    results_df = pd.DataFrame(results)

    # Reorder columns for easier reading.
    columns = [
        "group",
        "model",
        "observations",
        "actual_demand",
        "predicted_demand",
        "wMAPE",
        "MAE",
        "RMSE",
        "bias_pct"
    ]

    return results_df[columns]


# ------------------------------------------
# 8. RESTAURANT-LEVEL COMPARISON
# ------------------------------------------

restaurant_results = grouped_comparison(
    comparison_df,
    "restaurant_id"
)

restaurant_results.to_csv(
    OUTPUT_DIR / "v1_v2_by_restaurant.csv",
    index=False
)


# ------------------------------------------
# 9. MENU-ITEM COMPARISON
# ------------------------------------------

menu_item_results = grouped_comparison(
    comparison_df,
    "menu_item_id"
)

menu_item_results.to_csv(
    OUTPUT_DIR / "v1_v2_by_menu_item.csv",
    index=False
)


# ------------------------------------------
# 10. CATEGORY COMPARISON
# ------------------------------------------

category_results = grouped_comparison(
    comparison_df,
    "category"
)

category_results.to_csv(
    OUTPUT_DIR / "v1_v2_by_category.csv",
    index=False
)


# ------------------------------------------
# 11. MONTHLY COMPARISON
# ------------------------------------------

monthly_results = grouped_comparison(
    comparison_df,
    "month"
)

monthly_results.to_csv(
    OUTPUT_DIR / "v1_v2_by_month.csv",
    index=False
)


# ------------------------------------------
# 12. FORECAST HORIZON COMPARISON
# ------------------------------------------

horizon_results = grouped_comparison(
    comparison_df,
    "horizon_band"
)

horizon_results.to_csv(
    OUTPUT_DIR / "v1_v2_by_forecast_horizon.csv",
    index=False
)


# ------------------------------------------
# 13. CALCULATE PERFORMANCE CHANGE
# ------------------------------------------

def calculate_v2_improvement(results_df):

    pivot = results_df.pivot(
        index="group",
        columns="model",
        values="wMAPE"
    ).reset_index()

    pivot["wmape_improvement_pp"] = (
        pivot["CatBoost v1"] - pivot["CatBoost v2"]
    )

    pivot["relative_improvement_pct"] = (
        pivot["wmape_improvement_pp"]
        / pivot["CatBoost v1"]
    ) * 100

    # Positive values mean V2 improved.
    # Negative values mean V2 performed worse.
    return pivot


restaurant_improvement = calculate_v2_improvement(
    restaurant_results
)

menu_item_improvement = calculate_v2_improvement(
    menu_item_results
)

category_improvement = calculate_v2_improvement(
    category_results
)

monthly_improvement = calculate_v2_improvement(
    monthly_results
)

horizon_improvement = calculate_v2_improvement(
    horizon_results
)

restaurant_improvement.to_csv(
    OUTPUT_DIR / "v1_v2_restaurant_improvement.csv",
    index=False
)

menu_item_improvement.to_csv(
    OUTPUT_DIR / "v1_v2_menu_item_improvement.csv",
    index=False
)

category_improvement.to_csv(
    OUTPUT_DIR / "v1_v2_category_improvement.csv",
    index=False
)

monthly_improvement.to_csv(
    OUTPUT_DIR / "v1_v2_monthly_improvement.csv",
    index=False
)

horizon_improvement.to_csv(
    OUTPUT_DIR / "v1_v2_horizon_improvement.csv",
    index=False
)


# ------------------------------------------
# 14. SAVE DETAILED COMPARISON DATA
# ------------------------------------------

comparison_df.to_csv(
    OUTPUT_DIR / "v1_v2_detailed_error_comparison.csv",
    index=False
)


# ------------------------------------------
# 15. PRINT KEY FINDINGS
# ------------------------------------------

print("\n--- CATEGORY PERFORMANCE ---")
print(category_improvement.round(4).to_string(index=False))

print("\n--- MONTHLY PERFORMANCE ---")
print(monthly_improvement.round(4).to_string(index=False))

print("\n--- FORECAST HORIZON PERFORMANCE ---")
print(horizon_improvement.round(4).to_string(index=False))

print("\n--- RESTAURANT PERFORMANCE ---")
print(restaurant_improvement.round(4).to_string(index=False))

print("\n--- TOP 10 MENU ITEMS IMPROVED BY V2 ---")
print(
    menu_item_improvement.sort_values(
        "wmape_improvement_pp",
        ascending=False
    ).head(10).round(4).to_string(index=False)
)

print("\n--- TOP 10 MENU ITEMS WHERE V2 REGRESSED ---")
print(
    menu_item_improvement.sort_values(
        "wmape_improvement_pp",
        ascending=True
    ).head(10).round(4).to_string(index=False)
)


# ------------------------------------------
# 16. COMPLETION
# ------------------------------------------

print("\n--- FILES SAVED ---")
print("Overall comparison")
print("Restaurant comparison")
print("Menu-item comparison")
print("Category comparison")
print("Monthly comparison")
print("Forecast-horizon comparison")
print("Detailed row-level comparison")

print("\n--- V1 VS V2 ERROR ANALYSIS COMPLETE ---")
