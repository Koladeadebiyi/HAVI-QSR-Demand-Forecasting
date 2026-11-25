
# ==========================================
# HAVI QSR DEMAND FORECASTING
# CATBOOST V2
# RESIDUAL AND BIAS ANALYSIS
# ==========================================

import numpy as np
import pandas as pd

from pathlib import Path


# ------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------

DATA_DIR = Path("data/processed/model_data")
OUTPUT_DIR = Path("outputs/model_evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = (
    OUTPUT_DIR / "v1_v2_detailed_error_comparison.csv"
)

VALIDATION_FILE = DATA_DIR / "validation_data.csv"


# ------------------------------------------
# 2. LOAD DATA
# ------------------------------------------

comparison_df = pd.read_csv(COMPARISON_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

comparison_df["date"] = pd.to_datetime(
    comparison_df["date"]
)

validation_df["date"] = pd.to_datetime(
    validation_df["date"]
)

key_columns = [
    "date",
    "restaurant_id",
    "menu_item_id"
]

print("--- DATA LOADING ---")
print("Comparison data:", comparison_df.shape)
print("Validation data:", validation_df.shape)


# ------------------------------------------
# 3. MERGE HISTORICAL DEMAND FEATURES
# ------------------------------------------

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

available_demand_features = [
    col for col in demand_feature_columns
    if col in validation_df.columns
]

if "mean_28d" not in available_demand_features:
    raise ValueError(
        "mean_28d is required for demand-profile analysis."
    )

validation_features = validation_df[
    key_columns + ["category"] + available_demand_features
].copy()

if validation_features.duplicated(key_columns).any():
    raise ValueError(
        "Duplicate keys found in validation data."
    )

df = comparison_df.merge(
    validation_features,
    on=key_columns,
    how="left",
    validate="one_to_one",
    suffixes=("", "_validation")
)

if df["mean_28d"].isna().any():
    raise ValueError(
        "Some observations could not be matched to demand features."
    )

print("\n--- FEATURE MERGE ---")
print("Final analysis shape:", df.shape)
print("Demand features available:", available_demand_features)


# ------------------------------------------
# 4. CALCULATE RESIDUALS
# ------------------------------------------

# Residual = Actual - Predicted.
# Positive residual: model underpredicted.
# Negative residual: model overpredicted.

df["residual_v2"] = (
    df["actual"] - df["predicted_v2"]
)

df["absolute_error_v2"] = (
    df["residual_v2"].abs()
)

df["squared_error_v2"] = (
    df["residual_v2"] ** 2
)

df["overprediction"] = (
    df["predicted_v2"] > df["actual"]
)

df["underprediction"] = (
    df["predicted_v2"] < df["actual"]
)

df["exact_prediction"] = (
    df["predicted_v2"] == df["actual"]
)

df["actual_zero"] = (
    df["actual"] == 0
)

df["predicted_zero"] = (
    df["predicted_v2"] == 0
)

# Forecast error relative to actual demand.
# Avoid division by zero for zero-demand observations.
df["absolute_percentage_error"] = np.where(
    df["actual"] != 0,
    df["absolute_error_v2"] / np.abs(df["actual"]) * 100,
    np.nan
)

# Prediction bias:
# Positive means underprediction.
# Negative means overprediction.
df["bias_direction"] = np.select(
    [
        df["residual_v2"] > 0,
        df["residual_v2"] < 0
    ],
    [
        "Underprediction",
        "Overprediction"
    ],
    default="Exact prediction"
)


# ------------------------------------------
# 5. DEFINE GROUP PERFORMANCE FUNCTION
# ------------------------------------------

def calculate_v2_metrics(group):

    actual = group["actual"].to_numpy()
    predicted = group["predicted_v2"].to_numpy()

    absolute_error = np.abs(actual - predicted)

    total_actual = np.abs(actual).sum()

    wmape = (
        absolute_error.sum() / total_actual * 100
        if total_actual != 0
        else np.nan
    )

    bias_pct = (
        (predicted.sum() - actual.sum())
        / actual.sum() * 100
        if actual.sum() != 0
        else np.nan
    )

    return {
        "observations": len(group),
        "actual_demand": actual.sum(),
        "predicted_demand": predicted.sum(),
        "wMAPE": wmape,
        "MAE": absolute_error.mean(),
        "RMSE": np.sqrt(np.mean((actual - predicted) ** 2)),
        "bias_pct": bias_pct,
        "underprediction_count": (
            group["underprediction"].sum()
        ),
        "overprediction_count": (
            group["overprediction"].sum()
        ),
        "exact_prediction_count": (
            group["exact_prediction"].sum()
        )
    }


# ------------------------------------------
# 6. ITEM-LEVEL ERROR ANALYSIS
# ------------------------------------------

item_results = []

for item, group in df.groupby("menu_item_id"):

    metrics = calculate_v2_metrics(group)

    metrics["menu_item_id"] = item

    metrics["zero_sales_rate_pct"] = (
        group["actual_zero"].mean() * 100
    )

    metrics["mean_historical_demand"] = (
        group["mean_28d"].mean()
    )

    item_results.append(metrics)

item_results = pd.DataFrame(item_results)

item_results = item_results.sort_values(
    "wMAPE",
    ascending=False
)

item_results.to_csv(
    OUTPUT_DIR / "v2_residuals_by_menu_item.csv",
    index=False
)

print("\n--- 10 HIGHEST ERROR MENU ITEMS ---")
print(
    item_results.head(10).round(4).to_string(index=False)
)


# ------------------------------------------
# 7. RESTAURANT-LEVEL ERROR ANALYSIS
# ------------------------------------------

restaurant_results = []

for restaurant, group in df.groupby("restaurant_id"):

    metrics = calculate_v2_metrics(group)

    metrics["restaurant_id"] = restaurant

    restaurant_results.append(metrics)

restaurant_results = pd.DataFrame(restaurant_results)

restaurant_results = restaurant_results.sort_values(
    "wMAPE",
    ascending=False
)

restaurant_results.to_csv(
    OUTPUT_DIR / "v2_residuals_by_restaurant.csv",
    index=False
)


# ------------------------------------------
# 8. RESTAURANT-MENU ITEM ANALYSIS
# ------------------------------------------

combination_results = []

for (restaurant, item), group in df.groupby(
    ["restaurant_id", "menu_item_id"]
):

    metrics = calculate_v2_metrics(group)

    metrics["restaurant_id"] = restaurant
    metrics["menu_item_id"] = item

    metrics["zero_sales_rate_pct"] = (
        group["actual_zero"].mean() * 100
    )

    metrics["mean_historical_demand"] = (
        group["mean_28d"].mean()
    )

    combination_results.append(metrics)

combination_results = pd.DataFrame(combination_results)

combination_results = combination_results.sort_values(
    "wMAPE",
    ascending=False
)

combination_results.to_csv(
    OUTPUT_DIR / "v2_residuals_by_restaurant_item.csv",
    index=False
)

print("\n--- 15 HIGHEST ERROR RESTAURANT-ITEM COMBINATIONS ---")
print(
    combination_results.head(15).round(4).to_string(index=False)
)


# ------------------------------------------
# 9. BIAS ANALYSIS
# ------------------------------------------

bias_results = df.groupby(
    ["menu_item_id", "bias_direction"],
    observed=True
).size().unstack(fill_value=0)

for column in [
    "Underprediction",
    "Overprediction",
    "Exact prediction"
]:
    if column not in bias_results.columns:
        bias_results[column] = 0

bias_results = bias_results.reset_index()

bias_results["underprediction_rate_pct"] = (
    bias_results["Underprediction"]
    / (
        bias_results["Underprediction"]
        + bias_results["Overprediction"]
        + bias_results["Exact prediction"]
    )
) * 100

bias_results.to_csv(
    OUTPUT_DIR / "v2_prediction_direction_by_item.csv",
    index=False
)

print("\n--- PREDICTION DIRECTION COUNTS ---")
print(
    df["bias_direction"].value_counts().to_string()
)


# ------------------------------------------
# 10. ZERO-SALES ANALYSIS
# ------------------------------------------

zero_sales_results = []

for zero_status, group in df.groupby(
    "actual_zero",
    observed=True
):

    metrics = calculate_v2_metrics(group)

    metrics["actual_zero"] = zero_status

    metrics["average_prediction"] = (
        group["predicted_v2"].mean()
    )

    metrics["prediction_zero_rate_pct"] = (
        group["predicted_zero"].mean() * 100
    )

    zero_sales_results.append(metrics)

zero_sales_results = pd.DataFrame(zero_sales_results)

zero_sales_results.to_csv(
    OUTPUT_DIR / "v2_zero_sales_analysis.csv",
    index=False
)

print("\n--- ZERO-SALES PERFORMANCE ---")
print(
    zero_sales_results.round(4).to_string(index=False)
)


# ------------------------------------------
# 11. DEMAND VARIABILITY ANALYSIS
# ------------------------------------------

# CV = standard deviation / mean demand.
# We use the 28-day historical demand features
# available at the validation forecast origin.

if "std_28d" in df.columns:

    df["demand_cv_28d"] = np.where(
        df["mean_28d"] > 0,
        df["std_28d"] / df["mean_28d"],
        np.nan
    )

    df["demand_variability_band"] = pd.cut(
        df["demand_cv_28d"],
        bins=[-np.inf, 0.25, 0.40, 0.60, np.inf],
        labels=[
            "Low variability",
            "Moderate variability",
            "High variability",
            "Very high variability"
        ]
    )

    variability_results = []

    for band, group in df.groupby(
        "demand_variability_band",
        observed=True
    ):

        metrics = calculate_v2_metrics(group)

        metrics["demand_variability_band"] = band

        metrics["average_cv"] = (
            group["demand_cv_28d"].mean()
        )

        variability_results.append(metrics)

    variability_results = pd.DataFrame(
        variability_results
    )

    variability_results.to_csv(
        OUTPUT_DIR / "v2_error_by_demand_variability.csv",
        index=False
    )

    print("\n--- ERROR BY DEMAND VARIABILITY ---")
    print(
        variability_results.round(4).to_string(index=False)
    )


# ------------------------------------------
# 12. HISTORICAL DEMAND LEVEL ANALYSIS
# ------------------------------------------

df["historical_demand_band"] = pd.cut(
    df["mean_28d"],
    bins=[-np.inf, 10, 25, 50, np.inf],
    labels=[
        "Low demand",
        "Moderate demand",
        "High demand",
        "Very high demand"
    ]
)

demand_level_results = []

for band, group in df.groupby(
    "historical_demand_band",
    observed=True
):

    metrics = calculate_v2_metrics(group)

    metrics["historical_demand_band"] = band

    metrics["average_mean_28d"] = (
        group["mean_28d"].mean()
    )

    demand_level_results.append(metrics)

demand_level_results = pd.DataFrame(
    demand_level_results
)

demand_level_results.to_csv(
    OUTPUT_DIR / "v2_error_by_demand_level.csv",
    index=False
)

print("\n--- ERROR BY HISTORICAL DEMAND LEVEL ---")
print(
    demand_level_results.round(4).to_string(index=False)
)


# ------------------------------------------
# 13. SAVE DETAILED RESIDUAL DATA
# ------------------------------------------

df.to_csv(
    OUTPUT_DIR / "catboost_v2_residual_analysis_detail.csv",
    index=False
)


# ------------------------------------------
# 14. COMPLETION
# ------------------------------------------

print("\n--- FILES SAVED ---")
print("v2_residuals_by_menu_item.csv")
print("v2_residuals_by_restaurant.csv")
print("v2_residuals_by_restaurant_item.csv")
print("v2_prediction_direction_by_item.csv")
print("v2_zero_sales_analysis.csv")
print("v2_error_by_demand_variability.csv")
print("v2_error_by_demand_level.csv")
print("catboost_v2_residual_analysis_detail.csv")

print("\n--- RESIDUAL ANALYSIS COMPLETE ---")
