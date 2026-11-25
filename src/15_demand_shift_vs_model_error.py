
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "outputs" / "model_evaluation"

SHIFT_FILE = OUTPUT_DIR / "seasonally_adjusted_q3_demand_shift.csv"

ERROR_FILE = OUTPUT_DIR / "catboost_v1_error_details.csv"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading demand shift and model error data...")

shift_df = pd.read_csv(SHIFT_FILE)
error_df = pd.read_csv(ERROR_FILE)

print("Demand shift shape:", shift_df.shape)
print("Model error shape:", error_df.shape)

# --------------------------------------------------
# 3. AGGREGATE MODEL ERRORS BY RESTAURANT-ITEM
# --------------------------------------------------

error_summary = (
    error_df
    .groupby(["restaurant_id", "menu_item_id"])
    .agg(
        actual_total=("quantity", "sum"),
        predicted_total=("predicted_quantity", "sum"),
        absolute_error_sum=("absolute_error", "sum"),
        observations=("quantity", "count")
    )
    .reset_index()
)

# Calculate wMAPE for each restaurant-item combination

error_summary["combo_wmape_pct"] = (
    error_summary["absolute_error_sum"]
    / error_summary["actual_total"]
) * 100

# Calculate prediction bias
# Negative = underprediction
# Positive = overprediction

error_summary["prediction_bias_pct"] = (
    (
        error_summary["predicted_total"]
        - error_summary["actual_total"]
    )
    / error_summary["actual_total"]
) * 100

# --------------------------------------------------
# 4. MERGE DEMAND SHIFT WITH MODEL ERRORS
# --------------------------------------------------

comparison = shift_df.merge(
    error_summary,
    on=["restaurant_id", "menu_item_id"],
    how="inner"
)

print("\nMerged dataset shape:", comparison.shape)

# --------------------------------------------------
# 5. CORRELATION ANALYSIS
# --------------------------------------------------

wmape_correlation = comparison[
    ["q3_demand_change_pct", "combo_wmape_pct"]
].corr().iloc[0, 1]

bias_correlation = comparison[
    ["q3_demand_change_pct", "prediction_bias_pct"]
].corr().iloc[0, 1]

print("\nCORRELATION RESULTS")

print(
    "Demand change vs wMAPE:",
    round(wmape_correlation, 4)
)

print(
    "Demand change vs prediction bias:",
    round(bias_correlation, 4)
)

# --------------------------------------------------
# 6. GROUP COMBINATIONS BY DEMAND CHANGE
# --------------------------------------------------

comparison["demand_change_group"] = pd.cut(
    comparison["q3_demand_change_pct"],
    bins=[
        float("-inf"),
        0,
        5,
        10,
        float("inf")
    ],
    labels=[
        "Demand decline",
        "0% to 5% increase",
        "5% to 10% increase",
        "Above 10% increase"
    ]
)

group_summary = (
    comparison
    .groupby("demand_change_group", observed=False)
    .agg(
        combinations=("menu_item_id", "count"),
        average_demand_change=("q3_demand_change_pct", "mean"),
        average_wmape=("combo_wmape_pct", "mean"),
        median_wmape=("combo_wmape_pct", "median"),
        average_prediction_bias=("prediction_bias_pct", "mean")
    )
    .reset_index()
)

print("\nMODEL ERROR BY DEMAND CHANGE GROUP")

print(group_summary.to_string(index=False))

# --------------------------------------------------
# 7. IDENTIFY HIGH-ERROR COMBINATIONS
# --------------------------------------------------

print("\nTOP 15 HIGHEST WMAPE COMBINATIONS")

print(
    comparison.sort_values(
        "combo_wmape_pct",
        ascending=False
    )[
        [
            "restaurant_id",
            "menu_item_name",
            "category",
            "q3_demand_change_pct",
            "combo_wmape_pct",
            "prediction_bias_pct"
        ]
    ].head(15).to_string(index=False)
)

# --------------------------------------------------
# 8. EXPORT RESULTS
# --------------------------------------------------

comparison.to_csv(
    OUTPUT_DIR / "demand_shift_vs_model_error.csv",
    index=False
)

group_summary.to_csv(
    OUTPUT_DIR / "model_error_by_demand_shift_group.csv",
    index=False
)

print("\nAnalysis completed.")

print(
    "Saved:",
    OUTPUT_DIR / "demand_shift_vs_model_error.csv"
)

print(
    "Saved:",
    OUTPUT_DIR / "model_error_by_demand_shift_group.csv"
)


# --------------------------------------------------
# 6B. POOLED PERFORMANCE BY DEMAND CHANGE GROUP
# --------------------------------------------------

weighted_summary = (
    comparison
    .groupby("demand_change_group", observed=False)
    .agg(
        combinations=("menu_item_id", "count"),
        actual_total=("actual_total", "sum"),
        predicted_total=("predicted_total", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
        average_demand_change=("q3_demand_change_pct", "mean")
    )
    .reset_index()
)

# Calculate pooled wMAPE
weighted_summary["pooled_wmape_pct"] = (
    weighted_summary["absolute_error_sum"]
    / weighted_summary["actual_total"]
) * 100

# Calculate pooled prediction bias
weighted_summary["pooled_bias_pct"] = (
    (
        weighted_summary["predicted_total"]
        - weighted_summary["actual_total"]
    )
    / weighted_summary["actual_total"]
) * 100

print("\nVOLUME-WEIGHTED MODEL PERFORMANCE BY DEMAND GROUP")

print(
    weighted_summary[
        [
            "demand_change_group",
            "combinations",
            "actual_total",
            "predicted_total",
            "pooled_wmape_pct",
            "pooled_bias_pct"
        ]
    ].to_string(index=False)
)

# Save weighted results
weighted_summary.to_csv(
    OUTPUT_DIR / "weighted_model_error_by_demand_group.csv",
    index=False
)







from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DATA_DIR = PROJECT_DIR / "data" / "processed" / "model_data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "model_evaluation"

VALIDATION_FILE = MODEL_DATA_DIR / "validation_data.csv"
ERROR_FILE = OUTPUT_DIR / "catboost_v1_error_details.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD VALIDATION FEATURES AND MODEL ERRORS
# --------------------------------------------------

print("Loading validation features and prediction errors...")

validation_df = pd.read_csv(VALIDATION_FILE)
error_df = pd.read_csv(ERROR_FILE)

print("Validation data shape:", validation_df.shape)
print("Prediction error data shape:", error_df.shape)

# --------------------------------------------------
# 3. EXTRACT ROLLING FEATURES
# --------------------------------------------------

feature_columns = [
    "restaurant_id",
    "menu_item_id",
    "mean_7d",
    "mean_28d",
    "mean_56d",
    "mean_90d",
    "mean_365d"
]

# Rolling features are fixed at the validation forecast origin.
# Keep one feature record per restaurant-item combination.

rolling_features = (
    validation_df[feature_columns]
    .drop_duplicates(
        subset=["restaurant_id", "menu_item_id"]
    )
    .copy()
)

print("\nUnique restaurant-item feature records:", len(rolling_features))

# --------------------------------------------------
# 4. AGGREGATE VALIDATION MODEL ERRORS
# --------------------------------------------------

error_summary = (
    error_df
    .groupby(["restaurant_id", "menu_item_id"])
    .agg(
        actual_total=("quantity", "sum"),
        predicted_total=("predicted_quantity", "sum"),
        absolute_error_sum=("absolute_error", "sum"),
        observations=("quantity", "count")
    )
    .reset_index()
)

error_summary["combo_wmape_pct"] = (
    error_summary["absolute_error_sum"]
    / error_summary["actual_total"]
) * 100

error_summary["prediction_bias_pct"] = (
    (
        error_summary["predicted_total"]
        - error_summary["actual_total"]
    )
    / error_summary["actual_total"]
) * 100

# --------------------------------------------------
# 5. MERGE FEATURES WITH MODEL ERRORS
# --------------------------------------------------

diagnostics = rolling_features.merge(
    error_summary,
    on=["restaurant_id", "menu_item_id"],
    how="inner"
)

print("\nMerged diagnostic shape:", diagnostics.shape)

# --------------------------------------------------
# 6. CALCULATE DEMAND MOMENTUM RATIOS
# --------------------------------------------------

# Ratio above 1 means the shorter-term average
# is higher than the longer-term reference.

diagnostics["momentum_7d_28d"] = (
    diagnostics["mean_7d"]
    / diagnostics["mean_28d"].replace(0, np.nan)
)

diagnostics["momentum_28d_90d"] = (
    diagnostics["mean_28d"]
    / diagnostics["mean_90d"].replace(0, np.nan)
)

diagnostics["momentum_28d_365d"] = (
    diagnostics["mean_28d"]
    / diagnostics["mean_365d"].replace(0, np.nan)
)

# --------------------------------------------------
# 7. GROUP BY SHORT-TERM DEMAND MOMENTUM
# --------------------------------------------------

diagnostics["momentum_group"] = pd.cut(
    diagnostics["momentum_7d_28d"],
    bins=[
        float("-inf"),
        0.90,
        1.00,
        1.10,
        float("inf")
    ],
    labels=[
        "Below 0.90",
        "0.90 to 1.00",
        "1.00 to 1.10",
        "Above 1.10"
    ]
)

# --------------------------------------------------
# 8. CALCULATE VOLUME-WEIGHTED PERFORMANCE
# --------------------------------------------------

momentum_summary = (
    diagnostics
    .groupby("momentum_group", observed=False)
    .agg(
        combinations=("menu_item_id", "count"),
        actual_total=("actual_total", "sum"),
        predicted_total=("predicted_total", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
        average_momentum=("momentum_7d_28d", "mean")
    )
    .reset_index()
)

momentum_summary["pooled_wmape_pct"] = (
    momentum_summary["absolute_error_sum"]
    / momentum_summary["actual_total"]
) * 100

momentum_summary["pooled_bias_pct"] = (
    (
        momentum_summary["predicted_total"]
        - momentum_summary["actual_total"]
    )
    / momentum_summary["actual_total"]
) * 100

print("\nMODEL PERFORMANCE BY 7D/28D DEMAND MOMENTUM")

print(
    momentum_summary.to_string(index=False)
)

# --------------------------------------------------
# 9. CORRELATION ANALYSIS
# --------------------------------------------------

print("\nCORRELATION WITH MODEL PERFORMANCE")

print(
    "7D/28D momentum vs wMAPE:",
    round(
        diagnostics[
            ["momentum_7d_28d", "combo_wmape_pct"]
        ].corr().iloc[0, 1],
        4
    )
)

print(
    "7D/28D momentum vs prediction bias:",
    round(
        diagnostics[
            ["momentum_7d_28d", "prediction_bias_pct"]
        ].corr().iloc[0, 1],
        4
    )
)

# --------------------------------------------------
# 10. COMPARE OTHER ROLLING FEATURE RATIOS
# --------------------------------------------------

print("\nAVERAGE ROLLING FEATURE RATIOS")

print(
    diagnostics[
        [
            "momentum_7d_28d",
            "momentum_28d_90d",
            "momentum_28d_365d"
        ]
    ].describe().round(4)
)

# --------------------------------------------------
# 11. EXPORT RESULTS
# --------------------------------------------------

diagnostics.to_csv(
    OUTPUT_DIR / "rolling_feature_error_diagnostics.csv",
    index=False
)

momentum_summary.to_csv(
    OUTPUT_DIR / "model_error_by_rolling_momentum.csv",
    index=False
)

print("\nRolling feature diagnostics completed.")

print(
    "Saved:",
    OUTPUT_DIR / "rolling_feature_error_diagnostics.csv"
)

print(
    "Saved:",
    OUTPUT_DIR / "model_error_by_rolling_momentum.csv"
)
