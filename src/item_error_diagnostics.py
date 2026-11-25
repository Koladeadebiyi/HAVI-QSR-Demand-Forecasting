
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "model_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD HISTORICAL ITEM PROFILE
# --------------------------------------------------

historical_profile = pd.read_csv(
    PROJECT_DIR / "outputs" /"eda"/ "restaurant_item_demand_summary.csv"
)

# --------------------------------------------------
# 3. LOAD VALIDATION ERROR RESULTS
# --------------------------------------------------

validation_errors = pd.read_csv(
    OUTPUT_DIR / "catboost_v1_error_details.csv"
)

# --------------------------------------------------
# 4. AGGREGATE HISTORICAL CHARACTERISTICS
# --------------------------------------------------

historical_item_summary = (
    historical_profile
    .groupby(
        ["menu_item_id", "menu_item_name", "category"]
    )
    .agg(
        historical_average_demand=(
            "average_daily_quantity", "mean"
        ),
        historical_cv=(
            "coefficient_of_variation", "mean"
        ),
        historical_zero_sales_rate=(
            "zero_sales_rate", "mean"
        )
    )
    .reset_index()
)

# --------------------------------------------------
# 5. AGGREGATE VALIDATION PERFORMANCE
# --------------------------------------------------

validation_item_summary = (
    validation_errors
    .groupby("menu_item_id")
    .agg(
        validation_actual_total=("quantity", "sum"),
        validation_predicted_total=(
            "predicted_quantity", "sum"
        ),
        validation_actual_average=("quantity", "mean"),
        validation_predicted_average=(
            "predicted_quantity", "mean"
        ),
        validation_mae=("absolute_error", "mean"),
        validation_absolute_error=(
            "absolute_error", "sum"
        ),
        observations=("quantity", "count")
    )
    .reset_index()
)

validation_item_summary["validation_wmape"] = (
    validation_item_summary["validation_absolute_error"]
    / validation_item_summary["validation_actual_total"]
) * 100

validation_item_summary["prediction_bias_pct"] = (
    (
        validation_item_summary["validation_predicted_total"]
        - validation_item_summary["validation_actual_total"]
    )
    / validation_item_summary["validation_actual_total"]
) * 100

# --------------------------------------------------
# 6. COMBINE BOTH SUMMARIES
# --------------------------------------------------

diagnostics = historical_item_summary.merge(
    validation_item_summary,
    on="menu_item_id",
    how="inner"
)

diagnostics = diagnostics.sort_values(
    "validation_wmape",
    ascending=False
)

# --------------------------------------------------
# 6B. INVESTIGATE RELATIONSHIPS AND CATEGORY ERRORS
# --------------------------------------------------

print("\nCORRELATION: HISTORICAL DEMAND VS VALIDATION ERROR")

correlation_results = diagnostics[
    [
        "historical_average_demand",
        "historical_cv",
        "historical_zero_sales_rate",
        "validation_wmape"
    ]
].corr(method="pearson")

print(correlation_results["validation_wmape"].sort_values(
    ascending=False
))


# Category-level weighted performance

category_summary = (
    diagnostics
    .groupby("category")
    .agg(
        actual_total=("validation_actual_total", "sum"),
        absolute_error_total=("validation_absolute_error", "sum"),
        average_historical_cv=("historical_cv", "mean"),
        average_zero_sales_rate=(
            "historical_zero_sales_rate", "mean"
        ),
        menu_items=("menu_item_id", "nunique")
    )
    .reset_index()
)

category_summary["category_wmape"] = (
    category_summary["absolute_error_total"]
    / category_summary["actual_total"]
) * 100

print("\nCATEGORY-LEVEL DEMAND DIAGNOSTICS")

print(
    category_summary.sort_values(
        "category_wmape",
        ascending=False
    ).to_string(index=False)
)


# Best-performing menu items

print("\n10 BEST-PERFORMING MENU ITEMS")

print(
    diagnostics.sort_values("validation_wmape")[
        [
            "menu_item_id",
            "menu_item_name",
            "historical_cv",
            "historical_zero_sales_rate",
            "validation_wmape",
            "prediction_bias_pct"
        ]
    ].head(10).to_string(index=False)
)


# Save category summary

category_summary.to_csv(
    OUTPUT_DIR / "category_demand_diagnostics.csv",
    index=False
)


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

output_file = OUTPUT_DIR / "item_demand_diagnostics.csv"

diagnostics.to_csv(output_file, index=False)

# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\nITEM-LEVEL DEMAND DIAGNOSTICS")

print(
    diagnostics[
        [
            "menu_item_id",
            "menu_item_name",
            "category",
            "historical_average_demand",
            "historical_cv",
            "historical_zero_sales_rate",
            "validation_actual_average",
            "validation_wmape",
            "prediction_bias_pct"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved diagnostic file:")
print(output_file)
