
# ==========================================
# HAVI QSR DEMAND FORECASTING
# CATBOOST V1 - ERROR ANALYSIS
# ==========================================

import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------

MODEL_PATH = Path("models/catboost_v1.cbm")
DATA_DIR = Path("data/processed/model_data")
OUTPUT_DIR = Path("outputs/model_evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------
# 2. LOAD MODEL AND VALIDATION RESULTS
# ------------------------------------------

model = CatBoostRegressor()
model.load_model(MODEL_PATH)

predictions_df = pd.read_csv(
    OUTPUT_DIR / "catboost_v1_validation_predictions.csv"
)

validation_df = pd.read_csv(
    DATA_DIR / "validation_data.csv",
    usecols=["date", "restaurant_id", "menu_item_id", "category"]
)

# Add category information to the prediction results
df = predictions_df.merge(
    validation_df,
    on=["date", "restaurant_id", "menu_item_id"],
    how="left",
    validate="one_to_one"
)

df["date"] = pd.to_datetime(df["date"])

# Derive time-based grouping variables
df["month"] = df["date"].dt.month_name()
df["day_of_week"] = df["date"].dt.day_name()

# Calculate prediction errors
df["error"] = df["quantity"] - df["predicted_quantity"]
df["absolute_error"] = df["error"].abs()


# ------------------------------------------
# 3. DEFINE EVALUATION FUNCTION
# ------------------------------------------

def calculate_metrics(group):

    actual = group["quantity"].to_numpy()
    predicted = group["predicted_quantity"].to_numpy()

    total_actual = np.sum(np.abs(actual))

    wmape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual > 0 else np.nan
    )

    return pd.Series({
        "observations": len(group),
        "actual_total": np.sum(actual),
        "predicted_total": np.sum(predicted),
        "actual_average": np.mean(actual),
        "predicted_average": np.mean(predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "wMAPE": wmape
    })


# ------------------------------------------
# 4. ERROR ANALYSIS BY RESTAURANT
# ------------------------------------------

restaurant_analysis = (
    df.groupby("restaurant_id")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
    .sort_values("wMAPE")
)

restaurant_analysis.to_csv(
    OUTPUT_DIR / "error_by_restaurant.csv",
    index=False
)


# ------------------------------------------
# 5. ERROR ANALYSIS BY MENU ITEM
# ------------------------------------------

item_analysis = (
    df.groupby("menu_item_id")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
    .sort_values("wMAPE")
)

item_analysis.to_csv(
    OUTPUT_DIR / "error_by_menu_item.csv",
    index=False
)


# ------------------------------------------
# 6. ERROR ANALYSIS BY CATEGORY
# ------------------------------------------

category_analysis = (
    df.groupby("category")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
    .sort_values("wMAPE")
)

category_analysis.to_csv(
    OUTPUT_DIR / "error_by_category.csv",
    index=False
)


# ------------------------------------------
# 7. ERROR ANALYSIS BY MONTH
# ------------------------------------------

month_analysis = (
    df.groupby("month")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

month_order = [
    "July", "August", "September"
]

month_analysis["month"] = pd.Categorical(
    month_analysis["month"],
    categories=month_order,
    ordered=True
)

month_analysis = month_analysis.sort_values("month")

month_analysis.to_csv(
    OUTPUT_DIR / "error_by_month.csv",
    index=False
)


# ------------------------------------------
# 8. ERROR ANALYSIS BY DAY OF WEEK
# ------------------------------------------

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_analysis = (
    df.groupby("day_of_week")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

weekday_analysis["day_of_week"] = pd.Categorical(
    weekday_analysis["day_of_week"],
    categories=weekday_order,
    ordered=True
)

weekday_analysis = weekday_analysis.sort_values("day_of_week")

weekday_analysis.to_csv(
    OUTPUT_DIR / "error_by_weekday.csv",
    index=False
)


# ------------------------------------------
# 9. FEATURE IMPORTANCE
# ------------------------------------------

importance_values = model.get_feature_importance()

feature_names = model.feature_names_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
}).sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    OUTPUT_DIR / "catboost_v1_feature_importance.csv",
    index=False
)


# ------------------------------------------
# 10. SAVE DETAILED ERROR RECORDS
# ------------------------------------------

df.to_csv(
    OUTPUT_DIR / "catboost_v1_error_details.csv",
    index=False
)


# ------------------------------------------
# 11. DISPLAY RESULTS
# ------------------------------------------

print("\n--- RESTAURANT-LEVEL ERROR ANALYSIS ---")
print(restaurant_analysis.round(4).to_string(index=False))

print("\n--- TOP 10 MENU ITEMS BY wMAPE ---")
print(
    item_analysis.nlargest(10, "wMAPE")
    .round(4)
    .to_string(index=False)
)

print("\n--- CATEGORY-LEVEL ERROR ANALYSIS ---")
print(category_analysis.round(4).to_string(index=False))

print("\n--- MONTHLY ERROR ANALYSIS ---")
print(month_analysis.round(4).to_string(index=False))

print("\n--- WEEKDAY ERROR ANALYSIS ---")
print(weekday_analysis.round(4).to_string(index=False))

print("\n--- TOP 15 FEATURES BY IMPORTANCE ---")
print(feature_importance.head(15).round(4).to_string(index=False))


print("\n--- ERROR ANALYSIS COMPLETED ---")
print("Results saved in:", OUTPUT_DIR)
