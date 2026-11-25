
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
    / "q4_2025_actual_vs_forecast.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("Q4 FORECAST ACCURACY DIAGNOSTICS")
print("=" * 65)

print(f"\nOriginal records: {len(df):,}")
print("\nAvailable columns:")
print(df.columns.tolist())

# Accommodate the column names used in the Q4 evaluation output.
if "actual_quantity" in df.columns:
    actual_col = "actual_quantity"
elif "quantity" in df.columns:
    actual_col = "quantity"
else:
    raise ValueError("Actual quantity column not found.")

if "predicted_quantity" not in df.columns:
    raise ValueError("Predicted quantity column not found.")

required_columns = [
    "date",
    "restaurant_id",
    "menu_item_id",
    "category",
    actual_col,
    "predicted_quantity"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Retain records with observed actual quantities.
df = df.dropna(
    subset=[actual_col, "predicted_quantity"]
).copy()

df["date"] = pd.to_datetime(df["date"])

df["actual_quantity"] = pd.to_numeric(
    df[actual_col], errors="coerce"
)

df["predicted_quantity"] = pd.to_numeric(
    df["predicted_quantity"], errors="coerce"
)

df = df.dropna(
    subset=["actual_quantity", "predicted_quantity"]
)

# Residual: positive = underforecast; negative = overforecast.
df["residual"] = (
    df["actual_quantity"] - df["predicted_quantity"]
)

df["absolute_error"] = df["residual"].abs()

print(f"Scored records: {len(df):,}")
print(f"Restaurants: {df['restaurant_id'].nunique()}")
print(f"Menu items: {df['menu_item_id'].nunique()}")
print(f"Categories: {df['category'].nunique()}")


# ============================================================
# 3. REUSABLE ACCURACY FUNCTION
# ============================================================

def calculate_accuracy(data, group_column):

    results = (
        data.groupby(group_column)
        .agg(
            scored_records=("actual_quantity", "size"),
            actual_demand=("actual_quantity", "sum"),
            predicted_demand=("predicted_quantity", "sum"),
            absolute_error=("absolute_error", "sum"),
            squared_error=("residual", lambda x: np.square(x).sum()),
        )
        .reset_index()
    )

    results["wMAPE"] = np.where(
        results["actual_demand"] != 0,
        results["absolute_error"]
        / results["actual_demand"] * 100,
        np.nan
    )

    results["MAE"] = (
        results["absolute_error"] / results["scored_records"]
    )

    results["RMSE"] = np.sqrt(
        results["squared_error"] / results["scored_records"]
    )

    results["forecast_bias_pct"] = np.where(
        results["actual_demand"] != 0,
        (
            results["predicted_demand"]
            - results["actual_demand"]
        ) / results["actual_demand"] * 100,
        np.nan
    )

    results["forecast_error"] = (
        results["predicted_demand"]
        - results["actual_demand"]
    )

    return results.sort_values(
        "wMAPE", ascending=False
    ).reset_index(drop=True)


# ============================================================
# 4. RESTAURANT-LEVEL ACCURACY
# ============================================================

restaurant_accuracy = calculate_accuracy(
    df, "restaurant_id"
)

restaurant_accuracy.to_csv(
    OUTPUT_DIR / "q4_restaurant_accuracy.csv",
    index=False
)

print("\n" + "=" * 65)
print("RESTAURANT-LEVEL ACCURACY")
print("=" * 65)

print(
    restaurant_accuracy[
        [
            "restaurant_id",
            "scored_records",
            "actual_demand",
            "predicted_demand",
            "wMAPE",
            "MAE",
            "forecast_bias_pct"
        ]
    ].round(2).to_string(index=False)
)


# ============================================================
# 5. MENU ITEM-LEVEL ACCURACY
# ============================================================

item_accuracy = calculate_accuracy(
    df, "menu_item_id"
)

item_accuracy.to_csv(
    OUTPUT_DIR / "q4_menu_item_accuracy.csv",
    index=False
)

print("\n" + "=" * 65)
print("TOP 10 MENU ITEMS WITH HIGHEST FORECAST ERROR")
print("=" * 65)

print(
    item_accuracy[
        [
            "menu_item_id",
            "scored_records",
            "actual_demand",
            "wMAPE",
            "MAE",
            "forecast_bias_pct"
        ]
    ].head(10).round(2).to_string(index=False)
)

print("\nTOP 10 MENU ITEMS WITH LOWEST FORECAST ERROR")

print(
    item_accuracy[
        [
            "menu_item_id",
            "scored_records",
            "actual_demand",
            "wMAPE",
            "MAE",
            "forecast_bias_pct"
        ]
    ].tail(10).round(2).to_string(index=False)
)


# ============================================================
# 6. CATEGORY-LEVEL ACCURACY
# ============================================================

category_accuracy = calculate_accuracy(
    df, "category"
)

category_accuracy.to_csv(
    OUTPUT_DIR / "q4_category_accuracy.csv",
    index=False
)

print("\n" + "=" * 65)
print("CATEGORY-LEVEL ACCURACY")
print("=" * 65)

print(
    category_accuracy[
        [
            "category",
            "scored_records",
            "actual_demand",
            "predicted_demand",
            "wMAPE",
            "MAE",
            "forecast_bias_pct"
        ]
    ].round(2).to_string(index=False)
)


# ============================================================
# 7. RESTAURANT-MENU ITEM ACCURACY
# ============================================================

combo_accuracy = (
    df.groupby(["restaurant_id", "menu_item_id"])
    .agg(
        scored_records=("actual_quantity", "size"),
        actual_demand=("actual_quantity", "sum"),
        predicted_demand=("predicted_quantity", "sum"),
        absolute_error=("absolute_error", "sum"),
        squared_error=("residual", lambda x: np.square(x).sum()),
    )
    .reset_index()
)

combo_accuracy["wMAPE"] = np.where(
    combo_accuracy["actual_demand"] != 0,
    combo_accuracy["absolute_error"]
    / combo_accuracy["actual_demand"] * 100,
    np.nan
)

combo_accuracy["MAE"] = (
    combo_accuracy["absolute_error"]
    / combo_accuracy["scored_records"]
)

combo_accuracy["RMSE"] = np.sqrt(
    combo_accuracy["squared_error"]
    / combo_accuracy["scored_records"]
)

combo_accuracy["forecast_bias_pct"] = np.where(
    combo_accuracy["actual_demand"] != 0,
    (
        combo_accuracy["predicted_demand"]
        - combo_accuracy["actual_demand"]
    ) / combo_accuracy["actual_demand"] * 100,
    np.nan
)

combo_accuracy["forecast_error"] = (
    combo_accuracy["predicted_demand"]
    - combo_accuracy["actual_demand"]
)

combo_accuracy = combo_accuracy.sort_values(
    "wMAPE", ascending=False
)

combo_accuracy.to_csv(
    OUTPUT_DIR / "q4_restaurant_item_accuracy.csv",
    index=False
)

print("\n" + "=" * 65)
print("TOP 15 RESTAURANT-MENU ITEM COMBINATIONS BY wMAPE")
print("=" * 65)

print(
    combo_accuracy[
        [
            "restaurant_id",
            "menu_item_id",
            "scored_records",
            "actual_demand",
            "wMAPE",
            "MAE",
            "forecast_bias_pct"
        ]
    ].head(15).round(2).to_string(index=False)
)


# ============================================================
# 8. SAVE SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("OUTPUT FILES")
print("=" * 65)

print("1. q4_restaurant_accuracy.csv")
print("2. q4_menu_item_accuracy.csv")
print("3. q4_category_accuracy.csv")
print("4. q4_restaurant_item_accuracy.csv")

print("\nQ4 accuracy diagnostics completed successfully.")
