
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. LOAD ORIGINAL DATASET AND FORECAST
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "qsr_demand_dataset.csv"

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
    / "q4_2025_daily_forecast.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

predictions = pd.read_csv(PREDICTION_PATH)

predictions["date"] = pd.to_datetime(predictions["date"])


# ============================================================
# 2. EXTRACT Q4 ACTUAL RECORDS
# ============================================================

q4_actuals = df[
    (df["date"] >= "2025-10-01")
    & (df["date"] <= "2025-12-31")
].copy()

print("=" * 60)
print("ORIGINAL Q4 DATA AVAILABILITY")
print("=" * 60)

print("Total Q4 records:", len(q4_actuals))

print(
    "Available actual quantities:",
    q4_actuals["quantity"].notna().sum()
)

print(
    "Missing actual quantities:",
    q4_actuals["quantity"].isna().sum()
)

print(
    "Actual quantity availability (%):",
    round(q4_actuals["quantity"].notna().mean() * 100, 2)
)

print(
    "Actual demand total (available records):",
    q4_actuals["quantity"].sum()
)


# ============================================================
# 3. INVESTIGATE MISSING ACTUAL QUANTITIES
# ============================================================

missing_actuals = q4_actuals[
    q4_actuals["quantity"].isna()
].copy()

print("\n" + "=" * 60)
print("MISSING Q4 ACTUALS")
print("=" * 60)

print("Missing records by month:")
print(
    missing_actuals.groupby(
        missing_actuals["date"].dt.to_period("M")
    ).size()
)

print("\nMissing records by restaurant:")
print(
    missing_actuals.groupby("restaurant_id")
    .size()
    .sort_values(ascending=False)
)

print("\nMissing records by category:")
print(
    missing_actuals.groupby("category")
    .size()
    .sort_values(ascending=False)
)


# ============================================================
# 4. MERGE ACTUALS WITH Q4 PREDICTIONS
# ============================================================

actual_columns = [
    "date",
    "restaurant_id",
    "menu_item_id",
    "quantity",
]

actuals = q4_actuals[actual_columns].copy()

comparison = predictions.merge(
    actuals,
    on=["date", "restaurant_id", "menu_item_id"],
    how="left",
    validate="one_to_one",
)

print("\n" + "=" * 60)
print("FORECAST AND ACTUAL MATCHING")
print("=" * 60)

print("Forecast rows:", len(predictions))

print("Matched rows:", len(comparison))

print(
    "Rows with actual quantities:",
    comparison["quantity"].notna().sum()
)

print(
    "Rows without actual quantities:",
    comparison["quantity"].isna().sum()
)


# ============================================================
# 5. EVALUATE ONLY RECORDS WITH ACTUAL QUANTITIES
# ============================================================

scored = comparison.dropna(subset=["quantity"]).copy()

scored["error"] = (
    scored["predicted_quantity"] - scored["quantity"]
)

scored["absolute_error"] = scored["error"].abs()

total_actual = scored["quantity"].abs().sum()

wmape = (
    scored["absolute_error"].sum() / total_actual * 100
)

mae = scored["absolute_error"].mean()

rmse = np.sqrt(
    np.mean(scored["error"] ** 2)
)

bias = (
    scored["error"].sum() / total_actual * 100
)

print("\n" + "=" * 60)
print("Q4 FORECAST ACCURACY ON AVAILABLE ACTUALS")
print("=" * 60)

print("Scored observations:", len(scored))

print("Actual demand:", round(scored["quantity"].sum(), 2))

print(
    "Predicted demand on scored observations:",
    round(scored["predicted_quantity"].sum(), 2)
)

print("wMAPE:", round(wmape, 4), "%")

print("MAE:", round(mae, 4))

print("RMSE:", round(rmse, 4))

print("Bias:", round(bias, 4), "%")


# ============================================================
# 6. MONTHLY ACTUAL VS PREDICTED DEMAND
# ============================================================

monthly = (
    scored.groupby(
        scored["date"].dt.to_period("M")
    )
    .agg(
        actual_demand=("quantity", "sum"),
        predicted_demand=("predicted_quantity", "sum"),
        scored_records=("quantity", "count"),
    )
    .reset_index()
)

monthly["forecast_error"] = (
    monthly["predicted_demand"] - monthly["actual_demand"]
)

monthly["error_pct"] = (
    monthly["forecast_error"] / monthly["actual_demand"] * 100
)

print("\nMONTHLY ACTUAL VS PREDICTED")
print(monthly.round(2).to_string(index=False))


# ============================================================
# 7. SAVE COMPARISON FILES
# ============================================================

comparison.to_csv(
    OUTPUT_DIR / "q4_2025_actual_vs_forecast.csv",
    index=False,
)

monthly.to_csv(
    OUTPUT_DIR / "q4_2025_monthly_actual_vs_forecast.csv",
    index=False,
)

missing_actuals.to_csv(
    OUTPUT_DIR / "q4_2025_missing_actuals.csv",
    index=False,
)

print("\nComparison files saved successfully.")
