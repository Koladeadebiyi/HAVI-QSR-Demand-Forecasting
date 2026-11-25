
import pandas as pd

# Load full dataset
df = pd.read_csv("data/qsr_demand_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

# Define forecasting period
forecast_start = "2025-10-01"
forecast_end = "2025-12-31"

forecast_period = df[
    (df["date"] >= forecast_start) &
    (df["date"] <= forecast_end)
].copy()

# 1. Forecast period availability
print("\n1. FORECAST PERIOD AVAILABILITY")

print("Total records:", len(forecast_period))
print("Available quantities:",
      forecast_period["quantity"].notna().sum())

print("Missing quantities:",
      forecast_period["quantity"].isna().sum())

print("Percentage with observed quantities:",
      round(forecast_period["quantity"].notna().mean() * 100, 2))

# 2. Missing values by date
print("\n2. MISSING QUANTITIES BY DATE")

missing_by_date = (
    df[df["quantity"].isna()]
    .groupby("date")
    .size()
)

print(missing_by_date.describe())

print("\nDates with most missing quantities:")
print(missing_by_date.sort_values(
    ascending=False
).head(15))

# 3. Missing quantities by restaurant and year
print("\n3. MISSING BY RESTAURANT AND YEAR")

missing = df[df["quantity"].isna()].copy()

missing["year"] = missing["date"].dt.year

print(
    missing.groupby(
        ["restaurant_id", "year"]
    ).size().unstack(fill_value=0)
)

# 4. Missing quantities by restaurant-item
print("\n4. MISSING BY RESTAURANT AND MENU ITEM")

missing_matrix = missing.pivot_table(
    index="restaurant_id",
    columns="menu_item_id",
    values="date",
    aggfunc="count",
    fill_value=0
)

print(missing_matrix)

# 5. Zero sales analysis
print("\n5. ZERO SALES ANALYSIS")

zero_sales = (df["quantity"] == 0).sum()

print("Zero sales records:", zero_sales)

print(
    "Zero sales percentage:",
    round((zero_sales / df["quantity"].notna().sum()) * 100, 2)
)

# 6. Save results
forecast_period.to_csv(
    "data/processed/forecast_period_check.csv",
    index=False
)

missing_matrix.to_csv(
    "data/processed/missing_quantity_matrix.csv"
)

print("\nINVESTIGATION COMPLETED!")
