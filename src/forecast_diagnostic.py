
import pandas as pd
import os

# 1. Load the complete dataset
DATA_PATH = "data/qsr_demand_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Create output directory
os.makedirs("data/processed", exist_ok=True)

# 2. Basic date analysis
print("\n--- DATE ANALYSIS ---")
print("Earliest date:", df["date"].min())
print("Latest date:", df["date"].max())
print("Unique dates:", df["date"].nunique())

# 3. Missing target analysis
print("\n--- MISSING QUANTITY ANALYSIS ---")

missing = df[df["quantity"].isna()]

print("Missing quantity records:", len(missing))
print("Missing percentage:", round(
    df["quantity"].isna().mean() * 100, 2
))

print("\nMissing quantities by year:")
print(missing.groupby(
    missing["date"].dt.year
).size())

print("\nMissing quantities by month:")
print(missing.groupby(
    missing["date"].dt.to_period("M")
).size())

print("\nMissing quantities by restaurant:")
print(missing.groupby("restaurant_id").size())

print("\nMissing quantities by menu item:")
print(missing.groupby("menu_item_id").size())

# 4. Check daily record completeness
daily_records = df.groupby("date").size()

print("\n--- DAILY RECORD COUNTS ---")
print(daily_records.describe())

print("Days with records other than 750:")
print(daily_records[daily_records != 750])

# 5. Missing target by restaurant-item combination
missing_combinations = (
    missing.groupby(["restaurant_id", "menu_item_id"])
    .size()
    .sort_values(ascending=False)
)

print("\n--- MISSING RESTAURANT-ITEM COMBINATIONS ---")
print(missing_combinations.head(20))

# 6. Quantity distribution
print("\n--- QUANTITY DISTRIBUTION ---")
print(df["quantity"].describe())

print("\nZero sales records:", (df["quantity"] == 0).sum())

print("\nNegative sales records:", (df["quantity"] < 0).sum())

# 7. Missing quantity date range
if not missing.empty:
    print("\nFirst missing target date:", missing["date"].min())
    print("Last missing target date:", missing["date"].max())

# 8. Save missing target records for investigation
missing.to_csv(
    "data/processed/missing_quantity_records.csv",
    index=False
)

# 9. Save diagnostic report
with open(
    "data/processed/forecast_diagnostics.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("HAVi FORECASTING DIAGNOSTICS\n")
    f.write("=" * 50)

    f.write(f"\nDataset shape: {df.shape}")
    f.write(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    f.write(f"\nUnique dates: {df['date'].nunique()}")

    f.write("\n\nMissing quantities by year:\n")
    f.write(missing.groupby(missing["date"].dt.year).size().to_string())

    f.write("\n\nMissing quantities by month:\n")
    f.write(missing.groupby(missing["date"].dt.to_period("M")).size().to_string())

    f.write("\n\nMissing quantities by restaurant:\n")
    f.write(missing.groupby("restaurant_id").size().to_string())

    f.write("\n\nMissing quantities by menu item:\n")
    f.write(missing.groupby("menu_item_id").size().to_string())

    f.write("\n\nMissing restaurant-item combinations:\n")
    f.write(missing_combinations.to_string())

    f.write("\n\nDaily record count summary:\n")
    f.write(daily_records.describe().to_string())

    f.write("\n\nQuantity summary:\n")
    f.write(df["quantity"].describe().to_string())

print("\nFORECAST DIAGNOSTICS COMPLETED!")
