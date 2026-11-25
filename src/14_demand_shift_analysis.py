
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "qsr_demand_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "model_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

# Retain observed quantities only.
# Missing quantities are NOT zero sales.

observed_df = df[df["quantity"].notna()].copy()

print("Total dataset rows:", len(df))
print("Observed quantity rows:", len(observed_df))
print("Missing quantity rows:", df["quantity"].isna().sum())

# --------------------------------------------------
# 3. DEFINE ANALYSIS PERIODS
# --------------------------------------------------

training_df = observed_df[
    (observed_df["date"] >= "2021-01-01") &
    (observed_df["date"] < "2025-07-01")
].copy()

validation_df = observed_df[
    (observed_df["date"] >= "2025-07-01") &
    (observed_df["date"] < "2025-10-01")
].copy()

print("\nPERIOD RECORD COUNTS")
print("Training:", len(training_df))
print("Validation:", len(validation_df))


# --------------------------------------------------
# 3. SEASONALLY ADJUSTED DEMAND COMPARISON
# --------------------------------------------------

# Historical Q3: July-September 2021 to 2024
historical_q3 = observed_df[
    (observed_df["date"].dt.year >= 2021) &
    (observed_df["date"].dt.year <= 2024) &
    (observed_df["date"].dt.month.isin([7, 8, 9]))
].copy()

# Validation Q3: July-September 2025
validation_q3 = observed_df[
    (observed_df["date"] >= "2025-07-01") &
    (observed_df["date"] < "2025-10-01")
].copy()

group_columns = ["restaurant_id", "menu_item_id"]

historical_summary = (
    historical_q3
    .groupby(group_columns)
    .agg(
        historical_q3_average=("quantity", "mean"),
        historical_q3_total=("quantity", "sum"),
        historical_q3_observations=("quantity", "count")
    )
    .reset_index()
)

validation_summary = (
    validation_q3
    .groupby(group_columns)
    .agg(
        validation_q3_average=("quantity", "mean"),
        validation_q3_total=("quantity", "sum"),
        validation_q3_observations=("quantity", "count")
    )
    .reset_index()
)

demand_shift = historical_summary.merge(
    validation_summary,
    on=group_columns,
    how="inner"
)

demand_shift["q3_demand_change_pct"] = (
    (
        demand_shift["validation_q3_average"]
        - demand_shift["historical_q3_average"]
    )
    / demand_shift["historical_q3_average"]
) * 100

item_lookup = (
    df[
        ["menu_item_id", "menu_item_name", "category"]
    ]
    .drop_duplicates("menu_item_id")
)

demand_shift = demand_shift.merge(
    item_lookup,
    on="menu_item_id",
    how="left"
)

# --------------------------------------------------
# 4. MONTHLY DEMAND TRENDS: 2021-2025
# --------------------------------------------------

monthly_df = observed_df[
    (observed_df["date"] >= "2021-01-01") &
    (observed_df["date"] < "2025-10-01")
].copy()

monthly_df["year"] = monthly_df["date"].dt.year
monthly_df["month"] = monthly_df["date"].dt.month

monthly_demand = (
    monthly_df
    .groupby(
        ["restaurant_id", "menu_item_id", "year", "month"]
    )
    .agg(
        monthly_average=("quantity", "mean"),
        monthly_total=("quantity", "sum"),
        monthly_observations=("quantity", "count")
    )
    .reset_index()
)

monthly_demand = monthly_demand.merge(
    item_lookup,
    on="menu_item_id",
    how="left"
)

# --------------------------------------------------
# 5. DISPLAY SEASONALLY ADJUSTED RESULTS
# --------------------------------------------------

print("\nTOP 15 Q3 DEMAND INCREASES")

print(
    demand_shift.sort_values(
        "q3_demand_change_pct",
        ascending=False
    )[
        [
            "restaurant_id",
            "menu_item_name",
            "category",
            "historical_q3_average",
            "validation_q3_average",
            "q3_demand_change_pct"
        ]
    ].head(15).to_string(index=False)
)

print("\nTOP 15 Q3 DEMAND DECLINES")

print(
    demand_shift.sort_values(
        "q3_demand_change_pct",
        ascending=True
    )[
        [
            "restaurant_id",
            "menu_item_name",
            "category",
            "historical_q3_average",
            "validation_q3_average",
            "q3_demand_change_pct"
        ]
    ].head(15).to_string(index=False)
)

print("\nSEASONALLY ADJUSTED DEMAND SHIFT SUMMARY")

print(
    demand_shift["q3_demand_change_pct"].describe()
)

print(
    "Combinations with increases:",
    (demand_shift["q3_demand_change_pct"] > 0).sum()
)

print(
    "Combinations with declines:",
    (demand_shift["q3_demand_change_pct"] < 0).sum()
)

# --------------------------------------------------
# 6. EXPORT RESULTS
# --------------------------------------------------

demand_shift.to_csv(
    OUTPUT_DIR / "seasonally_adjusted_q3_demand_shift.csv",
    index=False
)

monthly_demand.to_csv(
    OUTPUT_DIR / "monthly_restaurant_item_demand_2021_2025.csv",
    index=False
)

print("\nSeasonal demand analysis completed.")


q3_output = OUTPUT_DIR / "seasonally_adjusted_q3_demand_shift.csv"
monthly_output = OUTPUT_DIR / "monthly_restaurant_item_demand_2021_2025.csv"

demand_shift.to_csv(q3_output, index=False)
monthly_demand.to_csv(monthly_output, index=False)

print("\nSeasonal demand analysis completed.")

print("Saved:")
print(q3_output)
print(monthly_output)