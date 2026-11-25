
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "model_evaluation"
    / "monthly_restaurant_item_demand_2021_2025.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD MONTHLY DEMAND DATA
# --------------------------------------------------

monthly_df = pd.read_csv(INPUT_FILE)

print("Loaded monthly demand data:", monthly_df.shape)

# --------------------------------------------------
# 3. SEPARATE HISTORICAL AND 2025 DATA
# --------------------------------------------------

historical = monthly_df[
    monthly_df["year"].between(2021, 2024)
].copy()

current_2025 = monthly_df[
    monthly_df["year"] == 2025
].copy()

# Historical average for each restaurant-item and calendar month
historical_baseline = (
    historical
    .groupby(["restaurant_id", "menu_item_id", "month"])
    .agg(
        historical_monthly_average=("monthly_average", "mean")
    )
    .reset_index()
)

# 2025 monthly demand
current_summary = current_2025[
    [
        "restaurant_id",
        "menu_item_id",
        "month",
        "monthly_average",
        "monthly_observations"
    ]
].rename(
    columns={
        "monthly_average": "2025_monthly_average",
        "monthly_observations": "2025_observations"
    }
)

# --------------------------------------------------
# 4. COMPARE 2025 WITH HISTORICAL MONTHLY BASELINE
# --------------------------------------------------

comparison = current_summary.merge(
    historical_baseline,
    on=["restaurant_id", "menu_item_id", "month"],
    how="inner"
)

comparison["demand_change_pct"] = (
    (
        comparison["2025_monthly_average"]
        - comparison["historical_monthly_average"]
    )
    / comparison["historical_monthly_average"]
) * 100

# --------------------------------------------------
# 5. SUMMARISE MONTHLY DEMAND CHANGES
# --------------------------------------------------

monthly_summary = (
    comparison
    .groupby("month")
    .agg(
        average_demand_change_pct=("demand_change_pct", "mean"),
        median_demand_change_pct=("demand_change_pct", "median"),
        combinations_increasing=(
            "demand_change_pct",
            lambda x: (x > 0).sum()
        ),
        combinations_declining=(
            "demand_change_pct",
            lambda x: (x < 0).sum()
        ),
        combinations_compared=("demand_change_pct", "count")
    )
    .reset_index()
)

month_names = {
    1: "January", 2: "February", 3: "March",
    4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September"
}

monthly_summary["month_name"] = monthly_summary["month"].map(month_names)

monthly_summary = monthly_summary[
    [
        "month",
        "month_name",
        "average_demand_change_pct",
        "median_demand_change_pct",
        "combinations_increasing",
        "combinations_declining",
        "combinations_compared"
    ]
]

# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

print("\n2025 MONTHLY DEMAND CHANGE VS HISTORICAL SAME-MONTH AVERAGE")

print(monthly_summary.to_string(index=False))

# --------------------------------------------------
# 7. EXPORT RESULTS
# --------------------------------------------------

comparison.to_csv(
    OUTPUT_DIR / "monthly_restaurant_item_shift_2025.csv",
    index=False
)

monthly_summary.to_csv(
    OUTPUT_DIR / "monthly_demand_shift_summary_2025.csv",
    index=False
)

print("\nMonthly shift analysis completed.")
print("Saved:", OUTPUT_DIR / "monthly_restaurant_item_shift_2025.csv")
print("Saved:", OUTPUT_DIR / "monthly_demand_shift_summary_2025.csv")
