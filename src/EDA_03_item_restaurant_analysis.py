
# ============================================================
# HAVI QSR DEMAND FORECASTING
# EDA EXERCISE 3
# Restaurant and Menu Item Demand Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_PATH = Path("data/qsr_demand_dataset.csv")

OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

# Exclude the competition forecast period.
# We must not use October-December 2025 quantities
# in exploratory target analysis.

analysis_df = df[
    df["date"] < "2025-10-01"
].copy()

print("Analysis period:",
      analysis_df["date"].min(),
      "to",
      analysis_df["date"].max())

# ------------------------------------------------------------
# 3. RESTAURANT-ITEM DEMAND SUMMARY
# ------------------------------------------------------------

# Group the data by restaurant and menu item.
# Missing quantities are excluded from the mean,
# but they are NOT replaced with zero.

item_restaurant_summary = (
    analysis_df.groupby(
        [
            "restaurant_id",
            "restaurant_name",
            "menu_item_id",
            "menu_item_name",
            "category"
        ],
        dropna=False
    )
    .agg(
        average_daily_quantity=("quantity", "mean"),
        median_daily_quantity=("quantity", "median"),
        total_observed_quantity=("quantity", "sum"),
        standard_deviation=("quantity", "std"),
        observed_records=("quantity", "count"),
        missing_records=("quantity", lambda x: x.isna().sum()),
        total_records=("quantity", "size")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 4. CALCULATE ZERO-SALES RATE
# ------------------------------------------------------------

# Count only records where quantity is available.
# Missing quantities must not be treated as zero sales.

zero_sales = (
    analysis_df.groupby(
        ["restaurant_id", "menu_item_id"]
    )["quantity"]
    .agg(
        total_observed="count",
        zero_count=lambda x: (x == 0).sum()
    )
    .reset_index()
)

# Calculate zero-sales rate using observed records only.

zero_sales["zero_sales_rate"] = (
    zero_sales["zero_count"]
    / zero_sales["total_observed"]
)

# Merge zero-sales rate into the main summary.

item_restaurant_summary = item_restaurant_summary.merge(
    zero_sales[
        ["restaurant_id", "menu_item_id", "zero_sales_rate"]
    ],
    on=["restaurant_id", "menu_item_id"],
    how="left"
)


# Coefficient of variation:
# Measures variability relative to average demand.
# Higher values indicate greater relative variability.

item_restaurant_summary["coefficient_of_variation"] = (
    item_restaurant_summary["standard_deviation"]
    / item_restaurant_summary["average_daily_quantity"]
)

# Save summary.
item_restaurant_summary.to_csv(
    OUTPUT_DIR / "restaurant_item_demand_summary.csv",
    index=False
)

print("\nTotal restaurant-item combinations:",
      len(item_restaurant_summary))

print("\nSummary statistics:")
print(
    item_restaurant_summary[
        [
            "average_daily_quantity",
            "median_daily_quantity",
            "standard_deviation",
            "coefficient_of_variation",
            "zero_sales_rate"
        ]
    ].describe()
)

print("\nEDA EXERCISE 3 - STAGE 1 COMPLETED")



# ============================================================
# EXERCISE 3 - STAGE 2
# ANALYSING HIGH AND LOW DEMAND COMBINATIONS
# ============================================================

# ------------------------------------------------------------
# 1. TOP 15 COMBINATIONS BY AVERAGE DEMAND
# ------------------------------------------------------------

top_demand = (
    item_restaurant_summary
    .nlargest(15, "average_daily_quantity")
    .copy()
)

top_demand["combination"] = (
    top_demand["restaurant_id"]
    + " - "
    + top_demand["menu_item_name"]
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=top_demand,
    x="average_daily_quantity",
    y="combination"
)

plt.title("Top 15 Restaurant-Item Combinations by Average Demand")
plt.xlabel("Average Daily Quantity Sold")
plt.ylabel("Restaurant - Menu Item")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "13_top_15_restaurant_item_demand.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 2. BOTTOM 15 COMBINATIONS BY AVERAGE DEMAND
# ------------------------------------------------------------

bottom_demand = (
    item_restaurant_summary
    .nsmallest(15, "average_daily_quantity")
    .copy()
)

bottom_demand["combination"] = (
    bottom_demand["restaurant_id"]
    + " - "
    + bottom_demand["menu_item_name"]
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=bottom_demand,
    x="average_daily_quantity",
    y="combination"
)

plt.title("Bottom 15 Restaurant-Item Combinations by Average Demand")
plt.xlabel("Average Daily Quantity Sold")
plt.ylabel("Restaurant - Menu Item")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "14_bottom_15_restaurant_item_demand.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 3. TOP 15 COMBINATIONS BY RELATIVE VARIABILITY
# ------------------------------------------------------------

top_volatility = (
    item_restaurant_summary
    .nlargest(15, "coefficient_of_variation")
    .copy()
)

top_volatility["combination"] = (
    top_volatility["restaurant_id"]
    + " - "
    + top_volatility["menu_item_name"]
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=top_volatility,
    x="coefficient_of_variation",
    y="combination"
)

plt.title("Top 15 Restaurant-Item Combinations by Demand Variability")
plt.xlabel("Coefficient of Variation")
plt.ylabel("Restaurant - Menu Item")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "15_top_15_demand_volatility.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 4. TOP 15 COMBINATIONS BY ZERO-SALES RATE
# ------------------------------------------------------------

top_zero_sales = (
    item_restaurant_summary
    .nlargest(15, "zero_sales_rate")
    .copy()
)

top_zero_sales["combination"] = (
    top_zero_sales["restaurant_id"]
    + " - "
    + top_zero_sales["menu_item_name"]
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=top_zero_sales,
    x="zero_sales_rate",
    y="combination"
)

plt.title("Top 15 Restaurant-Item Combinations by Zero-Sales Rate")
plt.xlabel("Zero-Sales Rate")
plt.ylabel("Restaurant - Menu Item")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "16_top_15_zero_sales_rate.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 5. EXPORT THE FOUR SUMMARIES
# ------------------------------------------------------------

top_demand.to_csv(
    OUTPUT_DIR / "top_15_demand.csv",
    index=False
)

bottom_demand.to_csv(
    OUTPUT_DIR / "bottom_15_demand.csv",
    index=False
)

top_volatility.to_csv(
    OUTPUT_DIR / "top_15_volatility.csv",
    index=False
)

top_zero_sales.to_csv(
    OUTPUT_DIR / "top_15_zero_sales.csv",
    index=False
)

print("\nEXERCISE 3 - STAGE 2 COMPLETED")
print("Four visualizations and four summary files saved.")



# ============================================================
# EXERCISE 3 - STAGE 3
# TIME-SERIES BEHAVIOUR OF SELECTED COMBINATIONS
# ============================================================


# ------------------------------------------------------------
# 1. SELECT RESTAURANT-ITEM COMBINATIONS
# ------------------------------------------------------------

selected_pairs = [
    ("R01", "Soft Drink Regular"),
    ("R13", "Brownie"),
    ("R12", "Oatmeal Cup"),
    ("R03", "Oatmeal Cup"),
    ("R01", "French Fries Regular")
]

# Build a filter using restaurant and item names.

selection_keys = {
    f"{restaurant}|{item}"
    for restaurant, item in selected_pairs
}

row_keys = (
    analysis_df["restaurant_id"].astype(str)
    + "|"
    + analysis_df["menu_item_name"].astype(str)
)

selected_data = analysis_df[
    row_keys.isin(selection_keys)
].copy()

selected_data["combination"] = (
    selected_data["restaurant_id"]
    + " - "
    + selected_data["menu_item_name"]
)

print("\nSelected combinations:")

print(
    selected_data[
        ["restaurant_id", "menu_item_name", "menu_item_id"]
    ].drop_duplicates()
)

print("\nNumber of selected combinations:",
      selected_data["combination"].nunique())


# IMPORTANT:
# Confirm the menu_item_id values against your dataset.
# The IDs above are placeholders if your actual IDs differ.
# We will use menu_item_name to verify them below.

selected_data = analysis_df[
    analysis_df.apply(
        lambda row: (
            row["restaurant_id"],
            row["menu_item_name"]
        ) in [
            ("R01", "Soft Drink Regular"),
            ("R13", "Brownie"),
            ("R12", "Oatmeal Cup"),
            ("R03", "Oatmeal Cup"),
            ("R01", "French Fries Regular")
        ],
        axis=1
    )
].copy()

# Create a readable combination label.

selected_data["combination"] = (
    selected_data["restaurant_id"]
    + " - "
    + selected_data["menu_item_name"]
)

print("\nSelected combinations:")
print(
    selected_data[
        ["restaurant_id", "menu_item_name", "menu_item_id"]
    ].drop_duplicates()
)

# ------------------------------------------------------------
# 2. DAILY DEMAND TREND
# ------------------------------------------------------------

for combination in selected_data["combination"].unique():

    subset = selected_data[
        selected_data["combination"] == combination
    ].sort_values("date")

    plt.figure(figsize=(16, 5))

    plt.plot(
        subset["date"],
        subset["quantity"],
        linewidth=0.8
    )

    plt.title(
        f"Daily Demand Trend - {combination}"
    )

    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")

    plt.tight_layout()

    filename = (
        combination.replace(" ", "_")
        .replace("-", "")
        .lower()
    )

    plt.savefig(
        OUTPUT_DIR / f"17_daily_trend_{filename}.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------------------
# 3. MONTHLY DEMAND PROFILE
# ------------------------------------------------------------

selected_data["month"] = (
    selected_data["date"].dt.month
)

monthly_profile = (
    selected_data.groupby(
        ["combination", "month"]
    )
    .agg(
        average_monthly_demand=("quantity", "mean"),
        observed_records=("quantity", "count"),
        missing_records=("quantity", lambda x: x.isna().sum())
    )
    .reset_index()
)

plt.figure(figsize=(14, 7))

sns.lineplot(
    data=monthly_profile,
    x="month",
    y="average_monthly_demand",
    hue="combination",
    marker="o"
)

plt.title(
    "Average Monthly Demand by Selected Restaurant-Item"
)

plt.xlabel("Calendar Month")
plt.ylabel("Average Daily Quantity Sold")

plt.xticks(range(1, 13))

plt.legend(
    title="Restaurant - Menu Item",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "22_monthly_demand_profiles.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 4. YEAR-OVER-YEAR MONTHLY COMPARISON
# ------------------------------------------------------------

selected_data["year"] = (
    selected_data["date"].dt.year
)

yearly_monthly = (
    selected_data.groupby(
        ["combination", "year", "month"]
    )
    .agg(
        average_daily_demand=("quantity", "mean"),
        observed_records=("quantity", "count")
    )
    .reset_index()
)

for combination in yearly_monthly["combination"].unique():

    subset = yearly_monthly[
        yearly_monthly["combination"] == combination
    ]

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=subset,
        x="month",
        y="average_daily_demand",
        hue="year",
        marker="o",
        palette="tab10"
    )

    plt.title(
        f"Year-over-Year Monthly Demand - {combination}"
    )

    plt.xlabel("Month")
    plt.ylabel("Average Daily Quantity Sold")

    plt.xticks(range(1, 13))

    plt.legend(title="Year")

    plt.tight_layout()

    filename = (
        combination.replace(" ", "_")
        .replace("-", "")
        .lower()
    )

    plt.savefig(
        OUTPUT_DIR / f"yearly_monthly_{filename}.png",
        dpi=300
    )

    plt.close()

# ------------------------------------------------------------
# 5. EXPORT ANALYSIS TABLES
# ------------------------------------------------------------

monthly_profile.to_csv(
    OUTPUT_DIR / "selected_items_monthly_profile.csv",
    index=False
)

yearly_monthly.to_csv(
    OUTPUT_DIR / "selected_items_yearly_monthly.csv",
    index=False
)

print("\nEXERCISE 3 - STAGE 3 COMPLETED")

print("\nDaily trend charts generated for:")
print(selected_data["combination"].unique())

print("\nMonthly and yearly comparison tables exported.")


