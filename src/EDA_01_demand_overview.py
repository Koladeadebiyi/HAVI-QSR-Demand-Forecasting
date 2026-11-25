
# ============================================================
# HAVI DEMAND FORECASTING
# EXERCISE 1: OVERALL DEMAND ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

DATA_PATH = "data/qsr_demand_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

print("Original dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. DEFINE OUR ANALYSIS PERIOD
# ------------------------------------------------------------

# Exclude October-December 2025 completely
analysis_df = df[
    df["date"] < "2025-10-01"
].copy()

print("Analysis dataset shape:", analysis_df.shape)

# We are NOT filling or deleting missing quantities.
print(
    "Missing quantities:",
    analysis_df["quantity"].isna().sum()
)

# ------------------------------------------------------------
# 3. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

os.makedirs("outputs/eda", exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 4. GENERAL DEMAND STATISTICS
# ------------------------------------------------------------

print("\n========== DEMAND SUMMARY ==========")

print(
    analysis_df["quantity"].describe()
)

print(
    "\nTotal observed units sold:",
    analysis_df["quantity"].sum()
)

print(
    "\nObserved sales records:",
    analysis_df["quantity"].count()
)

print(
    "\nZero-sales records:",
    (analysis_df["quantity"] == 0).sum()
)

# ------------------------------------------------------------
# 5. DAILY DEMAND TREND
# ------------------------------------------------------------

# Average observed quantity across all restaurant-item
# records for each day.

daily_demand = (
    analysis_df.groupby("date")["quantity"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(15, 5))

sns.lineplot(
    data=daily_demand,
    x="date",
    y="quantity"
)

plt.title("Average Daily Demand Across All Restaurants and Items")
plt.xlabel("Date")
plt.ylabel("Average Quantity Sold")

plt.tight_layout()

plt.savefig(
    "outputs/eda/01_daily_demand.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 6. MONTHLY DEMAND PATTERN
# ------------------------------------------------------------

analysis_df["year_month"] = (
    analysis_df["date"].dt.to_period("M").astype(str)
)

monthly_demand = (
    analysis_df.groupby("year_month")["quantity"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(16, 5))

sns.lineplot(
    data=monthly_demand,
    x="year_month",
    y="quantity",
    marker="o"
)

plt.title("Average Demand by Month")
plt.xlabel("Year-Month")
plt.ylabel("Average Quantity Sold per Record")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "outputs/eda/02_monthly_demand.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 7. DEMAND DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    data=analysis_df,
    x="quantity",
    bins=50,
    kde=True
)

plt.title("Distribution of Menu Item Demand")
plt.xlabel("Quantity Sold")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/eda/03_demand_distribution.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 8. RESTAURANT DEMAND
# ------------------------------------------------------------

restaurant_demand = (
    analysis_df.groupby(
        ["restaurant_id", "restaurant_name"]
    )["quantity"]
    .mean()
    .reset_index()
    .sort_values("quantity", ascending=False)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=restaurant_demand,
    x="restaurant_id",
    y="quantity"
)

plt.title("Average Demand by Restaurant")
plt.xlabel("Restaurant")
plt.ylabel("Average Quantity Sold per Record")

plt.tight_layout()

plt.savefig(
    "outputs/eda/04_restaurant_demand.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 9. MENU ITEM DEMAND
# ------------------------------------------------------------

item_demand = (
    analysis_df.groupby(
        ["menu_item_id", "menu_item_name"]
    )["quantity"]
    .mean()
    .reset_index()
    .sort_values("quantity", ascending=False)
)

top_items = item_demand.head(15)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_items,
    x="quantity",
    y="menu_item_name"
)

plt.title("Top 15 Menu Items by Average Demand")
plt.xlabel("Average Quantity Sold per Record")
plt.ylabel("Menu Item")

plt.tight_layout()

plt.savefig(
    "outputs/eda/05_top_menu_items.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 10. CATEGORY DEMAND
# ------------------------------------------------------------

category_demand = (
    analysis_df.groupby("category")["quantity"]
    .mean()
    .reset_index()
    .sort_values("quantity", ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=category_demand,
    x="quantity",
    y="category"
)

plt.title("Average Demand by Menu Category")
plt.xlabel("Average Quantity Sold per Record")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    "outputs/eda/06_category_demand.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 11. EXPORT SUMMARY TABLES
# ------------------------------------------------------------

restaurant_demand.to_csv(
    "outputs/eda/restaurant_demand.csv",
    index=False
)

item_demand.to_csv(
    "outputs/eda/item_demand.csv",
    index=False
)

category_demand.to_csv(
    "outputs/eda/category_demand.csv",
    index=False
)

monthly_demand.to_csv(
    "outputs/eda/monthly_demand.csv",
    index=False
)

# ------------------------------------------------------------
# 12. DISPLAY TOP RESULTS
# ------------------------------------------------------------

print("\n========== TOP RESTAURANTS ==========")
print(restaurant_demand.head(10).to_string(index=False))

print("\n========== TOP MENU ITEMS ==========")
print(item_demand.head(10).to_string(index=False))

print("\n========== CATEGORY DEMAND ==========")
print(category_demand.to_string(index=False))

print("\nEDA EXERCISE 1 COMPLETED!")
print("Charts and summary tables saved in outputs/eda/")
