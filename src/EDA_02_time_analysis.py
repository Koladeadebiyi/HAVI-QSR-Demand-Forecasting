
# ============================================================
# HAVI DEMAND FORECASTING
# EXERCISE 2: TIME-BASED DEMAND ANALYSIS
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

df["date"] = pd.to_datetime(df["date"])

print("Original dataset:", df.shape)

# ------------------------------------------------------------
# 2. EXCLUDE THE COMPETITION FORECAST PERIOD
# ------------------------------------------------------------

analysis_df = df[
    df["date"] < "2025-10-01"
].copy()

print("Analysis dataset:", analysis_df.shape)

print(
    "Missing quantities:",
    analysis_df["quantity"].isna().sum()
)

# We are not filling or deleting missing quantities.

# ------------------------------------------------------------
# 3. CREATE DATE FEATURES
# ------------------------------------------------------------

analysis_df["day_name"] = analysis_df["date"].dt.day_name()

analysis_df["day_num"] = analysis_df["date"].dt.dayofweek

analysis_df["month_num"] = analysis_df["date"].dt.month

analysis_df["month_name"] = analysis_df["date"].dt.month_name()

analysis_df["quarter"] = (
    "Q" + analysis_df["date"].dt.quarter.astype(str)
)

analysis_df["week_type"] = (
    analysis_df["day_num"]
    .apply(lambda x: "Weekend" if x >= 5 else "Weekday")
)

# Explicit weekday ordering
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

quarter_order = ["Q1", "Q2", "Q3", "Q4"]

# ------------------------------------------------------------
# 4. CREATE OUTPUT FOLDER
# ------------------------------------------------------------

os.makedirs("outputs/eda", exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 5. CREATE DAILY DEMAND TABLE
# ------------------------------------------------------------

# Important:
# Sum observed quantities for each date.
# Missing quantities remain missing in the source.
# Pandas sum skips missing values, so we also track
# the number of observed quantities per day.

daily_summary = (
    analysis_df.groupby("date")
    .agg(
        daily_observed_sales=("quantity", "sum"),
        observed_records=("quantity", "count"),
        missing_records=("quantity", lambda x: x.isna().sum())
    )
    .reset_index()
)

daily_summary["day_name"] = (
    daily_summary["date"].dt.day_name()
)

daily_summary["day_num"] = (
    daily_summary["date"].dt.dayofweek
)

daily_summary["week_type"] = (
    daily_summary["day_num"]
    .apply(lambda x: "Weekend" if x >= 5 else "Weekday")
)

daily_summary["month_name"] = (
    daily_summary["date"].dt.month_name()
)

daily_summary["month_num"] = (
    daily_summary["date"].dt.month
)

daily_summary["quarter"] = (
    "Q" + daily_summary["date"].dt.quarter.astype(str)
)

# ------------------------------------------------------------
# 6. WEEKDAY DEMAND ANALYSIS
# ------------------------------------------------------------

weekday_demand = (
    daily_summary.groupby("day_name")
    .agg(
        average_daily_sales=("daily_observed_sales", "mean"),
        median_daily_sales=("daily_observed_sales", "median"),
        average_observed_records=("observed_records", "mean"),
        number_of_days=("date", "nunique")
    )
    .reindex(weekday_order)
    .reset_index()
)

print("\n========== WEEKDAY DEMAND ==========")
print(weekday_demand.to_string(index=False))

plt.figure(figsize=(12, 6))

sns.barplot(
    data=weekday_demand,
    x="day_name",
    y="average_daily_sales"
)

plt.title("Average Daily Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Observed Units Sold per Day")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "outputs/eda/07_weekday_demand.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 7. WEEKDAY VS WEEKEND
# ------------------------------------------------------------

week_type_demand = (
    daily_summary.groupby("week_type")
    .agg(
        average_daily_sales=("daily_observed_sales", "mean"),
        median_daily_sales=("daily_observed_sales", "median"),
        average_observed_records=("observed_records", "mean"),
        number_of_days=("date", "nunique")
    )
    .reindex(["Weekday", "Weekend"])
    .reset_index()
)

print("\n========== WEEKDAY VS WEEKEND ==========")
print(week_type_demand.to_string(index=False))

plt.figure(figsize=(8, 6))

sns.barplot(
    data=week_type_demand,
    x="week_type",
    y="average_daily_sales"
)

plt.title("Average Daily Sales: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Average Observed Units Sold per Day")

plt.tight_layout()

plt.savefig(
    "outputs/eda/08_weekday_vs_weekend.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 8. MONTHLY DEMAND COMPARISON
# ------------------------------------------------------------

monthly_demand = (
    daily_summary.groupby("month_num")
    .agg(
        average_daily_sales=("daily_observed_sales", "mean"),
        median_daily_sales=("daily_observed_sales", "median"),
        number_of_days=("date", "nunique")
    )
    .reset_index()
)

monthly_demand["month_name"] = (
    monthly_demand["month_num"].apply(
        lambda month: pd.Timestamp(
            year=2000,
            month=int(month),
            day=1
        ).month_name()
    )
)

month_order = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August", "September",
    "October", "November", "December"
]

monthly_demand["month_name"] = pd.Categorical(
    monthly_demand["month_name"],
    categories=month_order,
    ordered=True
)

monthly_demand = monthly_demand.sort_values("month_name")

print("\n========== MONTHLY DEMAND ==========")
print(monthly_demand.to_string(index=False))

plt.figure(figsize=(14, 6))

sns.barplot(
    data=monthly_demand,
    x="month_name",
    y="average_daily_sales"
)

plt.title("Average Daily Sales by Calendar Month")
plt.xlabel("Month")
plt.ylabel("Average Observed Units Sold per Day")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/eda/09_monthly_daily_sales.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 9. QUARTERLY DEMAND
# ------------------------------------------------------------

quarterly_demand = (
    daily_summary.groupby("quarter")
    .agg(
        average_daily_sales=("daily_observed_sales", "mean"),
        median_daily_sales=("daily_observed_sales", "median"),
        number_of_days=("date", "nunique")
    )
    .reindex(quarter_order)
    .reset_index()
)

print("\n========== QUARTERLY DEMAND ==========")
print(quarterly_demand.to_string(index=False))

plt.figure(figsize=(9, 6))

sns.barplot(
    data=quarterly_demand,
    x="quarter",
    y="average_daily_sales"
)

plt.title("Average Daily Sales by Quarter")
plt.xlabel("Quarter")
plt.ylabel("Average Observed Units Sold per Day")

plt.tight_layout()

plt.savefig(
    "outputs/eda/10_quarterly_demand.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 10. RESTAURANT WEEKDAY ANALYSIS
# ------------------------------------------------------------

restaurant_weekday = (
    analysis_df.groupby(
        ["restaurant_id", "day_name"]
    )["quantity"]
    .mean()
    .reset_index(name="average_quantity_per_record")
)

restaurant_pivot = restaurant_weekday.pivot(
    index="restaurant_id",
    columns="day_name",
    values="average_quantity_per_record"
)

restaurant_pivot = restaurant_pivot[weekday_order]

print("\n========== RESTAURANT WEEKDAY ANALYSIS ==========")
print(restaurant_pivot.round(2))

plt.figure(figsize=(14, 8))

sns.heatmap(
    restaurant_pivot,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)

plt.title("Average Quantity by Restaurant and Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Restaurant")

plt.tight_layout()

plt.savefig(
    "outputs/eda/11_restaurant_weekday_heatmap.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 11. MENU CATEGORY WEEKDAY ANALYSIS
# ------------------------------------------------------------

category_weekday = (
    analysis_df.groupby(
        ["category", "day_name"]
    )["quantity"]
    .mean()
    .reset_index(name="average_quantity_per_record")
)

category_pivot = category_weekday.pivot(
    index="category",
    columns="day_name",
    values="average_quantity_per_record"
)

category_pivot = category_pivot[weekday_order]

print("\n========== CATEGORY WEEKDAY ANALYSIS ==========")
print(category_pivot.round(2))

plt.figure(figsize=(14, 7))

sns.heatmap(
    category_pivot,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd"
)

plt.title("Average Quantity by Menu Category and Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Menu Category")

plt.tight_layout()

plt.savefig(
    "outputs/eda/12_category_weekday_heatmap.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 12. SAVE SUMMARY TABLES
# ------------------------------------------------------------

weekday_demand.to_csv(
    "outputs/eda/weekday_demand.csv",
    index=False
)

week_type_demand.to_csv(
    "outputs/eda/weekday_vs_weekend.csv",
    index=False
)

monthly_demand.to_csv(
    "outputs/eda/monthly_demand_comparison.csv",
    index=False
)

quarterly_demand.to_csv(
    "outputs/eda/quarterly_demand.csv",
    index=False
)

restaurant_pivot.to_csv(
    "outputs/eda/restaurant_weekday_matrix.csv"
)

category_pivot.to_csv(
    "outputs/eda/category_weekday_matrix.csv"
)

daily_summary.to_csv(
    "outputs/eda/daily_sales_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 14. FAIR SEASONAL COMPARISON: 2021–2024 ONLY
# ------------------------------------------------------------

# Use four complete calendar years so that every month
# and quarter is represented equally across the years.

complete_year_df = daily_summary[
    daily_summary["date"].dt.year.between(2021, 2024)
].copy()

# Monthly comparison
complete_monthly = (
    complete_year_df.groupby("month_num")
    .agg(
        average_daily_sales=("daily_observed_sales", "mean"),
        median_daily_sales=("daily_observed_sales", "median"),
        number_of_days=("date", "nunique")
    )
    .reset_index()
)

complete_monthly["month_name"] = (
    complete_monthly["month_num"].apply(
        lambda month: pd.Timestamp(
            year=2000,
            month=int(month),
            day=1
        ).month_name()
    )
)

complete_monthly.to_csv(
    "outputs/eda/monthly_demand_2021_2024.csv",
    index=False
)

# Quarterly comparison
complete_quarterly = (
    complete_year_df.groupby("quarter")
    .agg(
        average_daily_sales=("daily_observed_sales", "mean"),
        median_daily_sales=("daily_observed_sales", "median"),
        number_of_days=("date", "nunique")
    )
    .reindex(quarter_order)
    .reset_index()
)

complete_quarterly.to_csv(
    "outputs/eda/quarterly_demand_2021_2024.csv",
    index=False
)

print("\n========== COMPLETE YEAR MONTHLY COMPARISON ==========")
print(complete_monthly.to_string(index=False))

print("\n========== COMPLETE YEAR QUARTERLY COMPARISON ==========")
print(complete_quarterly.to_string(index=False))

print("\nComplete-year comparison saved successfully!")



# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========== KEY OBSERVATIONS ==========")

highest_day = weekday_demand.loc[
    weekday_demand["average_daily_sales"].idxmax()
]

lowest_day = weekday_demand.loc[
    weekday_demand["average_daily_sales"].idxmin()
]

print(
    "Highest-demand weekday:",
    highest_day["day_name"],
    round(highest_day["average_daily_sales"], 2)
)

print(
    "Lowest-demand weekday:",
    lowest_day["day_name"],
    round(lowest_day["average_daily_sales"], 2)
)

highest_month = monthly_demand.loc[
    monthly_demand["average_daily_sales"].idxmax()
]

print(
    "Highest-demand month:",
    highest_month["month_name"],
    round(highest_month["average_daily_sales"], 2)
)

highest_quarter = quarterly_demand.loc[
    quarterly_demand["average_daily_sales"].idxmax()
]

print(
    "Highest-demand quarter:",
    highest_quarter["quarter"],
    round(highest_quarter["average_daily_sales"], 2)
)

print("\nEDA EXERCISE 2 COMPLETED!")
print("Charts and tables saved in outputs/eda/")
