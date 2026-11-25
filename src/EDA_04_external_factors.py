
# ============================================================
# EXERCISE 4 - EXTERNAL FACTORS AND DEMAND RELATIONSHIPS
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

OUTPUT_DIR = Path("outputs/eda_04_external")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("\nSTARTING EXERCISE 4: EXTERNAL FACTORS ANALYSIS")

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
# 1. PREPARE RESTAURANT-DAY DATA
# ------------------------------------------------------------

# Weather, holiday and event information is repeated across
# menu items. Aggregate to one record per restaurant per day.

# Check whether external factors are consistent
# across menu-item records within each restaurant-day.

context_columns = [
    "avg_temp_f",
    "precip_inches",
    "is_holiday",
    "is_special_event",
    "is_weekend",
    "month",
    "year"
]

context_consistency = (
    analysis_df.groupby(["date", "restaurant_id"])[context_columns]
    .nunique(dropna=False)
)

print("\nEXTERNAL FACTOR CONSISTENCY CHECK")
for column in context_columns:
    inconsistent = (context_consistency[column] > 1).sum()
    print(f"{column}: {inconsistent} inconsistent restaurant-days")

restaurant_day = (
    analysis_df.groupby(
        ["date", "restaurant_id"]
    )
    .agg(
        daily_average_quantity=("quantity", "mean"),
        daily_total_quantity=(
            "quantity",
            lambda x: x.sum(min_count=1)
        ),
        observed_items=("quantity", "count"),
        recorded_items=("menu_item_id", "count"),
        avg_temp_f=("avg_temp_f", "first"),
        precip_inches=("precip_inches", "first"),
        is_holiday=("is_holiday", "first"),
        is_special_event=("is_special_event", "first"),
        is_weekend=("is_weekend", "first"),
        month=("month", "first"),
        year=("year", "first")
    )
    .reset_index()
)

# Define rainy/snowy days using precipitation amount.
# We do not rely on missing precip_type values.

restaurant_day["has_precipitation"] = np.where(
    restaurant_day["precip_inches"].isna(),
    np.nan,
    (restaurant_day["precip_inches"] > 0).astype(int)
)

print(
    "Restaurant-days with missing precipitation:",
    restaurant_day["has_precipitation"].isna().sum()
)

print("\nRestaurant-day dataset:")
print(restaurant_day.shape)

# ------------------------------------------------------------
# 2. TEMPERATURE AND DEMAND RELATIONSHIP
# ------------------------------------------------------------

# Remove restaurant-level differences by centering demand
# and temperature within each restaurant.

restaurant_day["quantity_centered"] = (
    restaurant_day["daily_average_quantity"]
    - restaurant_day.groupby(
        "restaurant_id"
    )["daily_average_quantity"].transform("mean")
)

restaurant_day["temperature_centered"] = (
    restaurant_day["avg_temp_f"]
    - restaurant_day.groupby(
        "restaurant_id"
    )["avg_temp_f"].transform("mean")
)

temperature_corr = (
    restaurant_day[
        ["quantity_centered", "temperature_centered"]
    ]
    .corr(method="pearson")
    .iloc[0, 1]
)

temperature_spearman = (
    restaurant_day[
        ["quantity_centered", "temperature_centered"]
    ]
    .corr(method="spearman")
    .iloc[0, 1]
)

print("\nTEMPERATURE ANALYSIS")
print("Within-restaurant Pearson correlation:",
      round(temperature_corr, 4))
print("Within-restaurant Spearman correlation:",
      round(temperature_spearman, 4))

plt.figure(figsize=(10, 6))

sns.regplot(
    data=restaurant_day,
    x="temperature_centered",
    y="quantity_centered",
    scatter_kws={"alpha": 0.15, "s": 12},
    line_kws={"color": "red"},
)

plt.title("Temperature vs Average Daily Demand")
plt.xlabel("Temperature Deviation from Restaurant Average (°F)")
plt.ylabel("Average Quantity Deviation from Restaurant Average")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "23_temperature_vs_demand.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 3. PRECIPITATION AND DEMAND RELATIONSHIP
# ------------------------------------------------------------

precipitation_summary = (
    restaurant_day.groupby("has_precipitation")
    .agg(
        average_quantity=("daily_average_quantity", "mean"),
        median_quantity=("daily_average_quantity", "median"),
        restaurant_days=("date", "count")
    )
    .reset_index()
)

precipitation_summary["weather_condition"] = (
    precipitation_summary["has_precipitation"]
    .map({0: "Dry", 1: "Precipitation"})
)

print("\nPRECIPITATION ANALYSIS")
print(precipitation_summary)

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=restaurant_day,
    x="has_precipitation",
    y="daily_average_quantity"
)

plt.title("Demand Distribution: Dry vs Precipitation Days")
plt.xlabel("Precipitation (0 = Dry, 1 = Precipitation)")
plt.ylabel("Average Quantity per Menu Item")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "24_precipitation_vs_demand.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 4. HOLIDAY AND SPECIAL EVENT ANALYSIS
# ------------------------------------------------------------

event_summary = (
    restaurant_day.groupby(
        ["is_holiday", "is_special_event"]
    )
    .agg(
        average_quantity=("daily_average_quantity", "mean"),
        median_quantity=("daily_average_quantity", "median"),
        restaurant_days=("date", "count")
    )
    .reset_index()
)

print("\nHOLIDAY AND SPECIAL EVENT ANALYSIS")
print(event_summary)

# Holiday comparison

holiday_summary = (
    restaurant_day.groupby("is_holiday")
    .agg(
        average_quantity=("daily_average_quantity", "mean"),
        median_quantity=("daily_average_quantity", "median"),
        restaurant_days=("date", "count")
    )
    .reset_index()
)

holiday_summary["day_type"] = (
    holiday_summary["is_holiday"]
    .map({0: "Non-holiday", 1: "Holiday"})
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=holiday_summary,
    x="day_type",
    y="average_quantity"
)

plt.title("Average Demand: Holiday vs Non-Holiday")
plt.xlabel("Day Type")
plt.ylabel("Average Quantity per Menu Item")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "25_holiday_vs_demand.png",
    dpi=300
)

plt.close()

# Special event comparison

event_only_summary = (
    restaurant_day.groupby("is_special_event")
    .agg(
        average_quantity=("daily_average_quantity", "mean"),
        median_quantity=("daily_average_quantity", "median"),
        restaurant_days=("date", "count")
    )
    .reset_index()
)

event_only_summary["day_type"] = (
    event_only_summary["is_special_event"]
    .map({0: "No special event", 1: "Special event"})
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=event_only_summary,
    x="day_type",
    y="average_quantity"
)

plt.title("Average Demand: Special Event vs No Special Event")
plt.xlabel("Event Status")
plt.ylabel("Average Quantity per Menu Item")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "26_special_event_vs_demand.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 5. PROMOTION AND DEMAND ANALYSIS
# ------------------------------------------------------------

# Promotions are item-level features.
# Compare promoted and non-promoted demand within each
# restaurant-item combination to reduce baseline differences.

promotion_summary = (
    analysis_df.groupby(
        [
            "restaurant_id",
            "menu_item_id",
            "menu_item_name",
            "category",
            "is_promotion"
        ]
    )
    .agg(
        average_quantity=("quantity", "mean"),
        observed_records=("quantity", "count")
    )
    .reset_index()
)

promotion_pivot = promotion_summary.pivot(
    index=[
        "restaurant_id",
        "menu_item_id",
        "menu_item_name",
        "category"
    ],
    columns="is_promotion",
    values=["average_quantity", "observed_records"]
)

# Flatten the multi-level column names
promotion_pivot.columns = [
    f"{metric}_{status}"
    for metric, status in promotion_pivot.columns
]

promotion_pivot = promotion_pivot.reset_index()

# Rename columns for clarity
promotion_pivot = promotion_pivot.rename(
    columns={
        "average_quantity_0": "non_promotion_average",
        "average_quantity_1": "promotion_average",
        "observed_records_0": "non_promotion_observations",
        "observed_records_1": "promotion_observations"
    }
)

# Keep combinations with observations in both conditions.

promotion_pivot = promotion_pivot.dropna(
    subset=[
        "non_promotion_average",
        "promotion_average"
    ]
)

promotion_pivot["promotion_difference"] = (
    promotion_pivot["promotion_average"]
    - promotion_pivot["non_promotion_average"]
)

promotion_pivot["promotion_uplift_pct"] = (
    promotion_pivot["promotion_difference"]
    / promotion_pivot["non_promotion_average"].replace(0, np.nan)
) * 100

print("\nPROMOTION ANALYSIS")
print("\nAverage promotion difference:")
print(
    promotion_pivot["promotion_difference"].describe()
)

print("\nAverage promotion uplift:")
print(
    promotion_pivot["promotion_uplift_pct"].describe()
)

print("\nPROMOTION COVERAGE CHECK")

print(
    "Combinations with fewer than 10 promotional observations:",
    (promotion_pivot["promotion_observations"] < 10).sum()
)

print(
    "Combinations with fewer than 20 promotional observations:",
    (promotion_pivot["promotion_observations"] < 20).sum()
)

print(
    "Combinations with zero baseline demand:",
    (promotion_pivot["non_promotion_average"] == 0).sum()
)

print(
    "Combinations excluded from comparison:",
    750 - len(promotion_pivot)
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=promotion_pivot,
    x="promotion_uplift_pct",
    bins=30,
    kde=True
)

plt.title("Distribution of Promotion Uplift by Restaurant-Item")
plt.xlabel("Promotion Uplift (%)")
plt.ylabel("Number of Combinations")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "27_promotion_uplift_distribution.png",
    dpi=300
)

plt.close()

print("\nPROMOTION OBSERVATION COUNTS")
print(
    promotion_pivot[
        [
            "non_promotion_observations",
            "promotion_observations"
        ]
    ].describe()
)

print("\nPROMOTION UPLIFT SUMMARY")
print(
    promotion_pivot["promotion_uplift_pct"].describe()
)

print("\nPositive uplift combinations:",
      (promotion_pivot["promotion_uplift_pct"] > 0).sum())

print("Negative uplift combinations:",
      (promotion_pivot["promotion_uplift_pct"] < 0).sum())
# ------------------------------------------------------------
# 6. WEEKEND INTERACTION WITH EXTERNAL FACTORS
# ------------------------------------------------------------

weekend_external = (
    restaurant_day.groupby("is_weekend")
    .agg(
        average_temperature=("avg_temp_f", "mean"),
        average_precipitation=("precip_inches", "mean"),
        average_quantity=("daily_average_quantity", "mean"),
        holiday_restaurant_days=("is_holiday", "sum"),
        event_restaurant_days=("is_special_event", "sum"),
        restaurant_days=("date", "count")
    )
    .reset_index()
)

weekend_external["day_type"] = (
    weekend_external["is_weekend"]
    .map({0: "Weekday", 1: "Weekend"})
)

print("\nWEEKEND AND EXTERNAL FACTORS")
print(weekend_external)

# ------------------------------------------------------------
# 7. EXPORT ANALYSIS TABLES
# ------------------------------------------------------------

restaurant_day.to_csv(
    OUTPUT_DIR / "restaurant_day_external_factors.csv",
    index=False
)

precipitation_summary.to_csv(
    OUTPUT_DIR / "precipitation_summary.csv",
    index=False
)

holiday_summary.to_csv(
    OUTPUT_DIR / "holiday_summary.csv",
    index=False
)

event_only_summary.to_csv(
    OUTPUT_DIR / "special_event_summary.csv",
    index=False
)

promotion_pivot.to_csv(
    OUTPUT_DIR / "promotion_comparison.csv",
    index=False
)

weekend_external.to_csv(
    OUTPUT_DIR / "weekend_external_factors.csv",
    index=False
)

event_summary.to_csv(
    OUTPUT_DIR / "holiday_event_interaction.csv",
    index=False
)

# ------------------------------------------------------------
# 8. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\nEXERCISE 4 COMPLETED SUCCESSFULLY")

print("\nGenerated charts:")
print("23_temperature_vs_demand.png")
print("24_precipitation_vs_demand.png")
print("25_holiday_vs_demand.png")
print("26_special_event_vs_demand.png")
print("27_promotion_uplift_distribution.png")

print("\nAnalysis tables exported successfully.")

