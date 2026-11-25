import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/qsr_demand_dataset.csv")

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())

# --------------------------------------------------
# 3. ISOLATE FORECAST PERIOD
# --------------------------------------------------

forecast_df = df[
    (df["date"] >= "2025-10-01") &
    (df["date"] <= "2025-12-31")
].copy()

print("\nFORECAST PERIOD OVERVIEW")
print("Rows:", len(forecast_df))
print("Unique dates:", forecast_df["date"].nunique())
print("Restaurants:", forecast_df["restaurant_id"].nunique())
print("Menu items:", forecast_df["menu_item_id"].nunique())
print("Start date:", forecast_df["date"].min())
print("End date:", forecast_df["date"].max())


# --------------------------------------------------
# 4. CHECK FEATURE AVAILABILITY
# --------------------------------------------------

feature_columns = [
    "unit_price",
    "avg_temp_f",
    "precip_inches",
    "precip_type",
    "is_holiday",
    "holiday_name",
    "is_special_event",
    "special_event_name",
    "is_promotion"
]

availability_summary = pd.DataFrame({
    "column": feature_columns,
    "missing_values": [
        forecast_df[col].isna().sum()
        for col in feature_columns
    ],
    "missing_percentage": [
        forecast_df[col].isna().mean() * 100
        for col in feature_columns
    ],
    "unique_values": [
        forecast_df[col].nunique(dropna=True)
        for col in feature_columns
    ]
})

print("\nFORECAST FEATURE AVAILABILITY")
print(availability_summary.to_string(index=False))


# --------------------------------------------------
# 5. FEATURE DISTRIBUTIONS
# --------------------------------------------------

categorical_features = [
    "is_holiday",
    "is_special_event",
    "is_promotion",
    "is_weekend"
]

for column in categorical_features:

    print(f"\n{column.upper()} DISTRIBUTION")

    print(
        forecast_df[column]
        .value_counts(dropna=False)
        .sort_index()
    )

    # --------------------------------------------------
# 6. CHECK FORECAST FEATURE CONSISTENCY
# --------------------------------------------------

context_columns = [
    "avg_temp_f",
    "precip_inches",
    "precip_type",
    "is_holiday",
    "holiday_name",
    "is_special_event",
    "special_event_name",
    "is_weekend"
]

consistency_check = (
    forecast_df.groupby(["date", "restaurant_id"])[context_columns]
    .nunique(dropna=False)
)

print("\nFORECAST FEATURE CONSISTENCY")

for column in context_columns:

    inconsistent = (
        consistency_check[column] > 1
    ).sum()

    print(f"{column}: {inconsistent} inconsistent restaurant-days")

# --------------------------------------------------
# 7. PRECIPITATION TYPE INVESTIGATION
# --------------------------------------------------

print("\nPRECIPITATION TYPE VS AMOUNT")

precipitation_check = pd.crosstab(
    forecast_df["precip_inches"] == 0,
    forecast_df["precip_type"].isna(),
    margins=True
)

print(precipitation_check)


# --------------------------------------------------
# 8. UNIT PRICE INVESTIGATION
# --------------------------------------------------

price_summary = (
    forecast_df.groupby(
        ["restaurant_id", "menu_item_id"]
    )["unit_price"]
    .nunique()
)

print("\nUNIT PRICE VARIATION")

print("Restaurant-item combinations:", len(price_summary))

print(
    "Combinations with more than one price:",
    (price_summary > 1).sum()
)

print(
    "Maximum distinct prices per combination:",
    price_summary.max()
)

# --------------------------------------------------
# 9. CHECK EVENT AND PROMOTION FLAGS
# --------------------------------------------------

print("\nEVENT NAME CHECK")

print(
    pd.crosstab(
        forecast_df["is_special_event"],
        forecast_df["special_event_name"].isna()
    )
)

print("\nHOLIDAY NAME CHECK")

print(
    pd.crosstab(
        forecast_df["is_holiday"],
        forecast_df["holiday_name"].isna()
    )
)

print("\nPROMOTION DISTRIBUTION BY DATE")

promotion_by_date = (
    forecast_df.groupby("date")["is_promotion"]
    .sum()
)

print(promotion_by_date.describe())


# --------------------------------------------------
# 10. HISTORICAL UNIT PRICE STABILITY
# --------------------------------------------------

price_history = (
    df.groupby(["restaurant_id", "menu_item_id"])["unit_price"]
    .agg(
        distinct_prices="nunique",
        minimum_price="min",
        maximum_price="max"
    )
    .reset_index()
)

print("\nHISTORICAL UNIT PRICE STABILITY")

print(
    "Restaurant-item combinations:",
    len(price_history)
)

print(
    "Combinations with price changes:",
    (price_history["distinct_prices"] > 1).sum()
)

print(
    "Maximum distinct prices:",
    price_history["distinct_prices"].max()
)

print("\nCOMBINATIONS WITH PRICE CHANGES")

print(
    price_history[
        price_history["distinct_prices"] > 1
    ].to_string(index=False)
)

# --------------------------------------------------
# 11. PROMOTION FREQUENCY BY YEAR
# --------------------------------------------------

promotion_history = df.copy()

promotion_history["year"] = (
    promotion_history["date"].dt.year
)

promotion_yearly = (
    promotion_history.groupby("year")
    .agg(
        total_records=("is_promotion", "count"),
        promotional_records=("is_promotion", "sum")
    )
    .reset_index()
)

promotion_yearly["promotion_rate_pct"] = (
    promotion_yearly["promotional_records"] /
    promotion_yearly["total_records"] * 100
)

print("\nPROMOTION FREQUENCY BY YEAR")

print(promotion_yearly.to_string(index=False))

# --------------------------------------------------
# 12. PROMOTION DATE COVERAGE
# --------------------------------------------------

promotion_dates = (
    df.groupby("date")["is_promotion"]
    .sum()
    .reset_index(name="promotional_records")
)

promotion_dates["has_promotion"] = (
    promotion_dates["promotional_records"] > 0
)

print("\nPROMOTION DATE COVERAGE")

print(
    "Total dates:",
    len(promotion_dates)
)

print(
    "Dates with promotions:",
    promotion_dates["has_promotion"].sum()
)

print(
    "Dates without promotions:",
    (~promotion_dates["has_promotion"]).sum()
)

print(
    "Maximum promotional records on a date:",
    promotion_dates["promotional_records"].max()
)