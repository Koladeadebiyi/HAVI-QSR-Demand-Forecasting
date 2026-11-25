
# ==========================================
# HAVI QSR DEMAND FORECASTING
# ZERO-SALES AND INTERMITTENT-DEMAND ANALYSIS
# ==========================================

import numpy as np
import pandas as pd

from pathlib import Path


# ------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------

DATA_DIR = Path("data")
MODEL_DATA_DIR = DATA_DIR / "processed" / "model_data"
OUTPUT_DIR = Path("outputs/model_evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA_DIR / "qsr_demand_dataset.csv"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "v1_v2_detailed_error_comparison.csv"
)

VALIDATION_FILE = MODEL_DATA_DIR / "validation_data.csv"

VALIDATION_ORIGIN = pd.Timestamp("2025-06-30")


# ------------------------------------------
# 2. LOAD DATA
# ------------------------------------------

print("--- LOADING DATA ---")

raw_df = pd.read_csv(RAW_FILE)
predictions_df = pd.read_csv(PREDICTIONS_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

for df in [raw_df, predictions_df, validation_df]:
    df["date"] = pd.to_datetime(df["date"])

print("Raw historical data:", raw_df.shape)
print("Prediction data:", predictions_df.shape)
print("Validation data:", validation_df.shape)

required_raw_columns = [
    "date",
    "restaurant_id",
    "menu_item_id",
    "quantity"
]

missing_raw_columns = [
    col for col in required_raw_columns
    if col not in raw_df.columns
]

if missing_raw_columns:
    raise ValueError(
        f"Missing raw dataset columns: {missing_raw_columns}"
    )


# ------------------------------------------
# 3. PREPARE VALIDATION OBSERVATIONS
# ------------------------------------------

keys = [
    "date",
    "restaurant_id",
    "menu_item_id"
]

df = predictions_df[
    keys + [
        "actual",
        "predicted_v1",
        "predicted_v2"
    ]
].copy()

df = df.merge(
    validation_df[
        keys + ["category"]
    ],
    on=keys,
    how="left",
    validate="one_to_one"
)

if df["category"].isna().any():
    raise ValueError(
        "Some validation observations have no category."
    )

df["actual_zero"] = df["actual"].eq(0)

df["predicted_zero_v2"] = (
    df["predicted_v2"].eq(0)
)

df["false_positive_v2"] = (
    df["actual_zero"] & ~df["predicted_zero_v2"]
)

df["false_negative_v2"] = (
    df["actual"].gt(0) & df["predicted_zero_v2"]
)

df["zero_day_prediction"] = np.where(
    df["actual_zero"],
    df["predicted_v2"],
    np.nan
)

df["absolute_error_v2"] = (
    df["actual"] - df["predicted_v2"]
).abs()

df["forecast_horizon_day"] = (
    df["date"] - VALIDATION_ORIGIN
).dt.days

df["month"] = df["date"].dt.strftime("%Y-%m")

print("\n--- VALIDATION ZERO-SALES PROFILE ---")
print("Validation observations:", len(df))
print("Zero-sales observations:", df["actual_zero"].sum())
print(
    "Zero-sales rate:",
    round(df["actual_zero"].mean() * 100, 4),
    "%"
)


# ------------------------------------------
# 4. CALCULATE HISTORICAL ZERO-SALES RATES
# ------------------------------------------

print("\n--- CALCULATING HISTORICAL ZERO RATES ---")

# Only information available at the forecast origin.
history = raw_df[
    raw_df["date"] <= VALIDATION_ORIGIN
].copy()

# Exclude missing targets from the zero-rate denominator.
# A missing quantity is not automatically a zero sale.
history = history[
    history["quantity"].notna()
].copy()

history["is_zero_sale"] = (
    history["quantity"].eq(0).astype(int)
)

historical_windows = {
    "28d": 28,
    "90d": 90,
    "365d": 365
}

historical_rate_frames = []

for label, days in historical_windows.items():

    window_start = (
        VALIDATION_ORIGIN
        - pd.Timedelta(days=days - 1)
    )

    window_df = history[
        history["date"].between(
            window_start,
            VALIDATION_ORIGIN
        )
    ]

    rates = window_df.groupby(
        ["restaurant_id", "menu_item_id"]
    ).agg(
        observed_days=("quantity", "count"),
        zero_days=("is_zero_sale", "sum")
    ).reset_index()

    rates[f"historical_zero_rate_{label}"] = (
        rates["zero_days"] / rates["observed_days"]
    )

    rates = rates[
        [
            "restaurant_id",
            "menu_item_id",
            f"historical_zero_rate_{label}",
            "observed_days"
        ]
    ]

    rates = rates.rename(
        columns={
            "observed_days": f"observed_days_{label}"
        }
    )

    historical_rate_frames.append(rates)

historical_rates = historical_rate_frames[0]

for rates in historical_rate_frames[1:]:

    historical_rates = historical_rates.merge(
        rates,
        on=["restaurant_id", "menu_item_id"],
        how="outer",
        validate="one_to_one"
    )

df = df.merge(
    historical_rates,
    on=["restaurant_id", "menu_item_id"],
    how="left",
    validate="many_to_one"
)

print("Historical combinations:", len(historical_rates))
print(
    "Missing 28-day historical rates:",
    df["historical_zero_rate_28d"].isna().sum()
)


# ------------------------------------------
# 5. OVERALL ZERO-SALES CONTRIBUTION
# ------------------------------------------

print("\n--- OVERALL ZERO-SALES ERROR CONTRIBUTION ---")

total_actual = df["actual"].abs().sum()

zero_df = df[df["actual_zero"]].copy()
nonzero_df = df[~df["actual_zero"]].copy()

zero_error = zero_df["absolute_error_v2"].sum()
nonzero_error = nonzero_df["absolute_error_v2"].sum()

overall_wmape = (
    df["absolute_error_v2"].sum() / total_actual
) * 100

zero_error_contribution = (
    zero_error / total_actual
) * 100

nonzero_error_contribution = (
    nonzero_error / total_actual
) * 100

overall_zero_summary = pd.DataFrame([{
    "total_observations": len(df),
    "zero_sales_observations": len(zero_df),
    "zero_sales_rate_pct": len(zero_df) / len(df) * 100,
    "total_actual_demand": total_actual,
    "zero_sales_prediction_sum": zero_df["predicted_v2"].sum(),
    "zero_sales_absolute_error": zero_error,
    "nonzero_sales_absolute_error": nonzero_error,
    "overall_wmape_pct": overall_wmape,
    "zero_sales_wmape_contribution_pp": zero_error_contribution,
    "nonzero_sales_wmape_contribution_pp": nonzero_error_contribution
}])

overall_zero_summary.to_csv(
    OUTPUT_DIR / "zero_sales_overall_contribution.csv",
    index=False
)

print(overall_zero_summary.round(4).to_string(index=False))


# ------------------------------------------
# 6. ZERO SALES BY MENU ITEM
# ------------------------------------------

def zero_sales_group_summary(data, group_columns):

    results = []

    for group_values, group in data.groupby(
        group_columns,
        observed=True,
        dropna=False
    ):

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        zero_group = group[group["actual_zero"]]

        result = dict(zip(group_columns, group_values))

        result.update({
            "observations": len(group),
            "zero_sales_days": len(zero_group),
            "zero_sales_rate_pct": (
                len(zero_group) / len(group) * 100
            ),
            "predicted_demand_on_zero_days": (
                zero_group["predicted_v2"].sum()
            ),
            "average_prediction_on_zero_days": (
                zero_group["predicted_v2"].mean()
                if len(zero_group) else np.nan
            ),
            "historical_zero_rate_28d": (
                group["historical_zero_rate_28d"].iloc[0]
            ),
            "historical_zero_rate_90d": (
                group["historical_zero_rate_90d"].iloc[0]
            ),
            "historical_zero_rate_365d": (
                group["historical_zero_rate_365d"].iloc[0]
            )
        })

        results.append(result)

    return pd.DataFrame(results)


item_zero_results = zero_sales_group_summary(
    df,
    ["menu_item_id"]
).sort_values(
    "predicted_demand_on_zero_days",
    ascending=False
)

item_zero_results.to_csv(
    OUTPUT_DIR / "zero_sales_by_menu_item.csv",
    index=False
)

print("\n--- TOP 15 ITEMS BY FALSE-POSITIVE DEMAND ---")
print(
    item_zero_results.head(15).round(4).to_string(index=False)
)


# ------------------------------------------
# 7. ZERO SALES BY RESTAURANT
# ------------------------------------------

restaurant_zero_results = zero_sales_group_summary(
    df,
    ["restaurant_id"]
).sort_values(
    "predicted_demand_on_zero_days",
    ascending=False
)

restaurant_zero_results.to_csv(
    OUTPUT_DIR / "zero_sales_by_restaurant.csv",
    index=False
)

print("\n--- ZERO SALES BY RESTAURANT ---")
print(
    restaurant_zero_results.round(4).to_string(index=False)
)


# ------------------------------------------
# 8. ZERO SALES BY RESTAURANT-ITEM
# ------------------------------------------

combination_zero_results = zero_sales_group_summary(
    df,
    ["restaurant_id", "menu_item_id"]
).sort_values(
    "predicted_demand_on_zero_days",
    ascending=False
)

combination_zero_results.to_csv(
    OUTPUT_DIR / "zero_sales_by_restaurant_item.csv",
    index=False
)

print("\n--- TOP 15 RESTAURANT-ITEM ZERO-SALES COMBINATIONS ---")
print(
    combination_zero_results.head(15).round(4).to_string(index=False)
)


# ------------------------------------------
# 9. ZERO SALES BY MONTH AND HORIZON
# ------------------------------------------

monthly_zero_results = zero_sales_group_summary(
    df,
    ["month"]
).sort_values("month")

monthly_zero_results.to_csv(
    OUTPUT_DIR / "zero_sales_by_month.csv",
    index=False
)

df["horizon_band"] = pd.cut(
    df["forecast_horizon_day"],
    bins=[0, 7, 14, 30, 60, 92],
    labels=[
        "Days 1-7",
        "Days 8-14",
        "Days 15-30",
        "Days 31-60",
        "Days 61-92"
    ],
    include_lowest=True
)

horizon_zero_results = zero_sales_group_summary(
    df,
    ["horizon_band"]
)

horizon_zero_results.to_csv(
    OUTPUT_DIR / "zero_sales_by_horizon.csv",
    index=False
)

print("\n--- ZERO SALES BY MONTH ---")
print(
    monthly_zero_results.round(4).to_string(index=False)
)

print("\n--- ZERO SALES BY FORECAST HORIZON ---")
print(
    horizon_zero_results.round(4).to_string(index=False)
)


# ------------------------------------------
# 10. HISTORICAL INTERMITTENCY ANALYSIS
# ------------------------------------------

# Use the 28-day historical zero-sales rate
# to classify restaurant-item combinations.

df["historical_intermittency_band"] = pd.cut(
    df["historical_zero_rate_28d"],
    bins=[-0.000001, 0.05, 0.15, 0.30, 1.0],
    labels=[
        "Rare zero sales (0-5%)",
        "Occasional zero sales (5-15%)",
        "Frequent zero sales (15-30%)",
        "Highly intermittent (>30%)"
    ],
    include_lowest=True
)

intermittency_results = zero_sales_group_summary(
    df,
    ["historical_intermittency_band"]
)

intermittency_results.to_csv(
    OUTPUT_DIR / "zero_sales_by_historical_intermittency.csv",
    index=False
)

print("\n--- ZERO SALES BY HISTORICAL INTERMITTENCY ---")
print(
    intermittency_results.round(4).to_string(index=False)
)


# ------------------------------------------
# 11. RELATIONSHIP BETWEEN HISTORICAL AND
#     VALIDATION ZERO-SALES RATES
# ------------------------------------------

# First aggregate validation outcomes at the
# restaurant-item level to avoid treating every
# daily row as an independent historical profile.

combo_intermittency = df.groupby(
    ["restaurant_id", "menu_item_id"],
    observed=True
).agg(
    validation_observations=("actual", "size"),
    validation_zero_days=("actual_zero", "sum"),
    validation_zero_rate=("actual_zero", "mean"),
    historical_zero_rate_28d=(
        "historical_zero_rate_28d", "first"
    ),
    historical_zero_rate_90d=(
        "historical_zero_rate_90d", "first"
    ),
    historical_zero_rate_365d=(
        "historical_zero_rate_365d", "first"
    ),
    average_prediction_on_zero_days=(
        "zero_day_prediction", "mean"
    )
).reset_index()

combo_intermittency.to_csv(
    OUTPUT_DIR / "historical_vs_validation_intermittency.csv",
    index=False
)

correlation_results = []

for historical_col in [
    "historical_zero_rate_28d",
    "historical_zero_rate_90d",
    "historical_zero_rate_365d"
]:

    valid = combo_intermittency[
        [
            historical_col,
            "validation_zero_rate"
        ]
    ].dropna()

    correlation_results.append({
        "historical_window": historical_col,
        "combinations": len(valid),
        "pearson_correlation": (
            valid[historical_col].corr(
                valid["validation_zero_rate"],
                method="pearson"
            )
        ),
        "spearman_correlation": (
            valid[historical_col].corr(
                valid["validation_zero_rate"],
                method="spearman"
            )
        )
    })

correlation_results = pd.DataFrame(correlation_results)

correlation_results.to_csv(
    OUTPUT_DIR / "historical_validation_zero_rate_correlations.csv",
    index=False
)

print("\n--- HISTORICAL VS VALIDATION ZERO-RATE CORRELATION ---")
print(
    correlation_results.round(4).to_string(index=False)
)


# ------------------------------------------
# 12. SAVE DETAILED ANALYSIS DATA
# ------------------------------------------

df.to_csv(
    OUTPUT_DIR / "zero_sales_detailed_analysis.csv",
    index=False
)


# ------------------------------------------
# 13. COMPLETION
# ------------------------------------------

print("\n--- FILES SAVED ---")
print("zero_sales_overall_contribution.csv")
print("zero_sales_by_menu_item.csv")
print("zero_sales_by_restaurant.csv")
print("zero_sales_by_restaurant_item.csv")
print("zero_sales_by_month.csv")
print("zero_sales_by_horizon.csv")
print("zero_sales_by_historical_intermittency.csv")
print("historical_vs_validation_intermittency.csv")
print("historical_validation_zero_rate_correlations.csv")
print("zero_sales_detailed_analysis.csv")

print("\n--- ZERO-SALES ANALYSIS COMPLETE ---")
