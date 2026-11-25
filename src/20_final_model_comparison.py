
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = PROJECT_ROOT / "outputs" / "model_evaluation"
OUTPUT_DIR = EVALUATION_DIR

VALIDATION_START = pd.Timestamp("2025-07-01")

MODEL_FILES = {
    "CatBoost_v1": "catboost_v1_validation_predictions.csv",
    "CatBoost_v2": "catboost_v2_validation_predictions.csv",
    "CatBoost_v3": "catboost_v3_validation_predictions.csv",
}


# ============================================================
# 2. LOAD AND STANDARDISE MODEL PREDICTIONS
# ============================================================

def load_model_predictions(model_name, filename):

    file_path = EVALUATION_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Standardise actual demand column
    if "quantity" in df.columns:
        df = df.rename(columns={"quantity": "actual_quantity"})

    required_columns = [
        "date",
        "restaurant_id",
        "menu_item_id",
        "actual_quantity",
        "predicted_quantity",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{model_name} is missing columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    df["date"] = pd.to_datetime(df["date"])

    df["actual_quantity"] = pd.to_numeric(
        df["actual_quantity"], errors="coerce"
    )

    df["predicted_quantity"] = pd.to_numeric(
        df["predicted_quantity"], errors="coerce"
    )

    df["model"] = model_name

    return df


model_data = []

for model_name, filename in MODEL_FILES.items():

    model_df = load_model_predictions(model_name, filename)

    model_data.append(model_df)

    print(
        f"{model_name}: {len(model_df):,} rows loaded"
    )


# ============================================================
# 3. MERGE THE THREE MODELS ON IDENTICAL OBSERVATIONS
# ============================================================

merged = None

for model_df in model_data:

    model_name = model_df["model"].iloc[0]

    model_df = model_df.drop(columns="model")

    model_df = model_df.rename(
        columns={
            "actual_quantity": f"actual_{model_name}",
            "predicted_quantity": f"prediction_{model_name}",
        }
    )

    if merged is None:
        merged = model_df

    else:
        merged = merged.merge(
            model_df,
            on=["date", "restaurant_id", "menu_item_id"],
            how="inner",
            validate="one_to_one",
        )


# Confirm that actual quantities agree across models
actual_columns = [
    f"actual_{model}"
    for model in MODEL_FILES
]

for column in actual_columns[1:]:

    if not np.allclose(
        merged[actual_columns[0]],
        merged[column],
        equal_nan=True,
    ):
        raise ValueError(
            "Actual quantities do not match across model files."
        )


merged["actual_quantity"] = merged[actual_columns[0]]

prediction_columns = {
    model: f"prediction_{model}"
    for model in MODEL_FILES
}

# Remove rows with missing actuals or predictions
required_numeric_columns = [
    "actual_quantity",
    *prediction_columns.values(),
]

merged = merged.dropna(
    subset=required_numeric_columns
).copy()

# Keep the validation period
merged = merged[
    (merged["date"] >= pd.Timestamp("2025-07-01"))
    & (merged["date"] <= pd.Timestamp("2025-09-30"))
].copy()

print("\nMatched validation observations:", f"{len(merged):,}")
print("Restaurants:", merged["restaurant_id"].nunique())
print("Menu items:", merged["menu_item_id"].nunique())
print("Validation period:", merged["date"].min(), "to", merged["date"].max())


# ============================================================
# 4. ADD CATEGORY AND FORECAST HORIZON
# ============================================================

# Category is available in the v3 prediction file.
v3_categories = model_data[2][
    ["restaurant_id", "menu_item_id"]
].drop_duplicates()

# Retrieve category from the original v3 file
v3_raw = pd.read_csv(
    EVALUATION_DIR / MODEL_FILES["CatBoost_v3"],
    usecols=["restaurant_id", "menu_item_id", "category"],
)

v3_categories = v3_raw.drop_duplicates(
    subset=["restaurant_id", "menu_item_id"]
)

merged = merged.merge(
    v3_categories,
    on=["restaurant_id", "menu_item_id"],
    how="left",
    validate="many_to_one",
)

if merged["category"].isna().any():
    raise ValueError("Some observations have no category mapping.")

# Forecast horizon: July 1 is day 1
merged["forecast_horizon_day"] = (
    merged["date"] - VALIDATION_START
).dt.days + 1


def assign_horizon_band(day):

    if day <= 7:
        return "01-07 days"

    elif day <= 14:
        return "08-14 days"

    elif day <= 30:
        return "15-30 days"

    elif day <= 60:
        return "31-60 days"

    else:
        return "61-92 days"


merged["horizon_band"] = (
    merged["forecast_horizon_day"].apply(assign_horizon_band)
)


# ============================================================
# 5. DEFINE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(df, prediction_column):

    actual = df["actual_quantity"].to_numpy()
    predicted = df[prediction_column].to_numpy()

    errors = predicted - actual

    total_actual = np.sum(np.abs(actual))

    wmape = (
        np.sum(np.abs(errors)) / total_actual * 100
        if total_actual != 0 else np.nan
    )

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    bias = (
        np.sum(errors) / total_actual * 100
        if total_actual != 0 else np.nan
    )

    return {
        "observations": len(df),
        "actual_demand": np.sum(actual),
        "predicted_demand": np.sum(predicted),
        "wMAPE_pct": wmape,
        "MAE": mae,
        "RMSE": rmse,
        "Bias_pct": bias,
    }


# ============================================================
# 6. OVERALL MODEL COMPARISON
# ============================================================

overall_results = []

for model, prediction_column in prediction_columns.items():

    metrics = calculate_metrics(
        merged,
        prediction_column,
    )

    metrics["model"] = model

    overall_results.append(metrics)


overall = pd.DataFrame(overall_results)

overall = overall[
    [
        "model",
        "observations",
        "actual_demand",
        "predicted_demand",
        "wMAPE_pct",
        "MAE",
        "RMSE",
        "Bias_pct",
    ]
].sort_values("wMAPE_pct")


# ============================================================
# 7. SEGMENT-LEVEL COMPARISON
# ============================================================

def compare_by_segment(df, segment_column):

    results = []

    for segment_value, group in df.groupby(
        segment_column,
        dropna=False,
    ):

        for model, prediction_column in prediction_columns.items():

            metrics = calculate_metrics(
                group,
                prediction_column,
            )

            metrics[segment_column] = segment_value
            metrics["model"] = model

            results.append(metrics)

    return pd.DataFrame(results)


restaurant_results = compare_by_segment(
    merged,
    "restaurant_id",
)

category_results = compare_by_segment(
    merged,
    "category",
)

menu_item_results = compare_by_segment(
    merged,
    "menu_item_id",
)

horizon_results = compare_by_segment(
    merged,
    "horizon_band",
)


# ============================================================
# 8. DETERMINE THE BEST MODEL IN EACH SEGMENT
# ============================================================

def identify_segment_winners(results, segment_column):

    winners = (
        results.sort_values("wMAPE_pct")
        .groupby(segment_column, as_index=False)
        .first()
    )

    return winners[
        [
            segment_column,
            "model",
            "observations",
            "wMAPE_pct",
            "MAE",
            "RMSE",
            "Bias_pct",
        ]
    ]


restaurant_winners = identify_segment_winners(
    restaurant_results,
    "restaurant_id",
)

category_winners = identify_segment_winners(
    category_results,
    "category",
)

menu_item_winners = identify_segment_winners(
    menu_item_results,
    "menu_item_id",
)

horizon_winners = identify_segment_winners(
    horizon_results,
    "horizon_band",
)


# ============================================================
# 9. CALCULATE V2 AND V3 IMPROVEMENT AGAINST V1
# ============================================================

overall_comparison = overall.set_index("model")

baseline_wmape = overall_comparison.loc[
    "CatBoost_v1", "wMAPE_pct"
]

improvement_results = []

for model in ["CatBoost_v2", "CatBoost_v3"]:

    model_wmape = overall_comparison.loc[
        model, "wMAPE_pct"
    ]

    improvement_results.append({
        "model": model,
        "v1_wMAPE": baseline_wmape,
        "model_wMAPE": model_wmape,
        "improvement_percentage_points": (
            baseline_wmape - model_wmape
        ),
        "relative_improvement_pct": (
            (baseline_wmape - model_wmape)
            / baseline_wmape * 100
        ),
    })


improvement = pd.DataFrame(improvement_results)


# ============================================================
# 10. SAVE ALL RESULTS
# ============================================================

outputs = {
    "final_overall_model_comparison.csv": overall,
    "final_restaurant_comparison.csv": restaurant_results,
    "final_category_comparison.csv": category_results,
    "final_menu_item_comparison.csv": menu_item_results,
    "final_horizon_comparison.csv": horizon_results,
    "final_restaurant_winners.csv": restaurant_winners,
    "final_category_winners.csv": category_winners,
    "final_menu_item_winners.csv": menu_item_winners,
    "final_horizon_winners.csv": horizon_winners,
    "final_model_improvement_vs_v1.csv": improvement,
}

for filename, dataframe in outputs.items():

    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {filename}")


# ============================================================
# 11. DISPLAY KEY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FINAL OVERALL MODEL COMPARISON")
print("=" * 65)

print(
    overall.round(4).to_string(index=False)
)

print("\n" + "=" * 65)
print("BEST MODEL BY CATEGORY")
print("=" * 65)

print(
    category_winners.round(4).to_string(index=False)
)

print("\n" + "=" * 65)
print("BEST MODEL BY FORECAST HORIZON")
print("=" * 65)

print(
    horizon_winners.round(4).to_string(index=False)
)

print("\n" + "=" * 65)
print("IMPROVEMENT AGAINST CATBOOST V1")
print("=" * 65)

print(
    improvement.round(4).to_string(index=False)
)

print("\nFinal model comparison completed successfully.")
