
# ==========================================
# HAVI QSR DEMAND FORECASTING
# CATBOOST V2 - FORECAST-ORIGIN MODEL
# ==========================================

import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------

DATA_DIR = Path("data/processed/model_data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/model_evaluation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------
# 2. LOAD PREPARED DATASETS
# ------------------------------------------

training_df = pd.read_csv(
    DATA_DIR / "forecast_origin_training_data.csv"
)

validation_df = pd.read_csv(
    DATA_DIR / "validation_data.csv"
)

# Load v1 training data only to preserve
# exactly the same feature structure.
v1_training_df = pd.read_csv(
    DATA_DIR / "training_data.csv",
    nrows=5
)

print("--- DATASET LOADING ---")
print("V2 training shape:", training_df.shape)
print("Validation shape:", validation_df.shape)


# ------------------------------------------
# 3. PREPARE FEATURES AND TARGET
# ------------------------------------------

target_column = "quantity"

# Use the exact same features as CatBoost v1.
excluded_columns = ["date", target_column]

feature_columns = [
    col for col in v1_training_df.columns
    if col not in excluded_columns
]

categorical_features = [
    "restaurant_id",
    "menu_item_id",
    "category"
]

# Confirm that all required features exist.
missing_training_features = [
    col for col in feature_columns
    if col not in training_df.columns
]

missing_validation_features = [
    col for col in feature_columns
    if col not in validation_df.columns
]

if missing_training_features:
    raise ValueError(
        f"Missing training features: {missing_training_features}"
    )

if missing_validation_features:
    raise ValueError(
        f"Missing validation features: {missing_validation_features}"
    )

# Prepare training and validation datasets.
X_train = training_df[feature_columns].copy()
y_train = training_df[target_column].copy()

X_validation = validation_df[feature_columns].copy()
y_validation = validation_df[target_column].copy()

# Convert categorical variables to strings.
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_validation[col] = X_validation[col].astype(str)

# Confirm matching feature structure.
assert list(X_train.columns) == list(X_validation.columns)

print("\n--- MODEL DATA PREPARATION ---")
print("Training observations:", len(X_train))
print("Validation observations:", len(X_validation))
print("Number of features:", len(feature_columns))
print("Categorical features:", categorical_features)

print("\nForecast origins:", training_df["forecast_origin"].nunique())
print(
    "Forecast horizon range:",
    training_df["forecast_horizon_day"].min(),
    "to",
    training_df["forecast_horizon_day"].max()
)


# ------------------------------------------
# 4. TRAIN CATBOOST REGRESSOR
# ------------------------------------------

model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=100,
    allow_writing_files=False
)

print("\n--- CATBOOST V2 MODEL TRAINING ---")

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_validation, y_validation),
    use_best_model=True
)


# ------------------------------------------
# 5. GENERATE VALIDATION PREDICTIONS
# ------------------------------------------

validation_predictions = model.predict(X_validation)

# Quantities cannot be negative.
validation_predictions = np.maximum(
    validation_predictions, 0
)


# ------------------------------------------
# 6. EVALUATE MODEL PERFORMANCE
# ------------------------------------------

baseline_predictions = np.maximum(
    X_validation["mean_28d"].to_numpy(), 0
)

actual_values = y_validation.to_numpy()

baseline_wmape = (
    np.sum(np.abs(actual_values - baseline_predictions))
    / np.sum(np.abs(actual_values))
) * 100

catboost_wmape = (
    np.sum(np.abs(actual_values - validation_predictions))
    / np.sum(np.abs(actual_values))
) * 100

baseline_mae = mean_absolute_error(
    actual_values, baseline_predictions
)

mae = mean_absolute_error(
    actual_values, validation_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(actual_values, baseline_predictions)
)

rmse = np.sqrt(
    mean_squared_error(actual_values, validation_predictions)
)

improvement = baseline_wmape - catboost_wmape

print("\n--- CATBOOST V2 EVALUATION RESULTS ---")
print("Baseline wMAPE:", round(baseline_wmape, 4), "%")
print("CatBoost V2 wMAPE:", round(catboost_wmape, 4), "%")
print("Baseline MAE:", round(baseline_mae, 4))
print("CatBoost V2 MAE:", round(mae, 4))
print("Baseline RMSE:", round(baseline_rmse, 4))
print("CatBoost V2 RMSE:", round(rmse, 4))
print("wMAPE improvement:", round(improvement, 4), "percentage points")
print("Best iteration:", model.get_best_iteration())


# ------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------

model.save_model(MODEL_DIR / "catboost_v2.cbm")


# ------------------------------------------
# 8. SAVE VALIDATION PREDICTIONS
# ------------------------------------------

validation_results = validation_df[
    ["date", "restaurant_id", "menu_item_id", "quantity"]
].copy()

validation_results["predicted_quantity"] = validation_predictions

validation_results.to_csv(
    OUTPUT_DIR / "catboost_v2_validation_predictions.csv",
    index=False
)


# ------------------------------------------
# 9. SAVE MODEL COMPARISON
# ------------------------------------------

v2_summary = pd.DataFrame({
    "model": [
        "28-day historical average",
        "CatBoost v2"
    ],
    "wMAPE": [
        baseline_wmape,
        catboost_wmape
    ],
    "MAE": [
        baseline_mae,
        mae
    ],
    "RMSE": [
        baseline_rmse,
        rmse
    ]
})

v2_summary.to_csv(
    OUTPUT_DIR / "catboost_v2_model_comparison.csv",
    index=False
)

# Build a combined comparison without overwriting
# the original CatBoost v1 comparison file.
v1_comparison_path = OUTPUT_DIR / "model_comparison.csv"

if v1_comparison_path.exists():

    comparison_df = pd.read_csv(v1_comparison_path)

    comparison_df = comparison_df[
        comparison_df["model"] != "28-day historical average"
    ]

    combined_comparison = pd.concat(
        [comparison_df, v2_summary],
        ignore_index=True
    )

else:
    combined_comparison = v2_summary.copy()

combined_comparison.to_csv(
    OUTPUT_DIR / "model_comparison_v1_vs_v2.csv",
    index=False
)


# ------------------------------------------
# 10. SAVE FEATURE IMPORTANCE
# ------------------------------------------

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    OUTPUT_DIR / "catboost_v2_feature_importance.csv",
    index=False
)


# ------------------------------------------
# 11. COMPLETION SUMMARY
# ------------------------------------------

print("\n--- FILES SAVED ---")
print("Model:", MODEL_DIR / "catboost_v2.cbm")

print(
    "Validation predictions:",
    OUTPUT_DIR / "catboost_v2_validation_predictions.csv"
)

print(
    "V2 evaluation:",
    OUTPUT_DIR / "catboost_v2_model_comparison.csv"
)

print(
    "V1 vs V2 comparison:",
    OUTPUT_DIR / "model_comparison_v1_vs_v2.csv"
)

print(
    "Feature importance:",
    OUTPUT_DIR / "catboost_v2_feature_importance.csv"
)

print("\n--- CATBOOST V2 TRAINING COMPLETE ---")
