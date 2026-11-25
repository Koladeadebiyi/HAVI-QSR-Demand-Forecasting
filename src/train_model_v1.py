
# ==========================================
# HAVI QSR DEMAND FORECASTING
# FIRST CATBOOST MODEL
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

training_df = pd.read_csv(DATA_DIR / "training_data.csv")
validation_df = pd.read_csv(DATA_DIR / "validation_data.csv")

print("--- DATASET LOADING ---")
print("Training shape:", training_df.shape)
print("Validation shape:", validation_df.shape)


# ------------------------------------------
# 3. PREPARE FEATURES AND TARGET
# ------------------------------------------

target_column = "quantity"

excluded_columns = ["date", target_column]

feature_columns = [
    col for col in training_df.columns
    if col not in excluded_columns
]

categorical_features = [
    "restaurant_id",
    "menu_item_id",
    "category"
]

X_train = training_df[feature_columns].copy()
y_train = training_df[target_column].copy()

X_validation = validation_df[feature_columns].copy()
y_validation = validation_df[target_column].copy()

# Convert categorical variables to strings
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_validation[col] = X_validation[col].astype(str)

# Confirm matching feature structure
assert list(X_train.columns) == list(X_validation.columns)

print("\n--- MODEL DATA PREPARATION ---")
print("Training observations:", len(X_train))
print("Validation observations:", len(X_validation))
print("Number of features:", len(feature_columns))
print("Categorical features:", categorical_features)


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

print("\n--- CATBOOST MODEL TRAINING ---")

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

# Quantities cannot be negative
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

mae = mean_absolute_error(
    actual_values, validation_predictions
)

rmse = np.sqrt(
    mean_squared_error(actual_values, validation_predictions)
)

improvement = baseline_wmape - catboost_wmape

print("\n--- MODEL EVALUATION RESULTS ---")
print("Baseline wMAPE:", round(baseline_wmape, 4), "%")
print("CatBoost wMAPE:", round(catboost_wmape, 4), "%")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("wMAPE improvement:", round(improvement, 4), "percentage points")


# ------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------

model.save_model(MODEL_DIR / "catboost_v1.cbm")


# ------------------------------------------
# 8. SAVE VALIDATION PREDICTIONS
# ------------------------------------------

validation_results = validation_df[
    ["date", "restaurant_id", "menu_item_id", "quantity"]
].copy()

validation_results["predicted_quantity"] = validation_predictions

validation_results.to_csv(
    OUTPUT_DIR / "catboost_v1_validation_predictions.csv",
    index=False
)


# ------------------------------------------
# 9. SAVE EVALUATION SUMMARY
# ------------------------------------------

evaluation_summary = pd.DataFrame({
    "model": ["28-day historical average", "CatBoost v1"],
    "wMAPE": [baseline_wmape, catboost_wmape],
    "MAE": [
        mean_absolute_error(actual_values, baseline_predictions),
        mae
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(actual_values, baseline_predictions)),
        rmse
    ]
})

evaluation_summary.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\n--- FILES SAVED ---")
print("Model:", MODEL_DIR / "catboost_v1.cbm")
print("Validation predictions:",
      OUTPUT_DIR / "catboost_v1_validation_predictions.csv")
print("Evaluation summary:",
      OUTPUT_DIR / "model_comparison.csv")

print("\n--- TRAINING COMPLETE ---")
