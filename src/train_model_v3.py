
import pandas as pd
import numpy as np

from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_data"
    / "forecast_origin_training_data_v3_features.csv"
)

VALIDATION_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_data"
    / "validation_data_v3_features.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "model_evaluation"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("LOADING CATBOOST V3 DATA")
print("=" * 65)

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

print("Training shape:", train.shape)
print("Validation shape:", validation.shape)

# ============================================================
# 3. DEFINE MODEL FEATURES
# ============================================================

# Preserve the same 24 features used in v2.
# Add only the three historical zero-sales rates.

feature_columns = [
    # Restaurant and product information
    "restaurant_id",
    "menu_item_id",
    "category",
    "unit_price",

    # Calendar features
    "day_of_week_num",
    "is_weekend",
    "month",
    "quarter",
    "year",
    "day_of_month",
    "day_of_year",
    "week_of_year",

    # Commercial and event indicators
    "is_promotion",
    "is_holiday",
    "is_special_event",

    # Historical demand features
    "mean_7d",
    "mean_28d",
    "mean_56d",
    "mean_90d",
    "mean_365d",
    "std_28d",
    "std_90d",
    "observed_7d",
    "observed_28d",

    # New intermittent-demand features
    "zero_rate_28d",
    "zero_rate_90d",
    "zero_rate_365d"
]

categorical_features = [
    "restaurant_id",
    "menu_item_id",
    "category"
]

print("\nNumber of model features:", len(feature_columns))

# Confirm all required features exist.
for column in feature_columns:
    assert column in train.columns, (
        f"Missing training feature: {column}"
    )
    assert column in validation.columns, (
        f"Missing validation feature: {column}"
    )

# ============================================================
# 4. PREPARE TARGETS AND REMOVE MISSING TARGETS
# ============================================================

train = train.dropna(subset=["quantity"]).copy()
validation = validation.dropna(subset=["quantity"]).copy()

train["quantity"] = pd.to_numeric(
    train["quantity"], errors="coerce"
)

validation["quantity"] = pd.to_numeric(
    validation["quantity"], errors="coerce"
)

train = train.dropna(subset=["quantity"]).copy()
validation = validation.dropna(subset=["quantity"]).copy()

assert (train["quantity"] >= 0).all()
assert (validation["quantity"] >= 0).all()

# Classifier target:
# 1 = positive recorded sales
# 0 = zero recorded sales

train["has_sales"] = (
    train["quantity"] > 0
).astype(int)

validation["has_sales"] = (
    validation["quantity"] > 0
).astype(int)

print("\n========== TARGET DISTRIBUTION ==========")

print("\nTraining target:")
print(train["has_sales"].value_counts())
print(
    train["has_sales"]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

print("\nValidation target:")
print(validation["has_sales"].value_counts())
print(
    validation["has_sales"]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

# ============================================================
# 5. PREPARE MODEL MATRICES
# ============================================================

X_train = train[feature_columns].copy()
X_validation = validation[feature_columns].copy()

for column in categorical_features:
    X_train[column] = X_train[column].astype(str)
    X_validation[column] = X_validation[column].astype(str)

y_train_classifier = train["has_sales"]
y_validation_classifier = validation["has_sales"]

y_train_regression = train["quantity"]
y_validation = validation["quantity"]

# Check missing feature values.
print("\nMissing training feature values:")
print(X_train.isna().sum().sum())

print("\nMissing validation feature values:")
print(X_validation.isna().sum().sum())

assert X_train.isna().sum().sum() == 0
assert X_validation.isna().sum().sum() == 0

# ============================================================
# 6. TRAIN STAGE 1: POSITIVE-SALES CLASSIFIER
# ============================================================

print("\n" + "=" * 65)
print("STAGE 1: TRAINING POSITIVE-SALES CLASSIFIER")
print("=" * 65)

classifier = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="Logloss",
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=100,
    allow_writing_files=False
)

classifier.fit(
    X_train,
    y_train_classifier,
    cat_features=categorical_features,
    eval_set=(X_validation, y_validation_classifier),
    use_best_model=True
)

classifier_path = MODEL_DIR / "catboost_v3_classifier.cbm"

classifier.save_model(classifier_path)

print("\nClassifier best iteration:")
print(classifier.get_best_iteration())

print("Classifier saved:", classifier_path)

# ============================================================
# 7. CLASSIFIER VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFIER VALIDATION RESULTS")
print("=" * 65)

positive_probability = classifier.predict_proba(
    X_validation
)[:, 1]

# Default classification threshold.
# This is for diagnostic classification metrics only.
# Final demand predictions use probabilities directly.

classifier_predictions = (
    positive_probability >= 0.5
).astype(int)

classifier_metrics = {
    "accuracy": accuracy_score(
        y_validation_classifier,
        classifier_predictions
    ),
    "precision": precision_score(
        y_validation_classifier,
        classifier_predictions,
        zero_division=0
    ),
    "recall": recall_score(
        y_validation_classifier,
        classifier_predictions,
        zero_division=0
    ),
    "f1": f1_score(
        y_validation_classifier,
        classifier_predictions,
        zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_validation_classifier,
        positive_probability
    ),
    "log_loss": log_loss(
        y_validation_classifier,
        positive_probability
    )
}

print(pd.Series(classifier_metrics))

print("\nAverage predicted probability of positive sales:")
print(round(positive_probability.mean(), 5))

print("\nActual positive-sales rate:")
print(round(y_validation_classifier.mean(), 5))

# ============================================================
# 8. TRAIN STAGE 2: POSITIVE-QUANTITY REGRESSOR
# ============================================================

print("\n" + "=" * 65)
print("STAGE 2: TRAINING POSITIVE-QUANTITY REGRESSOR")
print("=" * 65)

# The regressor learns only from positive-sales observations.

positive_train_mask = train["quantity"] > 0
positive_validation_mask = validation["quantity"] > 0

X_train_positive = X_train.loc[
    positive_train_mask
].copy()

y_train_positive = train.loc[
    positive_train_mask, "quantity"
].copy()

X_validation_positive = X_validation.loc[
    positive_validation_mask
].copy()

y_validation_positive = validation.loc[
    positive_validation_mask, "quantity"
].copy()

print("Positive training observations:", len(X_train_positive))
print("Positive validation observations:", len(X_validation_positive))

regressor = CatBoostRegressor(
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

regressor.fit(
    X_train_positive,
    y_train_positive,
    cat_features=categorical_features,
    eval_set=(
        X_validation_positive,
        y_validation_positive
    ),
    use_best_model=True
)

regressor_path = MODEL_DIR / "catboost_v3_regressor.cbm"

regressor.save_model(regressor_path)

print("\nRegressor best iteration:")
print(regressor.get_best_iteration())

print("Regressor saved:", regressor_path)

# ============================================================
# 9. COMBINE BOTH MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("COMBINING CLASSIFIER AND REGRESSOR")
print("=" * 65)

# Predict conditional quantity for every validation observation.
conditional_quantity = regressor.predict(X_validation)

# Expected demand = probability of positive sales
# multiplied by predicted quantity conditional on positive sales.

final_predictions = (
    positive_probability * conditional_quantity
)

# Demand cannot be negative.
final_predictions = np.maximum(final_predictions, 0)

# ============================================================
# 10. EVALUATE COMBINED DEMAND FORECAST
# ============================================================

actual = y_validation.to_numpy()

mae = mean_absolute_error(
    actual,
    final_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        final_predictions
    )
)

wmape = (
    np.sum(np.abs(actual - final_predictions))
    / np.sum(np.abs(actual))
) * 100

bias = (
    np.sum(final_predictions - actual)
    / np.sum(actual)
) * 100

overall_metrics = {
    "model": "CatBoost v3 Two-Stage",
    "observations": len(actual),
    "wMAPE": wmape,
    "MAE": mae,
    "RMSE": rmse,
    "bias_pct": bias,
    "classifier_best_iteration": classifier.get_best_iteration(),
    "regressor_best_iteration": regressor.get_best_iteration()
}

print("\n========== COMBINED V3 RESULTS ==========")
print(pd.Series(overall_metrics))

# ============================================================
# 11. SAVE VALIDATION PREDICTIONS
# ============================================================

predictions = validation[
    [
        "date",
        "restaurant_id",
        "menu_item_id",
        "category",
        "quantity"
    ]
].copy()

predictions = predictions.rename(
    columns={"quantity": "actual_quantity"}
)

predictions["probability_positive_sales"] = positive_probability

predictions["predicted_quantity_if_positive"] = (
    conditional_quantity
)

predictions["predicted_quantity"] = final_predictions

predictions["predicted_sales_class"] = (
    classifier_predictions
)

predictions["actual_sales_class"] = (
    y_validation_classifier.to_numpy()
)

predictions["residual"] = (
    predictions["actual_quantity"]
    - predictions["predicted_quantity"]
)

predictions.to_csv(
    OUTPUT_DIR / "catboost_v3_validation_predictions.csv",
    index=False
)

# ============================================================
# 12. SAVE MODEL METRICS
# ============================================================

pd.DataFrame([overall_metrics]).to_csv(
    OUTPUT_DIR / "catboost_v3_model_metrics.csv",
    index=False
)

pd.DataFrame(
    [classifier_metrics]
).to_csv(
    OUTPUT_DIR / "catboost_v3_classifier_metrics.csv",
    index=False
)

# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

classifier_importance = pd.DataFrame({
    "feature": feature_columns,
    "classifier_importance": classifier.get_feature_importance()
})

regressor_importance = pd.DataFrame({
    "feature": feature_columns,
    "regressor_importance": regressor.get_feature_importance()
})

importance = classifier_importance.merge(
    regressor_importance,
    on="feature"
)

importance.to_csv(
    OUTPUT_DIR / "catboost_v3_feature_importance.csv",
    index=False
)

print("\n========== SAVED OUTPUTS ==========")

print("Classifier:", classifier_path)
print("Regressor:", regressor_path)

print(
    OUTPUT_DIR / "catboost_v3_validation_predictions.csv"
)

print(
    OUTPUT_DIR / "catboost_v3_model_metrics.csv"
)

print(
    OUTPUT_DIR / "catboost_v3_classifier_metrics.csv"
)

print(
    OUTPUT_DIR / "catboost_v3_feature_importance.csv"
)

print("\nCATBOOST V3 TRAINING COMPLETED.")
