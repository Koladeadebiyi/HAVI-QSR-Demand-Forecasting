
import pandas as pd
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_DIR / "data" / "qsr_demand_dataset.csv"

TRAINING_DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_data"
    / "forecast_origin_training_data.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_data"
    / "forecast_origin_training_data_v3_features.csv"
)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading raw historical data...")
raw = pd.read_csv(RAW_DATA_PATH)

print("Loading forecast-origin training data...")
training = pd.read_csv(TRAINING_DATA_PATH)

raw["date"] = pd.to_datetime(raw["date"])
training["forecast_origin"] = pd.to_datetime(
    training["forecast_origin"]
)

raw["quantity"] = pd.to_numeric(
    raw["quantity"], errors="coerce"
)

print("\nRaw data shape:", raw.shape)
print("Training data shape:", training.shape)

# ============================================================
# 3. PREPARE OBSERVED HISTORICAL QUANTITIES
# ============================================================

# Missing quantities are excluded.
# A recorded quantity of zero remains a genuine zero-sales day.

observed = raw.loc[
    raw["quantity"].notna(),
    ["date", "restaurant_id", "menu_item_id", "quantity"]
].copy()

observed["is_zero_sale"] = (
    observed["quantity"] == 0
).astype(int)

observed = observed.sort_values("date")

print("\nObserved quantity records:", len(observed))

# ============================================================
# 4. CALCULATE ZERO RATES AT EACH FORECAST ORIGIN
# ============================================================

windows = [28, 90, 365]

feature_frames = []

origins = sorted(
    training["forecast_origin"].dropna().unique()
)

for origin in origins:

    origin = pd.Timestamp(origin)

    print(f"\nCalculating features for origin: {origin.date()}")

    # Only observations available on or before this origin.
    history = observed.loc[
        observed["date"] <= origin
    ].copy()

    origin_features = None

    for window in windows:

        window_start = origin - pd.Timedelta(
            days=window - 1
        )

        window_data = history.loc[
            history["date"] >= window_start
        ]

        summary = (
            window_data
            .groupby(
                ["restaurant_id", "menu_item_id"],
                as_index=False
            )
            .agg(
                **{
                    f"observed_days_{window}d": (
                        "is_zero_sale", "count"
                    ),
                    f"zero_days_{window}d": (
                        "is_zero_sale", "sum"
                    )
                }
            )
        )

        summary[f"zero_rate_{window}d"] = (
            summary[f"zero_days_{window}d"]
            / summary[f"observed_days_{window}d"]
        )

        # Retain the forecast origin as part of the merge key.
        if origin_features is None:
            origin_features = summary
        else:
            origin_features = origin_features.merge(
                summary,
                on=["restaurant_id", "menu_item_id"],
                how="outer"
            )

    origin_features["forecast_origin"] = origin

    feature_frames.append(origin_features)

# Combine all historical forecast origins.
zero_features = pd.concat(
    feature_frames,
    ignore_index=True
)

# ============================================================
# 5. MERGE FEATURES INTO TRAINING DATA
# ============================================================

feature_columns = [
    "forecast_origin",
    "restaurant_id",
    "menu_item_id",
    "zero_rate_28d",
    "zero_rate_90d",
    "zero_rate_365d",
    "observed_days_28d",
    "observed_days_90d",
    "observed_days_365d",
    "zero_days_28d",
    "zero_days_90d",
    "zero_days_365d"
]

zero_features = zero_features[feature_columns]

training_v3 = training.merge(
    zero_features,
    on=[
        "forecast_origin",
        "restaurant_id",
        "menu_item_id"
    ],
    how="left",
    validate="many_to_one"
)

# ============================================================
# 6. VALIDATION CHECKS
# ============================================================

rate_columns = [
    "zero_rate_28d",
    "zero_rate_90d",
    "zero_rate_365d"
]

print("\n========== VALIDATION CHECKS ==========")

print("Original training rows:", len(training))
print("New training rows:", len(training_v3))

print("\nMissing zero-rate features:")
print(training_v3[rate_columns].isna().sum())

print("\nZero-rate descriptive statistics:")
print(training_v3[rate_columns].describe())

print("\nZero-rate values outside 0–1:")
for column in rate_columns:
    invalid = (
        (training_v3[column] < 0)
        | (training_v3[column] > 1)
    ).sum()

    print(f"{column}: {invalid}")

# Confirm that the merge did not multiply rows.
assert len(training_v3) == len(training), (
    "Row count changed after merging features."
)

# Confirm the expected forecast origins remain.
assert (
    training_v3["forecast_origin"].nunique()
    == training["forecast_origin"].nunique()
)

# ============================================================
# 7. SAVE ENRICHED DATA
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

training_v3.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved enriched training data to:")
print(OUTPUT_PATH)

print("\nFinal training data shape:", training_v3.shape)
print("\nStep 1A completed.")



import pandas as pd
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_DIR / "data" / "qsr_demand_dataset.csv"

VALIDATION_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_data"
    / "validation_data.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_data"
    / "validation_data_v3_features.csv"
)

FORECAST_ORIGIN = pd.Timestamp("2025-06-30")
WINDOWS = [28, 90, 365]

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading raw historical data...")
raw = pd.read_csv(RAW_DATA_PATH)

print("Loading validation data...")
validation = pd.read_csv(VALIDATION_PATH)

raw["date"] = pd.to_datetime(raw["date"])
validation["date"] = pd.to_datetime(validation["date"])

raw["quantity"] = pd.to_numeric(
    raw["quantity"], errors="coerce"
)

print("\nRaw data shape:", raw.shape)
print("Validation data shape:", validation.shape)

# ============================================================
# 3. SELECT OBSERVED HISTORY AVAILABLE AT ORIGIN
# ============================================================

history = raw.loc[
    (raw["date"] <= FORECAST_ORIGIN)
    & raw["quantity"].notna(),
    ["date", "restaurant_id", "menu_item_id", "quantity"]
].copy()

history["is_zero_sale"] = (
    history["quantity"] == 0
).astype(int)

print("\nObserved historical records:", len(history))

# ============================================================
# 4. CALCULATE ZERO-SALES FEATURES
# ============================================================

zero_features = None

for window in WINDOWS:

    window_start = (
        FORECAST_ORIGIN
        - pd.Timedelta(days=window - 1)
    )

    window_data = history.loc[
        history["date"] >= window_start
    ]

    summary = (
        window_data
        .groupby(
            ["restaurant_id", "menu_item_id"],
            as_index=False
        )
        .agg(
            **{
                f"observed_days_{window}d": (
                    "is_zero_sale", "count"
                ),
                f"zero_days_{window}d": (
                    "is_zero_sale", "sum"
                )
            }
        )
    )

    summary[f"zero_rate_{window}d"] = (
        summary[f"zero_days_{window}d"]
        / summary[f"observed_days_{window}d"]
    )

    if zero_features is None:
        zero_features = summary
    else:
        zero_features = zero_features.merge(
            summary,
            on=["restaurant_id", "menu_item_id"],
            how="outer"
        )

# ============================================================
# 5. MERGE WITH VALIDATION DATA
# ============================================================

validation_v3 = validation.merge(
    zero_features,
    on=["restaurant_id", "menu_item_id"],
    how="left",
    validate="many_to_one"
)

# ============================================================
# 6. VALIDATION CHECKS
# ============================================================

rate_columns = [
    "zero_rate_28d",
    "zero_rate_90d",
    "zero_rate_365d"
]

print("\n========== VALIDATION CHECKS ==========")

print("Original rows:", len(validation))
print("New rows:", len(validation_v3))

print("\nMissing zero-rate features:")
print(validation_v3[rate_columns].isna().sum())

print("\nZero-rate descriptive statistics:")
print(validation_v3[rate_columns].describe())

print("\nUnique restaurants:", validation_v3["restaurant_id"].nunique())
print("Unique menu items:", validation_v3["menu_item_id"].nunique())
print(
    "Unique restaurant-item combinations:",
    validation_v3[
        ["restaurant_id", "menu_item_id"]
    ].drop_duplicates().shape[0]
)

print("\nValidation dates:")
print(validation_v3["date"].min())
print(validation_v3["date"].max())

for column in rate_columns:

    invalid = (
        (validation_v3[column] < 0)
        | (validation_v3[column] > 1)
    ).sum()

    print(f"{column} outside 0–1: {invalid}")

# Confirm row count and uniqueness.
assert len(validation_v3) == len(validation), (
    "Row count changed after merging."
)

assert validation_v3[
    ["date", "restaurant_id", "menu_item_id"]
].duplicated().sum() == validation[
    ["date", "restaurant_id", "menu_item_id"]
].duplicated().sum(), (
    "Unexpected duplicate-key changes."
)

assert validation_v3[rate_columns].notna().all().all(), (
    "Missing zero-rate features detected."
)

# ============================================================
# 7. SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

validation_v3.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved validation data to:")
print(OUTPUT_PATH)

print("\nFinal validation shape:", validation_v3.shape)
print("\nPhase F completed successfully.")
