
import pandas as pd
import os

# 1. Define dataset location
DATA_PATH = "data"


# 2. Create processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# 2. Identify CSV files
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

for file in files:

    print("\n" + "=" * 60)
    print(f"INSPECTING: {file}")
    print("=" * 60)

    path = os.path.join(DATA_PATH, file)

    # Load dataset
    df = pd.read_csv(path)

    # Basic information
    print("\n1. DATASET DIMENSIONS")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n3. DATA TYPES")
    print(df.dtypes)

    print("\n4. MISSING VALUES")
    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })
    print(missing)

    print("\n5. DUPLICATE RECORDS")
    print(df.duplicated().sum())

    print("\n6. FIRST 10 RECORDS")
    print(df.head(10).to_string())

    print("\n7. NUMERICAL SUMMARY")
    print(df.describe().to_string())

    # Save a small sample for sharing
    sample_name = file.replace(".csv", "_sample.csv")
    df.head(500).to_csv(
        os.path.join("data/processed", sample_name),
        index=False
    )

    # Save inspection report
    report_name = file.replace(".csv", "_report.txt")

    with open(
        os.path.join("data/processed", report_name),
        "w",
        encoding="utf-8"
    ) as report:

        report.write(f"Dataset: {file}\n")
        report.write(f"Dimensions: {df.shape}\n\n")
        report.write("Column Types:\n")
        report.write(df.dtypes.to_string())
        report.write("\n\nMissing Values:\n")
        report.write(missing.to_string())
        report.write("\n\nDuplicates:\n")
        report.write(str(df.duplicated().sum()))
        report.write("\n\nSummary:\n")
        report.write(df.describe(include="all").to_string())

print("\nDATA INSPECTION COMPLETED!")
