
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

csv_files = sorted(DATASET_DIR.rglob("*.csv"))

for file_path in csv_files:
    print("\n" + "=" * 80)
    print(file_path.name)

    # Read only a few rows; do not load the full file
    df = pd.read_csv(file_path, nrows=5, low_memory=False)

    print("\nFirst 12 column names:")
    for column in df.columns[:12]:
        print(repr(column))

    # Find timestamp-related columns, regardless of capitalization
    timestamp_columns = [
        column for column in df.columns
        if "time" in column.lower()
    ]

    print("\nTimestamp-related columns:", timestamp_columns)

    for column in timestamp_columns:
        print(f"\nSample values from {column!r}:")
        print(df[column].head(5).to_list())

    label_columns = [
        column for column in df.columns
        if column.strip().lower() == "label"
    ]

    if label_columns:
        print("\nSample labels:")
        print(df[label_columns[0]].head(5).to_list())
