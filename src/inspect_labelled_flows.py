
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "GeneratedLabelledFlows"

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in: {DATA_DIR}")
    raise SystemExit(1)

print(f"Found {len(csv_files)} CSV files.")

for path in csv_files:
    print("\n" + "=" * 75)
    print("File:", path.name)
    print(f"Size: {path.stat().st_size / (1024**2):.2f} MB")

    try:
        # Read only a few rows; don't load large files into RAM.
        df = pd.read_csv(path, nrows=5, low_memory=False)

        # Remove leading/trailing whitespace from column names.
        df.columns = df.columns.str.strip()

        print("Column count:", len(df.columns))
        print("Columns:")
        print(df.columns.tolist())

        time_columns = [
            c for c in df.columns
            if any(word in c.lower() for word in
                   ["timestamp", "time", "date"])
        ]

        print("Potential time columns:", time_columns)

        for column in time_columns:
            print(f"Sample values in {column!r}:")
            print(df[column].head(3).tolist())

        label_columns = [
            c for c in df.columns
            if c.lower() == "label"
        ]

        if label_columns:
            print("Sample labels:")
            print(df[label_columns[0]].head(5).tolist())

    except Exception as error:
        print("Inspection error:", error)
