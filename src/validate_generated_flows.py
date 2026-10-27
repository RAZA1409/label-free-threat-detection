
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "GeneratedLabelledFlows"

csv_files = sorted(DATA_DIR.rglob("*.csv"))

if not csv_files:
    raise SystemExit(f"No CSV files found in: {DATA_DIR}")

print(f"Found {len(csv_files)} CSV files\n")

for path in csv_files:
    print("=" * 75)
    print(f"File: {path.name}")

    df = pd.read_csv(
    path,
    low_memory=False,
    encoding="cp1252",
    skip_blank_lines=True
    )

# Remove rows that are entirely empty
    df = df.dropna(how="all")

    
    df.columns = df.columns.str.strip()

    # Parse timestamps
    if "Timestamp" in df.columns:
        timestamps = pd.to_datetime(
            df["Timestamp"].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True
        )

        valid = timestamps.dropna()

        print(f"Rows: {len(df):,}")
        print(f"Valid timestamps: {len(valid):,}")
        print(f"Invalid timestamps: {timestamps.isna().sum():,}")

        if not valid.empty:
            print(f"Start time: {valid.min()}")
            print(f"End time:   {valid.max()}")
            print(
                "Time span (hours):",
                round(
                    (valid.max() - valid.min()).total_seconds() / 3600,
                    3
                )
            )
    else:
        print("WARNING: Timestamp column missing")

    if "Label" in df.columns:
        print("\nLabel counts:")
        print(df["Label"].astype(str).str.strip().value_counts().to_string())

    # Check numeric feature quality
    numeric = df.select_dtypes(include=[np.number])
    missing = int(df.isna().sum().sum())
    infinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())

    print(f"\nMissing cells: {missing:,}")
    print(f"Infinite numeric cells: {infinite:,}")
    print(f"Numeric columns: {len(numeric.columns)}")

    # Verify that network identifiers are present
    for column in [
        "Flow ID", "Source IP", "Destination IP",
        "Source Port", "Destination Port", "Protocol"
    ]:
        print(f"{column}: {'present' if column in df.columns else 'missing'}")

    del df
