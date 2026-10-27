
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "GeneratedLabelledFlows"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cleaned_flows.parquet"
REPORT_FILE = OUTPUT_DIR / "preprocessing_report.json"

csv_files = sorted(DATA_DIR.rglob("*.csv"))

if not csv_files:
    raise SystemExit(f"No CSV files found in {DATA_DIR}")

frames = []
report = {
    "files_processed": [],
    "total_rows_loaded": 0,
    "total_empty_rows_removed": 0,
    "total_rows_saved": 0,
    "label_counts": {},
    "invalid_timestamps": 0,
    "infinite_values_replaced": 0,
    "missing_numeric_values_before_imputation": 0,
}

for path in csv_files:
    print(f"\nProcessing: {path.name}")

    # Use the encoding verified for this dataset.
    df = pd.read_csv(
        path,
        encoding="cp1252",
        low_memory=False,
        skip_blank_lines=True,
    )

    df.columns = df.columns.str.strip()
    rows_loaded = len(df)

    # Remove completely empty records only.
    empty_rows = int(df.isna().all(axis=1).sum())
    df = df.dropna(how="all").copy()

    # Normalize label whitespace but preserve attack names.
    df["Label"] = df["Label"].astype("string").str.strip()

    # Parse timestamps. CIC-IDS2017 dates are day/month/year
    # or month/day/year depending on the file; values are
    # ambiguous in some files, so we will check them later.
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"].astype("string").str.strip(),
        errors="coerce",
        dayfirst=True,
    )

    invalid_timestamps = int(df["Timestamp"].isna().sum())

    # Remove rows that lack essential evaluation metadata.
    missing_label = df["Label"].isna() | df["Label"].eq("")
    missing_timestamp = df["Timestamp"].isna()
    unusable_rows = missing_label | missing_timestamp

    removed_unusable = int(unusable_rows.sum())
    df = df.loc[~unusable_rows].copy()

    # Remove identifiers from model features, but preserve
    # Timestamp and Label for evaluation and dashboard use.
    identifier_columns = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
    ]

    df.drop(
        columns=[
            col for col in identifier_columns if col in df.columns
        ],
        inplace=True,
    )

    # Convert remaining feature columns to numeric where possible.
    feature_columns = [
        col for col in df.columns
        if col not in ["Timestamp", "Label"]
    ]

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    infinite_count = int(
        np.isinf(df[numeric_columns].to_numpy(dtype=float)).sum()
    )
    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    missing_numeric = int(df[numeric_columns].isna().sum().sum())

    # Save file-level information.
    counts = df["Label"].value_counts().to_dict()
    for label, count in counts.items():
        report["label_counts"][label] = (
            report["label_counts"].get(label, 0) + int(count)
        )

    report["files_processed"].append({
        "file": path.name,
        "rows_loaded": rows_loaded,
        "empty_rows_removed": empty_rows,
        "rows_removed_for_missing_label_or_timestamp": removed_unusable,
        "rows_saved": len(df),
        "invalid_timestamps_before_removal": invalid_timestamps,
        "infinite_values_replaced": infinite_count,
        "missing_numeric_values_after_infinity_replacement": missing_numeric,
    })

    report["total_rows_loaded"] += rows_loaded
    report["total_empty_rows_removed"] += empty_rows
    report["total_rows_saved"] += len(df)
    report["invalid_timestamps"] += invalid_timestamps
    report["infinite_values_replaced"] += infinite_count
    report["missing_numeric_values_before_imputation"] += missing_numeric

    frames.append(df)
    print(f"Rows retained: {len(df):,}")
    print(f"Infinite values replaced: {infinite_count:,}")

print("\nCombining files...")
combined = pd.concat(frames, ignore_index=True)
del frames

# Sort chronologically for time-based dashboard and evaluation.
combined.sort_values("Timestamp", inplace=True, ignore_index=True)

# Save the cleaned dataset.
combined.to_parquet(OUTPUT_FILE, index=False, compression="snappy")

report["combined_rows"] = len(combined)
report["columns_saved"] = combined.columns.tolist()
report["output_file"] = str(OUTPUT_FILE)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Rows saved: {len(combined):,}")
print(f"Columns saved: {len(combined.columns)}")
print(f"Parquet file: {OUTPUT_FILE}")
print(f"Report file: {REPORT_FILE}")
print("\nTop labels:")
print(combined["Label"].value_counts().head(15).to_string())
