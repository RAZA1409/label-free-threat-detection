
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "dataset_inspection"

CHUNK_SIZE = 100_000


def inspect_csv(file_path):
    """Inspect a large CSV in chunks without loading it all into RAM."""

    print(f"\nInspecting: {file_path.name}")

    row_count = 0
    label_counts = Counter()
    missing_counts = Counter()
    infinite_counts = Counter()
    duplicate_rows_within_chunks = 0
    timestamp_min = None
    timestamp_max = None
    columns = []
    numeric_columns = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):
        # Normalize accidental spaces in column names
        chunk.columns = chunk.columns.str.strip()

        if not columns:
            columns = chunk.columns.tolist()
            numeric_columns = chunk.select_dtypes(
                include=np.number
            ).columns.tolist()

            print(f"  Columns: {len(columns)}")

        row_count += len(chunk)

        # Count missing values
        missing = chunk.isna().sum()
        for column, count in missing.items():
            missing_counts[column] += int(count)

        # Count positive/negative infinity in numeric features
        numeric = chunk.select_dtypes(include=np.number)
        if not numeric.empty:
            inf_mask = np.isinf(numeric.to_numpy())
            inf_per_column = pd.DataFrame(
                inf_mask,
                columns=numeric.columns
            ).sum()

            for column, count in inf_per_column.items():
                infinite_counts[column] += int(count)

        # Count labels without using them for model training
        label_column = next(
            (
                column for column in chunk.columns
                if column.lower() == "label"
            ),
            None
        )

        if label_column:
            labels = (
                chunk[label_column]
                .fillna("<MISSING>")
                .astype(str)
                .str.strip()
            )
            label_counts.update(labels.value_counts().to_dict())

        # Parse timestamps when available
        timestamp_column = next(
            (
                column for column in chunk.columns
                if column.lower() in {"timestamp", "flow start"}
            ),
            None
        )

        if timestamp_column:
            parsed = pd.to_datetime(
                chunk[timestamp_column],
                errors="coerce",
                dayfirst=False
            ).dropna()

            if not parsed.empty:
                current_min = parsed.min().isoformat()
                current_max = parsed.max().isoformat()

                if timestamp_min is None or current_min < timestamp_min:
                    timestamp_min = current_min

                if timestamp_max is None or current_max > timestamp_max:
                    timestamp_max = current_max

        # This detects duplicates within each chunk only.
        duplicate_rows_within_chunks += int(
            chunk.duplicated().sum()
        )

        if chunk_number % 10 == 0:
            print(f"  Processed {row_count:,} rows...")

    file_size_mb = file_path.stat().st_size / (1024 ** 2)

    report = {
        "file": file_path.name,
        "file_size_mb": round(file_size_mb, 2),
        "rows": row_count,
        "columns_count": len(columns),
        "columns": columns,
        "numeric_columns_count": len(numeric_columns),
        "label_counts": dict(label_counts),
        "missing_values": {
            k: v for k, v in missing_counts.items() if v > 0
        },
        "infinite_values": {
            k: v for k, v in infinite_counts.items() if v > 0
        },
        "duplicate_rows_within_chunks_only": duplicate_rows_within_chunks,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
    }

    print(f"  Rows: {row_count:,}")
    print(f"  Features/columns: {len(columns)}")
    print(f"  Labels: {dict(label_counts)}")
    print(f"  Missing cells: {sum(missing_counts.values()):,}")
    print(f"  Infinite numeric cells: {sum(infinite_counts.values()):,}")
    print(f"  Duplicate rows within chunks: {duplicate_rows_within_chunks:,}")
    print(f"  Timestamp range: {timestamp_min} to {timestamp_max}")

    return report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Search the dataset directory and its subfolders
    csv_files = sorted(
        path for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    if not csv_files:
        print(f"No CSV files found inside: {DATASET_DIR}")
        print("Check that your eight dataset files are in the dataset folder.")
        return

    print(f"Found {len(csv_files)} CSV files.")
    print("Labels are being inspected only, not used for training.")

    reports = []

    for file_path in csv_files:
        try:
            reports.append(inspect_csv(file_path))
        except Exception as error:
            print(f"ERROR inspecting {file_path.name}: {error}")

    # Save a complete report for later analysis
    report_path = OUTPUT_DIR / "inspection_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(reports, report_file, indent=2)

    # Save a compact summary
    summary = []

    for report in reports:
        summary.append({
            "file": report["file"],
            "size_mb": report["file_size_mb"],
            "rows": report["rows"],
            "columns": report["columns_count"],
            "labels": json.dumps(report["label_counts"]),
            "missing_cells": sum(report["missing_values"].values()),
            "infinite_cells": sum(report["infinite_values"].values()),
            "timestamp_min": report["timestamp_min"],
            "timestamp_max": report["timestamp_max"],
        })

    summary_path = OUTPUT_DIR / "file_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print(f"Files inspected successfully: {len(reports)}/{len(csv_files)}")
    print(f"Detailed report: {report_path}")
    print(f"Summary CSV:     {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
