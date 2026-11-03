
from pathlib import Path
import pandas as pd
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "GeneratedLabelledFlows"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.rglob("*.csv"))
if not files:
    raise SystemExit(f"No CSV files found in {DATA_DIR}")

report = []

for path in files:
    print(f"\nChecking: {path.name}")

    rows = 0
    valid_count = 0
    invalid_count = 0
    min_time = None
    max_time = None
    label_counts = {}

    # Read only timestamp and label columns in chunks.
    for chunk in pd.read_csv(
        path,
        encoding="cp1252",
        usecols=lambda c: c.strip() in {"Timestamp", "Label"},
        chunksize=100_000,
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.dropna(how="all")

        timestamps = pd.to_datetime(
            chunk["Timestamp"].astype("string").str.strip(),
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

        rows += len(chunk)
        valid_count += int(timestamps.notna().sum())
        invalid_count += int(timestamps.isna().sum())

        valid_times = timestamps.dropna()
        if not valid_times.empty:
            chunk_min = valid_times.min()
            chunk_max = valid_times.max()
            min_time = chunk_min if min_time is None else min(min_time, chunk_min)
            max_time = chunk_max if max_time is None else max(max_time, chunk_max)

        labels = chunk["Label"].astype("string").str.strip()
        counts = labels.value_counts(dropna=False).to_dict()
        for label, count in counts.items():
            key = "<MISSING>" if pd.isna(label) else str(label)
            label_counts[key] = label_counts.get(key, 0) + int(count)

    item = {
        "file": path.name,
        "rows": rows,
        "valid_timestamps": valid_count,
        "invalid_timestamps": invalid_count,
        "start": str(min_time),
        "end": str(max_time),
        "labels": label_counts,
    }
    report.append(item)

    print(f"Rows: {rows:,}")
    print(f"Valid timestamps: {valid_count:,}")
    print(f"Invalid timestamps: {invalid_count:,}")
    print(f"Start: {min_time}")
    print(f"End: {max_time}")

report_path = OUTPUT_DIR / "timestamp_audit.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nAudit report saved to: {report_path}")
print("\nTimestamp audit complete.")
