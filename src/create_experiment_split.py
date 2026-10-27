
from pathlib import Path
import json
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    PROJECT_DIR / "outputs" / "preprocessing" / "cleaned_flows.parquet"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "split_report.json"

print("Loading cleaned dataset...")
df = pd.read_parquet(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Label"] = df["Label"].astype("string").str.strip()

if df["Timestamp"].isna().any() or df["Label"].isna().any():
    raise ValueError("Missing timestamp or label found.")

df["CaptureDate"] = df["Timestamp"].dt.date

# Monday is the benign-only training/calibration day.
monday_date = pd.Timestamp("2017-07-03").date()
monday = df[df["CaptureDate"] == monday_date].copy()

if monday.empty:
    raise ValueError("No Monday records found.")

if not monday["Label"].eq("BENIGN").all():
    raise ValueError(
        "Monday contains unexpected non-benign labels. "
        "Inspect the data before continuing."
    )

# Split by unique timestamps to avoid putting identical timestamps
# on both sides of the training/calibration boundary.
unique_times = monday["Timestamp"].drop_duplicates().sort_values()

if len(unique_times) < 2:
    raise ValueError("Not enough unique timestamps on Monday.")

cut_index = int(len(unique_times) * 0.70)
cut_index = max(1, min(cut_index, len(unique_times) - 1))
cutoff = unique_times.iloc[cut_index]

train = monday[monday["Timestamp"] < cutoff].copy()
calibration = monday[monday["Timestamp"] >= cutoff].copy()

# Test on all records from Tuesday through Friday.
test = df[df["CaptureDate"] > monday_date].copy()

if train.empty or calibration.empty or test.empty:
    raise ValueError("One or more experiment partitions are empty.")

if not train["Label"].eq("BENIGN").all():
    raise ValueError("Training partition is not benign-only.")

if not calibration["Label"].eq("BENIGN").all():
    raise ValueError("Calibration partition is not benign-only.")

# Ensure the same timestamp is not shared between Monday partitions.
if set(train["Timestamp"].unique()) & set(calibration["Timestamp"].unique()):
    raise ValueError("Timestamp overlap between training and calibration.")

# Features used for ML: exclude target, time and helper metadata.
excluded = {"Timestamp", "Label", "CaptureDate"}
feature_columns = [
    c for c in df.columns
    if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]

if not feature_columns:
    raise ValueError("No numeric model features found.")

# Persist partitions for repeatable experiments.
train.to_parquet(OUTPUT_DIR / "train_benign.parquet", index=False)
calibration.to_parquet(
    OUTPUT_DIR / "calibration_benign.parquet", index=False
)
test.to_parquet(OUTPUT_DIR / "test_tue_fri.parquet", index=False)

report = {
    "strategy": "Chronological Monday benign train/calibration; Tuesday-Friday test",
    "training_rows": len(train),
    "calibration_rows": len(calibration),
    "test_rows": len(test),
    "training_labels": train["Label"].value_counts().to_dict(),
    "calibration_labels": calibration["Label"].value_counts().to_dict(),
    "test_labels": test["Label"].value_counts().to_dict(),
    "training_start": str(train["Timestamp"].min()),
    "training_end": str(train["Timestamp"].max()),
    "calibration_start": str(calibration["Timestamp"].min()),
    "calibration_end": str(calibration["Timestamp"].max()),
    "test_start": str(test["Timestamp"].min()),
    "test_end": str(test["Timestamp"].max()),
    "cutoff_timestamp": str(cutoff),
    "target_calibration_fpr": 0.01,
    "feature_columns": feature_columns,
    "excluded_columns": ["Timestamp", "Label", "CaptureDate"],
    "test_attack_categories": sorted(
        test.loc[test["Label"] != "BENIGN", "Label"].unique().tolist()
    ),
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print("\n=== EXPERIMENT SPLIT CREATED ===")
print(f"Training rows:    {len(train):,}")
print(f"Calibration rows: {len(calibration):,}")
print(f"Test rows:        {len(test):,}")
print(f"Training cutoff:  {cutoff}")

print("\nTest label distribution:")
print(test["Label"].value_counts().to_string())

print(f"\nReport saved to: {REPORT_FILE}")
print("Feature count:", len(feature_columns))
print("\nNo model has been trained yet.")
