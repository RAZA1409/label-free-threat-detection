
from pathlib import Path
import pandas as pd
import numpy as np
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "outputs" / "preprocessing" / "cleaned_flows.parquet"
REPORT_FILE = PROJECT_DIR / "outputs" / "preprocessing" / "preprocessing_report.json"

print("Loading prepared dataset...")
df = pd.read_parquet(DATA_FILE)

with open(REPORT_FILE, encoding="utf-8") as f:
    report = json.load(f)

print("\n=== DATASET CHECK ===")
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))
print("Duplicate column names:", df.columns.duplicated().sum())
print("Missing labels:", df["Label"].isna().sum())
print("Missing timestamps:", df["Timestamp"].isna().sum())

print("\n=== TIMESTAMP CHECK ===")
print("Earliest:", df["Timestamp"].min())
print("Latest:", df["Timestamp"].max())
print("Chronologically sorted:",
      df["Timestamp"].is_monotonic_increasing)

print("\n=== NUMERIC FEATURE CHECK ===")
excluded = {"Timestamp", "Label"}
features = [
    c for c in df.columns
    if c not in excluded
    and pd.api.types.is_numeric_dtype(df[c])
]

print("Numeric feature count:", len(features))
print("Missing numeric cells:",
      f"{int(df[features].isna().sum().sum()):,}")

values = df[features].to_numpy(dtype=np.float64)
print("Infinite numeric cells:",
      f"{int(np.isinf(values).sum()):,}")

print("\n=== IDENTIFIER CHECK ===")
for col in [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port"
]:
    print(f"{col}: {'PRESENT' if col in df.columns else 'excluded'}")

print("\n=== LABEL DISTRIBUTION ===")
print(df["Label"].value_counts().to_string())

print("\n=== SOURCE FILE REPORT ===")
print("Files processed:", len(report["files_processed"]))
print("Rows saved in report:", f'{report["total_rows_saved"]:,}')

del values, df
print("\nVerification finished.")
