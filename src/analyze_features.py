
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "experiment" / "train_benign.parquet"
OUT = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading benign training data...")
df = pd.read_parquet(DATA)

excluded = {"Timestamp", "Label", "CaptureDate"}
features = [
    c for c in df.columns
    if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]

X = df[features].replace([np.inf, -np.inf], np.nan)

# Drop features that contain no usable values or are constant.
usable = X.columns[X.notna().any()]
X = X[usable]
nunique = X.nunique(dropna=True)
constant_features = nunique[nunique <= 1].index.tolist()
X = X.drop(columns=constant_features)

print(f"Training rows: {len(df):,}")
print(f"Numeric features before filtering: {len(features)}")
print(f"Constant features removed: {len(constant_features)}")
print(f"Features retained: {X.shape[1]}")

# Inspect skewness and missingness.
summary = pd.DataFrame({
    "missing_fraction": X.isna().mean(),
    "zero_fraction": X.eq(0).mean(),
    "skewness": X.skew(),
    "unique_values": X.nunique(dropna=True),
    "median": X.median(),
    "q25": X.quantile(0.25),
    "q75": X.quantile(0.75),
})

summary["abs_skewness"] = summary["skewness"].abs()
summary = summary.sort_values("abs_skewness", ascending=False)

summary.to_csv(OUT / "benign_feature_analysis.csv")

print("\n=== MOST SKEWED FEATURES ===")
print(summary.head(20).to_string())

print("\n=== FEATURES WITH MOST ZEROS ===")
print(
    summary.sort_values("zero_fraction", ascending=False)
    [["zero_fraction", "missing_fraction", "unique_values"]]
    .head(15)
    .to_string()
)

print("\nSaved report to outputs/evaluation/benign_feature_analysis.csv")
