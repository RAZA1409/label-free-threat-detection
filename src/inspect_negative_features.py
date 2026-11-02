
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "outputs/experiment/train_benign.parquet"

FEATURES = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

df = pd.read_parquet(TRAIN_PATH)

print("\n=== Negative-value inspection: benign training data ===")

for feature in FEATURES:
    values = pd.to_numeric(df[feature], errors="coerce")
    negative = values[values < 0]

    print(f"\nFeature: {feature}")
    print(f"Total rows: {len(values):,}")
    print(f"Negative rows: {len(negative):,}")
    print(f"Negative percentage: {len(negative) / len(values) * 100:.5f}%")

    if len(negative):
        print("Most frequent negative values:")
        print(negative.value_counts().head(10).to_string())
        print("Negative-value quantiles:")
        print(negative.quantile([0, 0.25, 0.50, 0.75, 0.99, 1]).to_string())

    print(f"Positive infinity: {np.isposinf(values).sum():,}")
    print(f"Negative infinity: {np.isneginf(values).sum():,}")
