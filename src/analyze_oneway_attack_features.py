"""
src/analyze_oneway_attack_features.py
Analyze why forward-only features differ between attacks and benign traffic,
and identify high-signal one-way features for advanced anomaly detection.
"""

from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from features import ONE_WAY_FEATURES, sanitize_labels, clean_feature_frame

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "outputs" / "experiment" / "test_tue_fri.parquet"

print("Loading test data sample...")
test = pd.read_parquet(TEST_FILE)
test["Clean_Label"] = sanitize_labels(test["Label"])

# Select Benign + top attacks
target_labels = ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "DoS GoldenEye", "DoS Slowhttptest"]
subset = test[test["Clean_Label"].isin(target_labels)].copy()

cleaned_features = clean_feature_frame(subset, features=ONE_WAY_FEATURES)
cleaned_features["Label"] = subset["Clean_Label"]

print("\nMedian values of Forward Features across Attack Categories:")
medians = cleaned_features.groupby("Label").median().T
print(medians.to_string())

print("\n90th percentile values of Forward Features across Attack Categories:")
p90 = cleaned_features.groupby("Label").quantile(0.90).T
print(p90.to_string())

OUT_DIR = ROOT / "outputs" / "evaluation"
medians.to_csv(OUT_DIR / "oneway_feature_medians_by_attack.csv")
p90.to_csv(OUT_DIR / "oneway_feature_p90_by_attack.csv")
print(f"\nSaved analysis to {OUT_DIR}/oneway_feature_medians_by_attack.csv")

