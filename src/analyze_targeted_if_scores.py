
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

pred_path = ROOT / "outputs/evaluation/targeted_if_predictions.parquet"
metrics_path = ROOT / "outputs/evaluation/targeted_if_metrics.json"

pred = pd.read_parquet(pred_path)

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

print("Prediction columns:", pred.columns.tolist())
print("\nMetrics:")
print(json.dumps(metrics, indent=2))

print("\nPrediction preview:")
print(pred.head())

print("\nScore summary:")
# Update this name if the displayed columns use a different name.
score_candidates = [
    c for c in pred.columns
    if "score" in c.lower() or "anomaly" in c.lower()
]
print("Possible score columns:", score_candidates)

for col in score_candidates:
    print(f"\n{col}:")
    print(pred[col].describe(percentiles=[0.50, 0.90, 0.95, 0.99]))

print("\nPrediction value counts:")
for col in pred.columns:
    if "pred" in col.lower() or "anomaly" in col.lower():
        print(col, pred[col].value_counts(dropna=False).to_dict())

print("\nFull output saved to targeted_if_scores_log.txt")