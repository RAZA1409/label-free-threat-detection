
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]

predictions_path = ROOT / "outputs/evaluation/isolation_forest_predictions.parquet"
threshold_path = ROOT / "models/anomaly_threshold.json"
output_path = ROOT / "outputs/evaluation/threshold_analysis.csv"

print("Loading saved evaluation predictions...")
df = pd.read_parquet(predictions_path)

with open(threshold_path, "r", encoding="utf-8") as f:
    threshold_info = json.load(f)

# The saved score is -decision_function, so larger scores
# indicate more anomalous flows.
scores = df["anomaly_score"].to_numpy() if "anomaly_score" in df.columns else df["score"].to_numpy()
y_true = (df["Label"] != "BENIGN").astype(int).to_numpy() if "Label" in df.columns else (df["label"] != "BENIGN").astype(int).to_numpy()

current_threshold = float(threshold_info["threshold"])

# Test thresholds around the current calibration threshold.
multipliers = [0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
thresholds = sorted(set(
    [current_threshold * m for m in multipliers] + [current_threshold]
))

rows = []

for threshold in thresholds:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    rows.append({
        "threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "detection_rate_recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "flagged_flows": int(tp + fp),
    })

results = pd.DataFrame(rows).sort_values("threshold")
results.to_csv(output_path, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

print("\n=== THRESHOLD ANALYSIS ===")
print(f"Current calibrated threshold: {current_threshold:.6f}")
print(results.to_string(
    index=False,
    formatters={
        "threshold": "{:.6f}".format,
        "detection_rate_recall": "{:.2%}".format,
        "false_positive_rate": "{:.2%}".format,
        "precision": "{:.2%}".format,
    }
))
print(f"\nSaved results to: {output_path}")
print("Note: these are test-set comparisons, not a new independent evaluation.")
