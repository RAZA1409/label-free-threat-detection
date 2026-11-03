
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"

predictions_path = EVAL_DIR / "isolation_forest_predictions.parquet"
output_path = EVAL_DIR / "attack_detection_report.csv"

print("Loading saved Isolation Forest predictions...")
df = pd.read_parquet(predictions_path)

required = {"Label", "predicted_anomaly"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Missing required columns: {required - set(df.columns)}"
    )

df["is_attack"] = df["Label"] != "BENIGN"
df["predicted_anomaly"] = df["predicted_anomaly"].astype(bool)

rows = []

for category, group in df[df["is_attack"]].groupby("Label"):
    actual = [1] * len(group)
    predicted = group["predicted_anomaly"].astype(int).tolist()

    tp = sum(predicted)
    fn = len(predicted) - tp

    rows.append({
        "attack_category": category,
        "total_flows": len(group),
        "detected": tp,
        "missed": fn,
        "detection_rate": recall_score(
            actual, predicted, zero_division=0
        ),
        "precision_within_category": precision_score(
            actual, predicted, zero_division=0
        ),
        "f1_score": f1_score(
            actual, predicted, zero_division=0
        ),
    })

report = pd.DataFrame(rows)
report = report.sort_values("detection_rate", ascending=False)
report.to_csv(output_path, index=False)

display_report = report.copy()
for column in [
    "detection_rate",
    "precision_within_category",
    "f1_score",
]:
    display_report[column] = display_report[column].map(
        lambda value: f"{value:.2%}"
    )

print("\n=== PER-ATTACK DETECTION REPORT ===")
print(display_report.to_string(index=False))
print(f"\nSaved report to: {output_path}")
