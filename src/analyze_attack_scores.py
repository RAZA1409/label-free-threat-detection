
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"

predictions_path = EVAL_DIR / "isolation_forest_predictions.parquet"
output_path = EVAL_DIR / "attack_score_analysis.csv"

print("Loading saved anomaly scores...")
df = pd.read_parquet(predictions_path)

required = {"Label", "anomaly_score", "predicted_anomaly"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# The model uses larger scores to indicate more anomalous flows.
df["is_attack"] = df["Label"] != "BENIGN"

benign_scores = df.loc[~df["is_attack"], "anomaly_score"].dropna()

if benign_scores.empty:
    raise ValueError("No benign scores found.")

# Compare each attack category with the benign score distribution.
rows = []

for category, group in df[df["is_attack"]].groupby("Label"):
    scores = group["anomaly_score"].dropna()

    rows.append({
        "attack_category": category,
        "flow_count": len(scores),
        "detected_at_current_threshold": int(
            group["predicted_anomaly"].astype(bool).sum()
        ),
        "detection_rate": float(
            group["predicted_anomaly"].astype(bool).mean()
        ),
        "score_median": float(scores.median()),
        "score_p90": float(scores.quantile(0.90)),
        "score_p99": float(scores.quantile(0.99)),
        "fraction_above_benign_p95": float(
            (scores > benign_scores.quantile(0.95)).mean()
        ),
        "fraction_above_benign_p99": float(
            (scores > benign_scores.quantile(0.99)).mean()
        ),
    })

report = pd.DataFrame(rows).sort_values(
    "score_median", ascending=False
)

benign_summary = {
    "flow_count": int(len(benign_scores)),
    "score_median": float(benign_scores.median()),
    "score_p90": float(benign_scores.quantile(0.90)),
    "score_p95": float(benign_scores.quantile(0.95)),
    "score_p99": float(benign_scores.quantile(0.99)),
}

report.to_csv(output_path, index=False)

print("\n=== BENIGN SCORE DISTRIBUTION ===")
for key, value in benign_summary.items():
    if key == "flow_count":
        print(f"{key}: {value:,}")
    else:
        print(f"{key}: {value:.6f}")

print("\n=== ATTACK SCORE ANALYSIS ===")
print(report.to_string(
    index=False,
    formatters={
        "detection_rate": "{:.2%}".format,
        "score_median": "{:.6f}".format,
        "score_p90": "{:.6f}".format,
        "score_p99": "{:.6f}".format,
        "fraction_above_benign_p95": "{:.2%}".format,
        "fraction_above_benign_p99": "{:.2%}".format,
    },
))

print(f"\nSaved report to: {output_path}")
