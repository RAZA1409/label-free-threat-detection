
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs" / "evaluation"

if_predictions = pd.read_parquet(
    EVALUATION / "isolation_forest_predictions.parquet"
)
pca_predictions = pd.read_parquet(
    EVALUATION / "pca_predictions.parquet"
)

# Confirm both models were evaluated on identical, ordered test flows.
for column in ["Timestamp", "Label"]:
    if not if_predictions[column].equals(pca_predictions[column]):
        raise ValueError(
            f"Test data mismatch in {column}. Comparison stopped."
        )

if "predicted_anomaly" not in if_predictions:
    raise ValueError("Isolation Forest predictions column is missing.")

if "predicted_anomaly" not in pca_predictions:
    raise ValueError("PCA predictions column is missing.")

if_predictions = if_predictions.copy()
pca_predictions = pca_predictions.copy()

if_predictions["predicted_anomaly"] = (
    if_predictions["predicted_anomaly"].astype(bool)
)
pca_predictions["predicted_anomaly"] = (
    pca_predictions["predicted_anomaly"].astype(bool)
)

labels = if_predictions["Label"].astype(str)
rows = []

for category in sorted(labels.unique()):
    mask = labels == category
    count = int(mask.sum())

    if_detected = int(
        if_predictions.loc[mask, "predicted_anomaly"].sum()
    )
    pca_detected = int(
        pca_predictions.loc[mask, "predicted_anomaly"].sum()
    )

    rows.append({
        "category": category,
        "flows": count,
        "if_detected": if_detected,
        "if_detection_rate_pct": round(100 * if_detected / count, 3),
        "pca_detected": pca_detected,
        "pca_detection_rate_pct": round(100 * pca_detected / count, 3),
        "pca_minus_if_percentage_points": round(
            100 * (pca_detected - if_detected) / count, 3
        ),
    })

result = pd.DataFrame(rows).sort_values(
    "flows", ascending=False
)

output_file = EVALUATION / "attack_detection_comparison.csv"
result.to_csv(output_file, index=False)

print("\n=== Detection by Traffic Category ===")
print(result.to_string(index=False))
print(f"\nSaved comparison to: {output_file}")
