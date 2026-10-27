
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "evaluation"

original_path = EVAL / "pca_predictions.parquet"
candidate_path = EVAL / "pca_feature_detector_predictions.parquet"

for path in (original_path, candidate_path):
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

original = pd.read_parquet(original_path)
candidate = pd.read_parquet(candidate_path)

# Verify that rows and labels align before comparing detectors.
if len(original) != len(candidate):
    raise ValueError("Prediction files have different row counts.")

if not original["Label"].astype(str).equals(
    candidate["Label"].astype(str)
):
    raise ValueError("Prediction labels/order do not match.")

original["detector"] = "Original PCA"
candidate["detector"] = "PCA feature detector"

combined = pd.concat(
    [
        original[["Label", "predicted_anomaly", "detector"]],
        candidate[["Label", "predicted_anomaly", "detector"]],
    ],
    ignore_index=True,
)

summary = (
    combined.groupby(["Label", "detector"], sort=False)
    .agg(
        flows=("predicted_anomaly", "size"),
        detected=("predicted_anomaly", "sum"),
    )
    .reset_index()
)

summary["missed"] = summary["flows"] - summary["detected"]
summary["detection_rate_pct"] = (
    100 * summary["detected"] / summary["flows"]
)

comparison = summary.pivot(
    index="Label",
    columns="detector",
    values=["flows", "detected", "missed", "detection_rate_pct"],
)

comparison.columns = [
    f"{metric}_{detector.replace(' ', '_').lower()}"
    for metric, detector in comparison.columns
]
comparison = comparison.reset_index()

output_path = EVAL / "pca_feature_detector_attack_comparison.csv"
comparison.to_csv(output_path, index=False)

print("=== Per-category detector comparison ===")
print(comparison.to_string(index=False))
print(f"\nSaved to: {output_path}")
print(
    "\nNote: exploratory comparison on a previously examined test set; "
    "not independent final validation."
)
