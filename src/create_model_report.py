
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs" / "evaluation"

metric_files = {
    "Isolation Forest": "isolation_forest_metrics.json",
    "Robust Isolation Forest": "isolation_forest_robust_metrics.json",
    "Signed-log Isolation Forest": "isolation_forest_log_metrics.json",
    "Local Outlier Factor": "lof_metrics.json",
    "PCA Reconstruction Error": "pca_metrics.json",
    "Combined IF + PCA": "combined_metrics.json",
}

rows = []

for model_name, filename in metric_files.items():
    path = EVALUATION / filename

    if not path.exists():
        print(f"Skipping missing file: {filename}")
        continue

    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Older experiment scripts use pr_auc_average_precision.
    ap = metrics.get(
        "average_precision",
        metrics.get("pr_auc_average_precision")
    )

    required = [
        "detection_rate",
        "false_positive_rate",
        "precision",
    ]
    missing = [key for key in required if key not in metrics]

    if ap is None:
        missing.append("average_precision/pr_auc_average_precision")

    if missing:
        raise ValueError(
            f"{filename} is missing expected metrics: {missing}"
        )

    rows.append({
        "model": model_name,
        "detection_rate_pct": round(
            metrics["detection_rate"] * 100, 2
        ),
        "false_positive_rate_pct": round(
            metrics["false_positive_rate"] * 100, 2
        ),
        "precision_pct": round(
            metrics["precision"] * 100, 2
        ),
        "average_precision": round(ap, 4),
        "threshold": metrics.get("threshold"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
    })

if not rows:
    raise RuntimeError("No model metric files were found.")

report = pd.DataFrame(rows).sort_values(
    "average_precision", ascending=False
)

output_csv = EVALUATION / "consolidated_model_report.csv"
report.to_csv(output_csv, index=False)

print("\n=== Consolidated Model Comparison ===")
print(report.to_string(index=False))
print(f"\nSaved report: {output_csv}")
