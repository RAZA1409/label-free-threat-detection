
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"

model_files = {
    "Isolation Forest": "isolation_forest_metrics.json",
    "LOF": "lof_metrics.json",
    "Robust Isolation Forest": "isolation_forest_robust_metrics.json",
    "Signed Log Isolation Forest": "isolation_forest_log_metrics.json",
}

rows = []

for model_name, filename in model_files.items():
    path = EVAL_DIR / filename

    if not path.exists():
        print(f"Skipping {model_name}: missing {filename}")
        continue

    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)

    rows.append({
        "Model": model_name,
        "Detection Rate": metrics["detection_rate"],
        "False Positive Rate": metrics["false_positive_rate"],
        "Precision": metrics["precision"],
        "PR-AUC (AP)": metrics["pr_auc_average_precision"],
        "True Positives": metrics["true_positives"],
        "False Positives": metrics["false_positives"],
        "False Negatives": metrics["false_negatives"],
        "Threshold": metrics["threshold"],
    })

if not rows:
    raise FileNotFoundError(
        f"No model metrics JSON files found in {EVAL_DIR}"
    )

results = pd.DataFrame(rows)
results = results.sort_values(
    "PR-AUC (AP)", ascending=False
)

output_path = EVAL_DIR / "model_comparison.csv"
results.to_csv(output_path, index=False)

display = results.copy()

for column in [
    "Detection Rate",
    "False Positive Rate",
    "Precision",
]:
    display[column] = display[column].map(
        lambda value: f"{value:.2%}"
    )

display["PR-AUC (AP)"] = display["PR-AUC (AP)"].map(
    lambda value: f"{value:.4f}"
)

display["Threshold"] = display["Threshold"].map(
    lambda value: f"{value:.6f}"
)

print("\n=== MODEL COMPARISON ===")
print(display.to_string(index=False))
print(f"\nSaved comparison report to: {output_path}")
