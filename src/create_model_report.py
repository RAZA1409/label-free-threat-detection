import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs" / "evaluation"

metric_files = {
    # --- Strict One-Way / Unidirectional Models (SIH26145) ---
    "Deep One-Way Autoencoder": "oneway_autoencoder_metrics.json",
    "Enhanced One-Way Log-PCA": "oneway_log_pca_metrics.json",
    "One-Way Isolation Forest": "oneway_iforest_metrics.json",
    "One-Way Ensemble (PCA+IF)": "oneway_ensemble_metrics.json",
    "Naive One-Way PCA": "oneway_pca_metrics.json",
    # --- Bidirectional Reference Baselines ---
    "Bidirectional 39-Feature PCA": "reduced_pca_metrics.json",
    "Bidirectional PCA Reconstruction": "pca_metrics.json",
    "PCA + Feature Errors": "pca_feature_detector_metrics.json",
    "PCA + LOF Hybrid": "pca_lof_hybrid_metrics.json",
    "Combined IF + PCA": "combined_metrics.json",
    "MiniBatchKMeans": "kmeans_metrics.json",
    "Standard Isolation Forest": "isolation_forest_metrics.json",
    "Robust Isolation Forest": "isolation_forest_robust_metrics.json",
    "Signed-log Isolation Forest": "isolation_forest_log_metrics.json",
    "Targeted Isolation Forest": "targeted_if_metrics.json",
    "Local Outlier Factor": "lof_metrics.json",
}

rows = []

for model_name, filename in metric_files.items():
    path = EVALUATION / filename

    if not path.exists():
        continue

    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Resolve AP
    ap = metrics.get("average_precision", metrics.get("pr_auc_average_precision"))
    if ap is None:
        continue

    # Resolve detection rate (percentage)
    if "detection_rate_pct" in metrics:
        det_rate = float(metrics["detection_rate_pct"])
    elif "detection_rate" in metrics:
        det_rate = round(float(metrics["detection_rate"]) * 100, 2)
    else:
        continue

    # Resolve false positive rate (percentage)
    if "false_positive_rate_pct" in metrics:
        fpr = float(metrics["false_positive_rate_pct"])
    elif "false_positive_rate" in metrics:
        fpr = round(float(metrics["false_positive_rate"]) * 100, 2)
    else:
        continue

    # Resolve precision (percentage)
    if "precision_pct" in metrics:
        prec = float(metrics["precision_pct"])
    elif "precision" in metrics:
        prec = round(float(metrics["precision"]) * 100, 2)
    else:
        continue

    rows.append({
        "model": model_name,
        "detection_rate_pct": det_rate,
        "false_positive_rate_pct": fpr,
        "precision_pct": prec,
        "average_precision": round(float(ap), 4),
        "threshold": metrics.get("threshold"),
        "elapsed_seconds": metrics.get("elapsed_seconds", metrics.get("runtime_seconds")),
    })

if not rows:
    raise RuntimeError("No model metric files were found.")

report = pd.DataFrame(rows).sort_values("average_precision", ascending=False)

output_csv = EVALUATION / "consolidated_model_report.csv"
report.to_csv(output_csv, index=False)

print("\n=== Consolidated Model Comparison ===")
print(report.to_string(index=False))
print(f"\nSaved report: {output_csv}")
