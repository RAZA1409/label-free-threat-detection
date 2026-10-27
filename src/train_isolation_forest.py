
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "outputs" / "experiment"
AUDIT_FILE = (
    PROJECT_DIR / "outputs" / "preprocessing" / "timestamp_audit.json"
)
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "outputs" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading experiment partitions...")

train = pd.read_parquet(DATA_DIR / "train_benign.parquet")
calibration = pd.read_parquet(DATA_DIR / "calibration_benign.parquet")
test = pd.read_parquet(DATA_DIR / "test_tue_fri.parquet")

train["Label"] = train["Label"].astype(str).str.strip()
calibration["Label"] = calibration["Label"].astype(str).str.strip()
test["Label"] = test["Label"].astype(str).str.strip()

if not train["Label"].eq("BENIGN").all():
    raise ValueError("Training data must contain benign traffic only.")

if not calibration["Label"].eq("BENIGN").all():
    raise ValueError("Calibration data must contain benign traffic only.")

# Exclude metadata and labels from model inputs.
excluded = {"Timestamp", "Label", "CaptureDate"}
feature_columns = [
    col for col in train.columns
    if col not in excluded
    and pd.api.types.is_numeric_dtype(train[col])
]

if not feature_columns:
    raise ValueError("No numeric model features found.")

# Keep feature order identical across every partition.
X_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
X_cal = calibration[feature_columns].replace([np.inf, -np.inf], np.nan)
X_test = test[feature_columns].replace([np.inf, -np.inf], np.nan)

# Fit imputation ONLY on training data.
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_cal = imputer.transform(X_cal)
X_test = imputer.transform(X_test)

print(f"Training rows: {len(X_train):,}")
print(f"Calibration rows: {len(X_cal):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Features: {len(feature_columns)}")

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train)

# Lower decision_function values indicate more anomalous samples.
# Negate them so higher scores mean more anomalous.
cal_scores = -model.decision_function(X_cal)
test_scores = -model.decision_function(X_test)

# Threshold targets approximately 1% anomalies on benign calibration.
target_fpr = 0.01
threshold = float(np.quantile(cal_scores, 1 - target_fpr))

y_true = (test["Label"].to_numpy() != "BENIGN").astype(int)
y_pred = (test_scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

benign_count = int((y_true == 0).sum())
attack_count = int((y_true == 1).sum())

detection_rate = tp / (tp + fn) if (tp + fn) else 0.0
false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
pr_auc = average_precision_score(y_true, test_scores)

print("\n=== ISOLATION FOREST RESULTS ===")
print(f"Calibration target FPR: {target_fpr:.2%}")
print(f"Chosen anomaly threshold: {threshold:.6f}")
print(f"Test rows: {len(test):,}")
print(f"Benign test flows: {benign_count:,}")
print(f"Attack test flows: {attack_count:,}")
print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"True negatives: {tn:,}")
print(f"False negatives: {fn:,}")
print(f"Attack detection rate: {detection_rate:.4f}")
print(f"Test false-positive rate: {false_positive_rate:.4f}")
print(f"Detection precision: {precision:.4f}")
print(f"Binary PR-AUC (average precision): {pr_auc:.4f}")

print("\n=== DETECTION RATE BY ATTACK CATEGORY ===")
attack_results = {}

for label in sorted(test.loc[y_true == 1, "Label"].unique()):
    mask = test["Label"].to_numpy() == label
    detected = int(y_pred[mask].sum())
    total = int(mask.sum())
    rate = detected / total if total else 0.0

    attack_results[label] = {
        "total": total,
        "detected": detected,
        "detection_rate": rate,
    }
    print(f"{label}: {detected:,}/{total:,} ({rate:.2%})")

# Estimate observed capture duration from audited source-file intervals.
# This sums individual capture intervals, avoiding overnight gaps.
capture_hours = None

if AUDIT_FILE.exists():
    with open(AUDIT_FILE, encoding="utf-8") as f:
        audit = json.load(f)

    total_seconds = 0.0
    for item in audit:
        start = pd.to_datetime(item["start"], errors="coerce")
        end = pd.to_datetime(item["end"], errors="coerce")

        if pd.isna(start) or pd.isna(end):
            continue

        # Include Tuesday-Friday files; exclude Monday training day.
        if start.date() > pd.Timestamp("2017-07-03").date():
            total_seconds += (end - start).total_seconds()

    if total_seconds > 0:
        capture_hours = total_seconds / 3600

false_positives_per_hour = None
if capture_hours:
    false_positives_per_hour = fp / capture_hours
    print(f"\nEstimated observed capture hours: {capture_hours:.3f}")
    print(f"False positives per capture hour: {false_positives_per_hour:.3f}")
else:
    print("\nFalse positives per hour: not calculated (duration unavailable).")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    zero_division=0,
))

# Save model artifacts locally.
joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")
joblib.dump(imputer, MODEL_DIR / "feature_imputer.joblib")
joblib.dump(feature_columns, MODEL_DIR / "feature_columns.joblib")

with open(MODEL_DIR / "anomaly_threshold.json", "w", encoding="utf-8") as f:
    json.dump({
        "threshold": threshold,
        "target_calibration_fpr": target_fpr,
        "feature_count": len(feature_columns),
        "training_rows": len(train),
        "calibration_rows": len(calibration),
        "random_state": 42,
    }, f, indent=2)

# Save row-level evaluation output for dashboard development.
predictions = test[["Timestamp", "Label"]].copy()
predictions["anomaly_score"] = test_scores
predictions["predicted_anomaly"] = y_pred.astype(bool)
predictions.to_parquet(
    RESULT_DIR / "isolation_forest_predictions.parquet",
    index=False,
)

results = {
    "model": "IsolationForest",
    "threshold": threshold,
    "target_calibration_fpr": target_fpr,
    "test_rows": len(test),
    "benign_test_flows": benign_count,
    "attack_test_flows": attack_count,
    "true_positives": int(tp),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "false_negatives": int(fn),
    "detection_rate": detection_rate,
    "false_positive_rate": false_positive_rate,
    "precision": precision,
    "pr_auc_average_precision": pr_auc,
    "observed_capture_hours_estimate": capture_hours,
    "false_positives_per_capture_hour_estimate": false_positives_per_hour,
    "attack_category_results": attack_results,
    "feature_count": len(feature_columns),
}

with open(RESULT_DIR / "isolation_forest_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved model artifacts in:", MODEL_DIR)
print("Saved evaluation results in:", RESULT_DIR)
print("\nTraining and evaluation complete.")
