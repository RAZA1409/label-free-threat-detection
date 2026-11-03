
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "experiment"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "outputs" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_FPR = 0.01
N_ESTIMATORS = 200

print("Loading experiment partitions...")
train = pd.read_parquet(DATA_DIR / "train_benign.parquet")
calibration = pd.read_parquet(DATA_DIR / "calibration_benign.parquet")
test = pd.read_parquet(DATA_DIR / "test_tue_fri.parquet")

feature_columns = [
    c for c in train.columns
    if c not in {"Timestamp", "Label", "CaptureDate"}
    and pd.api.types.is_numeric_dtype(train[c])
]

if not (train["Label"] == "BENIGN").all():
    raise ValueError("Training data must contain only BENIGN flows.")
if not (calibration["Label"] == "BENIGN").all():
    raise ValueError("Calibration data must contain only BENIGN flows.")

def clean_features(frame):
    return frame[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

print(f"Training rows: {len(train):,}")
print(f"Calibration rows: {len(calibration):,}")
print(f"Test rows: {len(test):,}")
print(f"Features: {len(feature_columns)}")

print("\nImputing missing values...")
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(clean_features(train))

print("\nFitting robust scaler on training data...")
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)

print("\nTraining robust-preprocessed Isolation Forest...")
start = time.time()

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples="auto",
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train)

print(f"Training completed in {(time.time() - start) / 60:.2f} minutes.")

# Transform calibration and test data using training-fitted objects.
X_cal = scaler.transform(imputer.transform(clean_features(calibration)))
X_test = scaler.transform(imputer.transform(clean_features(test)))

# Higher scores mean more anomalous.
cal_scores = -model.decision_function(X_cal)
test_scores = -model.decision_function(X_test)

threshold = float(np.quantile(cal_scores, 1 - TARGET_FPR))

y_true = (test["Label"] != "BENIGN").astype(int).to_numpy()
y_pred = (test_scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "model": "IsolationForest_RobustScaler",
    "training_rows": int(len(train)),
    "features": int(len(feature_columns)),
    "n_estimators": N_ESTIMATORS,
    "target_calibration_fpr": TARGET_FPR,
    "threshold": threshold,
    "test_rows": int(len(test)),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "false_negatives": int(fn),
    "detection_rate": float(recall_score(y_true, y_pred, zero_division=0)),
    "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "pr_auc_average_precision": float(
        average_precision_score(y_true, test_scores)
    ),
}

joblib.dump(model, MODEL_DIR / "isolation_forest_robust.joblib")
joblib.dump(imputer, MODEL_DIR / "isolation_forest_robust_imputer.joblib")
joblib.dump(scaler, MODEL_DIR / "isolation_forest_robust_scaler.joblib")
joblib.dump(feature_columns, MODEL_DIR / "isolation_forest_robust_features.joblib")

with open(MODEL_DIR / "isolation_forest_robust_threshold.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

predictions = pd.DataFrame({
    "Timestamp": test["Timestamp"].to_numpy(),
    "Label": test["Label"].to_numpy(),
    "anomaly_score": test_scores,
    "predicted_anomaly": y_pred.astype(bool),
})
predictions.to_parquet(
    EVAL_DIR / "isolation_forest_robust_predictions.parquet",
    index=False,
)

with open(EVAL_DIR / "isolation_forest_robust_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n=== ROBUST ISOLATION FOREST RESULTS ===")
print(f"Threshold: {threshold:.6f}")
print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"True negatives: {tn:,}")
print(f"False negatives: {fn:,}")
print(f"Attack detection rate: {metrics['detection_rate']:.2%}")
print(f"Test false-positive rate: {metrics['false_positive_rate']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"PR-AUC (average precision): {metrics['pr_auc_average_precision']:.4f}")

print(f"\nSaved model artifacts in: {MODEL_DIR}")
print(f"Saved evaluation results in: {EVAL_DIR}")
