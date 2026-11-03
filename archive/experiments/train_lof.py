
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
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
MAX_TRAIN_ROWS = 30_000
TARGET_FPR = 0.01
N_NEIGHBORS = 35

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

print(f"Training rows available: {len(train):,}")
print(f"LOF training sample limit: {MAX_TRAIN_ROWS:,}")
print(f"Calibration rows: {len(calibration):,}")
print(f"Test rows: {len(test):,}")
print(f"Features: {len(feature_columns)}")

# Sample training rows reproducibly. LOF is distance-based and can
# become expensive with a very large reference set.
if len(train) > MAX_TRAIN_ROWS:
    train_sample = train.sample(
        n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE
    )
else:
    train_sample = train

def clean_features(frame):
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    return X

print("\nPreparing features...")
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(clean_features(train_sample))
X_cal = imputer.transform(clean_features(calibration))
X_test = imputer.transform(clean_features(test))

# LOF uses distances, so scale each feature using training data only.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cal = scaler.transform(X_cal)
X_test = scaler.transform(X_test)

print("\nFitting Local Outlier Factor...")
start = time.time()

model = LocalOutlierFactor(
    n_neighbors=N_NEIGHBORS,
    contamination="auto",
    novelty=True,
    n_jobs=-1,
)
model.fit(X_train)

print(f"LOF fit completed in {(time.time() - start) / 60:.1f} minutes.")

# Larger values indicate greater abnormality.
cal_scores = -model.score_samples(X_cal)
test_scores = -model.score_samples(X_test)

threshold = float(np.quantile(cal_scores, 1 - TARGET_FPR))
y_true = (test["Label"] != "BENIGN").astype(int).to_numpy()
y_pred = (test_scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "model": "LocalOutlierFactor",
    "training_rows_used": int(len(train_sample)),
    "features": int(len(feature_columns)),
    "n_neighbors": N_NEIGHBORS,
    "target_calibration_fpr": TARGET_FPR,
    "threshold": threshold,
    "test_rows": int(len(test)),
    "benign_test_flows": int((y_true == 0).sum()),
    "attack_test_flows": int((y_true == 1).sum()),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "false_negatives": int(fn),
    "detection_rate": float(recall_score(y_true, y_pred, zero_division=0)),
    "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "pr_auc_average_precision": float(average_precision_score(y_true, test_scores)),
}

joblib.dump(model, MODEL_DIR / "lof_model.joblib")
joblib.dump(imputer, MODEL_DIR / "lof_imputer.joblib")
joblib.dump(scaler, MODEL_DIR / "lof_scaler.joblib")
joblib.dump(feature_columns, MODEL_DIR / "lof_feature_columns.joblib")

with open(MODEL_DIR / "lof_threshold.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

predictions = pd.DataFrame({
    "Timestamp": test["Timestamp"].to_numpy(),
    "Label": test["Label"].to_numpy(),
    "anomaly_score": test_scores,
    "predicted_anomaly": y_pred.astype(bool),
})
predictions.to_parquet(
    EVAL_DIR / "lof_predictions.parquet", index=False
)

with open(EVAL_DIR / "lof_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n=== LOF RESULTS ===")
print(f"Threshold: {threshold:.6f}")
print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"True negatives: {tn:,}")
print(f"False negatives: {fn:,}")
print(f"Attack detection rate: {metrics['detection_rate']:.2%}")
print(f"Test false-positive rate: {metrics['false_positive_rate']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"PR-AUC (average precision): {metrics['pr_auc_average_precision']:.4f}")

print(f"\nSaved LOF model and preprocessing artifacts in: {MODEL_DIR}")
print(f"Saved evaluation results in: {EVAL_DIR}")
