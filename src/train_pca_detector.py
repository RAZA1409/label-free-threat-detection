
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "experiment"
OUTPUTS = ROOT / "outputs" / "evaluation"
MODELS = ROOT / "models"

OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA / "train_benign.parquet"
CALIBRATION_FILE = DATA / "calibration_benign.parquet"
TEST_FILE = DATA / "test_tue_fri.parquet"

RANDOM_STATE = 42
TARGET_FPR = 0.01
MAX_TRAIN_ROWS = 100_000
MAX_CALIBRATION_ROWS = 50_000
BATCH_SIZE = 20_000


def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the experiment split script first."
        )
    return pd.read_parquet(path)


def select_numeric_features(frame):
    excluded = {"Timestamp", "Label"}
    columns = [
        col for col in frame.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not columns:
        raise ValueError("No numeric features found.")
    return columns


def reconstruction_errors(model, scaler, imputer, frame, features):
    errors = []

    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE][features]
        values = imputer.transform(batch)
        values = scaler.transform(values)

        reconstructed = model.inverse_transform(model.transform(values))
        batch_errors = np.mean(
            np.square(values - reconstructed), axis=1
        )
        errors.append(batch_errors)

    return np.concatenate(errors)


def main():
    started = time.time()

    print("Loading experiment datasets...")
    train = load_frame(TRAIN_FILE)
    calibration = load_frame(CALIBRATION_FILE)
    test = load_frame(TEST_FILE)

    features = select_numeric_features(train)

    # Keep training and calibration sampling reproducible.
    train_sample = train.sample(
        n=min(MAX_TRAIN_ROWS, len(train)),
        random_state=RANDOM_STATE
    )
    calibration_sample = calibration.sample(
        n=min(MAX_CALIBRATION_ROWS, len(calibration)),
        random_state=RANDOM_STATE
    )

    print(f"Numeric features: {len(features)}")
    print(f"Training rows used: {len(train_sample):,}")
    print(f"Calibration rows used: {len(calibration_sample):,}")
    print(f"Test rows: {len(test):,}")

    # Fit preprocessing on training data only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train = imputer.fit_transform(train_sample[features])
    x_train = scaler.fit_transform(x_train)

    # Retain 95% of training variance.
    print("Fitting PCA; this may take a while...")
    pca = PCA(
        n_components=0.95,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )
    pca.fit(x_train)

    print(f"PCA components retained: {pca.n_components_}")
    print(
        "Explained variance retained: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    # Calibrate only on benign calibration data.
    calibration_errors = reconstruction_errors(
        pca, scaler, imputer, calibration_sample, features
    )
    threshold = float(np.quantile(calibration_errors, 1 - TARGET_FPR))

    print(f"Target calibration FPR: {TARGET_FPR:.2%}")
    print(f"PCA reconstruction threshold: {threshold:.8f}")

    # Evaluate once on the held-out test split.
    print("Scoring test data in batches...")
    test_errors = reconstruction_errors(
        pca, scaler, imputer, test, features
    )
    y_true = (test["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    y_pred = (test_errors > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "PCA reconstruction error",
        "random_state": RANDOM_STATE,
        "training_rows_used": int(len(train_sample)),
        "calibration_rows_used": int(len(calibration_sample)),
        "test_rows": int(len(test)),
        "feature_count": int(len(features)),
        "pca_components": int(pca.n_components_),
        "explained_variance_ratio": float(
            pca.explained_variance_ratio_.sum()
        ),
        "target_calibration_fpr": TARGET_FPR,
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "detection_rate": float(recall_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(
            average_precision_score(y_true, test_errors)
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "note": (
            "Exploratory test evaluation. Test labels must not be used "
            "to select preprocessing, features, or thresholds."
        ),
    }

    prediction_file = OUTPUTS / "pca_predictions.parquet"
    pd.DataFrame({
        "Timestamp": test["Timestamp"].to_numpy(),
        "Label": test["Label"].to_numpy(),
        "anomaly_score": test_errors,
        "predicted_anomaly": y_pred,
    }).to_parquet(prediction_file, index=False)

    metrics_file = OUTPUTS / "pca_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(imputer, MODELS / "pca_imputer.joblib")
    joblib.dump(scaler, MODELS / "pca_scaler.joblib")
    joblib.dump(pca, MODELS / "pca_model.joblib")
    joblib.dump(features, MODELS / "pca_feature_columns.joblib")
    (MODELS / "pca_threshold.json").write_text(
        json.dumps({
            "threshold": threshold,
            "target_calibration_fpr": TARGET_FPR,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== PCA Detector Results ===")
    print(f"Threshold:       {threshold:.8f}")
    print(f"TP:              {tp:,}")
    print(f"FP:              {fp:,}")
    print(f"TN:              {tn:,}")
    print(f"FN:              {fn:,}")
    print(f"Detection rate:  {metrics['detection_rate']:.2%}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.2%}")
    print(f"Precision:       {metrics['precision']:.2%}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Elapsed seconds: {metrics['elapsed_seconds']}")
    print(f"Metrics saved:   {metrics_file}")
    print(f"Predictions saved: {prediction_file}")


if __name__ == "__main__":
    main()
