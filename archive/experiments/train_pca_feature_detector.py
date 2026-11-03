
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

RANDOM_STATE = 42
TARGET_FPR = 0.01
MAX_TRAIN_ROWS = 100_000
MAX_CALIBRATION_ROWS = 50_000
BATCH_SIZE = 20_000

# Exploratory candidate features.
# They were identified through earlier test-set analysis, so
# this experiment is NOT an independent final evaluation.
CANDIDATE_FEATURES = [
    "Protocol",
    "PSH Flag Count",
    "Init_Win_bytes_backward",
    "Packet Length Std",
]


def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_parquet(path)


def select_numeric_features(frame):
    excluded = {"Timestamp", "Label"}
    return [
        col for col in frame.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(frame[col])
    ]


def score_batches(pca, scaler, imputer, frame, features):
    overall_scores = []
    feature_scores = []

    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE][features]
        values = scaler.transform(imputer.transform(batch))

        reconstructed = pca.inverse_transform(pca.transform(values))
        squared_errors = np.square(values - reconstructed)

        # Original PCA score: mean reconstruction error over all features.
        overall_scores.append(squared_errors.mean(axis=1))

        # Per-feature errors, retained only for selected features.
        indices = [features.index(f) for f in CANDIDATE_FEATURES]
        feature_scores.append(squared_errors[:, indices])

    return np.concatenate(overall_scores), np.concatenate(feature_scores, axis=0)


def main():
    started = time.time()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train = load_frame(DATA / "train_benign.parquet")
    calibration = load_frame(DATA / "calibration_benign.parquet")
    test = load_frame(DATA / "test_tue_fri.parquet")

    features = select_numeric_features(train)
    missing = [f for f in CANDIDATE_FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing candidate features: {missing}")

    train_sample = train.sample(
        n=min(MAX_TRAIN_ROWS, len(train)),
        random_state=RANDOM_STATE,
    )
    calibration_sample = calibration.sample(
        n=min(MAX_CALIBRATION_ROWS, len(calibration)),
        random_state=RANDOM_STATE,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train = scaler.fit_transform(imputer.fit_transform(train_sample[features]))

    pca = PCA(
        n_components=0.95,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )
    pca.fit(x_train)

    print(f"PCA components: {pca.n_components_}")
    print(f"Candidate features: {CANDIDATE_FEATURES}")

    calibration_overall, calibration_feature = score_batches(
        pca, scaler, imputer, calibration_sample, features
    )
    test_overall, test_feature = score_batches(
        pca, scaler, imputer, test, features
    )

    # Calibrate each score component using benign calibration data only.
    overall_scale = max(
        float(np.quantile(calibration_overall, 0.99)), 1e-12
    )
    feature_scales = np.maximum(
        np.quantile(calibration_feature, 0.99, axis=0),
        1e-12,
    )

    # Convert each error into a ratio of its benign 99th-percentile scale.
    # Combine the overall PCA error with the largest selected-feature error.
    calibration_combined = np.maximum(
        calibration_overall / overall_scale,
        (calibration_feature / feature_scales).max(axis=1),
    )
    test_combined = np.maximum(
        test_overall / overall_scale,
        (test_feature / feature_scales).max(axis=1),
    )

    threshold = float(np.quantile(calibration_combined, 1 - TARGET_FPR))
    y_true = (
        test["Label"].astype(str).str.upper() != "BENIGN"
    ).astype(int).to_numpy()
    y_pred = (test_combined > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "PCA plus selected-feature reconstruction errors",
        "candidate_features": CANDIDATE_FEATURES,
        "feature_count": len(features),
        "pca_components": int(pca.n_components_),
        "training_rows_used": int(len(train_sample)),
        "calibration_rows_used": int(len(calibration_sample)),
        "test_rows": int(len(test)),
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
            average_precision_score(y_true, test_combined)
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "note": (
            "Exploratory only. Candidate features were motivated by prior "
            "test-set exploration; independent validation is required."
        ),
    }

    pd.DataFrame({
        "Timestamp": test["Timestamp"].to_numpy(),
        "Label": test["Label"].to_numpy(),
        "anomaly_score": test_combined,
        "pca_reconstruction_error": test_overall,
        "max_selected_feature_error_ratio": (
            test_feature / feature_scales
        ).max(axis=1),
        "predicted_anomaly": y_pred,
    }).to_parquet(
        OUTPUTS / "pca_feature_detector_predictions.parquet",
        index=False,
    )

    (OUTPUTS / "pca_feature_detector_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\n=== PCA Feature Detector Results ===")
    print(f"Threshold:          {threshold:.6f}")
    print(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
    print(f"Detection rate:     {metrics['detection_rate']:.2%}")
    print(f"False-positive rate:{metrics['false_positive_rate']:.2%}")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Average precision:  {metrics['average_precision']:.4f}")
    print(f"Runtime:            {metrics['elapsed_seconds']} seconds")
    print(f"Metrics: {OUTPUTS / 'pca_feature_detector_metrics.json'}")


if __name__ == "__main__":
    main()
