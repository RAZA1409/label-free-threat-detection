
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

TRAIN_FILE = DATA / "train_benign.parquet"
CAL_FILE = DATA / "calibration_benign.parquet"
TEST_FILE = DATA / "test_tue_fri.parquet"

TARGET_FPR = 0.01
RANDOM_STATE = 42
BATCH_SIZE = 20_000


def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def get_features(train):
    features = [
        col for col in train.columns
        if col not in {"Timestamp", "Label"}
        and pd.api.types.is_numeric_dtype(train[col])
    ]
    if not features:
        raise ValueError("No numeric features found in training data.")
    return features


def score_iforest(frame, features, imputer, model):
    scores = []
    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE][features]
        values = imputer.transform(batch)
        scores.append(-model.decision_function(values))
    return np.concatenate(scores)


def score_pca(frame, features, imputer, scaler, model):
    scores = []
    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE][features]
        values = imputer.transform(batch)
        values = scaler.transform(values)
        reconstructed = model.inverse_transform(model.transform(values))
        errors = np.mean(np.square(values - reconstructed), axis=1)
        scores.append(errors)
    return np.concatenate(scores)


def main():
    started = time.time()

    print("Loading data and saved models...")
    train = load_frame(TRAIN_FILE)
    calibration = load_frame(CAL_FILE)
    test = load_frame(TEST_FILE)

    features = get_features(train)

    if_model = joblib.load(MODELS / "isolation_forest.joblib")
    if_imputer = joblib.load(MODELS / "feature_imputer.joblib")

    pca_model = joblib.load(MODELS / "pca_model.joblib")
    pca_imputer = joblib.load(MODELS / "pca_imputer.joblib")
    pca_scaler = joblib.load(MODELS / "pca_scaler.joblib")
    pca_features = joblib.load(MODELS / "pca_feature_columns.joblib")

    if list(pca_features) != list(features):
        raise ValueError(
            "PCA feature columns differ from the training feature list."
        )

    # Fit score-normalization references on benign training rows only.
    train_sample = train.sample(
        n=min(100_000, len(train)),
        random_state=RANDOM_STATE,
    )

    print("Scoring benign training sample...")
    if_train = score_iforest(
        train_sample, features, if_imputer, if_model
    )
    pca_train = score_pca(
        train_sample, features, pca_imputer, pca_scaler, pca_model
    )

    # Empirical percentile ranks place both score types on a comparable scale.
    if_reference = np.sort(if_train)
    pca_reference = np.sort(pca_train)

    def percentile_scores(values, reference):
        ranks = np.searchsorted(reference, values, side="right")
        return ranks / len(reference)

    print("Scoring benign calibration data...")
    if_cal = score_iforest(calibration, features, if_imputer, if_model)
    pca_cal = score_pca(
        calibration, features, pca_imputer, pca_scaler, pca_model
    )

    combined_cal = (
        percentile_scores(if_cal, if_reference)
        + percentile_scores(pca_cal, pca_reference)
    ) / 2.0

    threshold = float(np.quantile(combined_cal, 1 - TARGET_FPR))
    calibration_fpr = float(np.mean(combined_cal > threshold))

    print("Scoring test data...")
    if_test = score_iforest(test, features, if_imputer, if_model)
    pca_test = score_pca(test, features, pca_imputer, pca_scaler, pca_model)

    combined_test = (
        percentile_scores(if_test, if_reference)
        + percentile_scores(pca_test, pca_reference)
    ) / 2.0

    y_true = (
        test["Label"].astype(str).str.upper() != "BENIGN"
    ).astype(int)
    y_pred = (combined_test > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "Combined Isolation Forest and PCA percentile scores",
        "target_calibration_fpr": TARGET_FPR,
        "calibration_fpr": calibration_fpr,
        "threshold": threshold,
        "training_reference_rows": int(len(train_sample)),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "detection_rate": float(recall_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(
            average_precision_score(y_true, combined_test)
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "note": (
            "Exploratory test evaluation. Threshold calibrated on benign "
            "calibration data; test labels were not used to set threshold."
        ),
    }

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "Timestamp": test["Timestamp"].to_numpy(),
        "Label": test["Label"].to_numpy(),
        "anomaly_score": combined_test,
        "predicted_anomaly": y_pred,
    })
    predictions.to_parquet(
        OUTPUTS / "combined_predictions.parquet", index=False
    )

    (OUTPUTS / "combined_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Store the normalization references and operating threshold.
    joblib.dump(if_reference, MODELS / "combined_if_reference.joblib")
    joblib.dump(pca_reference, MODELS / "combined_pca_reference.joblib")
    (MODELS / "combined_threshold.json").write_text(
        json.dumps({
            "threshold": threshold,
            "target_calibration_fpr": TARGET_FPR,
            "combination": "mean of empirical percentile scores",
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== Combined Detector Results ===")
    print(f"Threshold:          {threshold:.6f}")
    print(f"Calibration FPR:    {calibration_fpr:.2%}")
    print(f"Detection rate:     {metrics['detection_rate']:.2%}")
    print(f"False-positive rate:{metrics['false_positive_rate']:.2%}")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Average precision:  {metrics['average_precision']:.4f}")
    print(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
    print(f"Runtime: {metrics['elapsed_seconds']} seconds")
    print(f"Metrics: {OUTPUTS / 'combined_metrics.json'}")


if __name__ == "__main__":
    main()
