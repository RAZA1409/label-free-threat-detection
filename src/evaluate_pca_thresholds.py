
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "experiment"
OUTPUTS = ROOT / "outputs" / "evaluation"
MODELS = ROOT / "models"

CALIBRATION_FILE = DATA / "calibration_benign.parquet"
TEST_FILE = DATA / "test_tue_fri.parquet"

RANDOM_STATE = 42
MAX_CALIBRATION_ROWS = 50_000
BATCH_SIZE = 20_000

# Candidate false-positive rates, chosen BEFORE inspecting test results.
CALIBRATION_FPRS = [0.005, 0.01, 0.02, 0.03, 0.05]


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
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("Loading saved PCA model and preprocessing...")
    pca = joblib.load(MODELS / "pca_model.joblib")
    imputer = joblib.load(MODELS / "pca_imputer.joblib")
    scaler = joblib.load(MODELS / "pca_scaler.joblib")
    features = joblib.load(MODELS / "pca_feature_columns.joblib")

    calibration = pd.read_parquet(CALIBRATION_FILE)
    test = pd.read_parquet(TEST_FILE)

    calibration_sample = calibration.sample(
        n=min(MAX_CALIBRATION_ROWS, len(calibration)),
        random_state=RANDOM_STATE,
    )

    print("Scoring benign calibration sample...")
    calibration_scores = reconstruction_errors(
        pca, scaler, imputer, calibration_sample, features
    )

    # Each threshold is based only on benign calibration scores.
    thresholds = {
        f"{fpr:.1%}": float(
            np.quantile(calibration_scores, 1 - fpr)
        )
        for fpr in CALIBRATION_FPRS
    }

    print("Scoring test data...")
    test_scores = reconstruction_errors(
        pca, scaler, imputer, test, features
    )

    # Labels are used only to evaluate the preselected thresholds.
    y_true = (
        test["Label"].astype(str).str.upper() != "BENIGN"
    ).to_numpy(dtype=np.int8)

    results = []

    for target_fpr, threshold in thresholds.items():
        y_pred = (test_scores > threshold).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        attack_detection = tp / (tp + fn) if tp + fn else 0.0
        false_positive_rate = (
            fp / (fp + tn) if fp + tn else 0.0
        )
        precision = tp / (tp + fp) if tp + fp else 0.0

        results.append({
            "target_calibration_fpr": target_fpr,
            "threshold": threshold,
            "test_detection_rate_pct": round(
                attack_detection * 100, 4
            ),
            "test_false_positive_rate_pct": round(
                false_positive_rate * 100, 4
            ),
            "test_precision_pct": round(precision * 100, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    report = pd.DataFrame(results)
    output_file = OUTPUTS / "pca_threshold_sensitivity.csv"
    report.to_csv(output_file, index=False)

    print("\n=== PCA Threshold Sensitivity ===")
    print(report.to_string(index=False))
    print(f"\nSaved: {output_file}")
    print(
        "\nIMPORTANT: Exploratory test results only. "
        "Do not select a threshold based on these test metrics."
    )


if __name__ == "__main__":
    main()
