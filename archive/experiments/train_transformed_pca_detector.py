
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "outputs/experiment/train_benign.parquet"
CAL_PATH = ROOT / "outputs/experiment/calibration_benign.parquet"
TEST_PATH = ROOT / "outputs/experiment/test_tue_fri.parquet"

OUT_DIR = ROOT / "outputs/evaluation"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 100_000
CAL_SAMPLE_SIZE = 50_000
BATCH_SIZE = 50_000
TARGET_FPR = 0.01

LOG_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Fwd Packets/s",
    "Bwd Packets/s",
]

INVALID_NEGATIVE_FEATURES = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
]

def prepare_features(frame, features):
    """Apply fixed, predefined feature handling and transformations."""
    X = frame[features].copy()

    # These three features should be nonnegative. Treat negative sentinel
    # observations as missing; do not treat them as real negative rates.
    for feature in INVALID_NEGATIVE_FEATURES:
        X.loc[X[feature] < 0, feature] = np.nan

    # Log transform only nonnegative, skewed features.
    for feature in LOG_FEATURES:
        X[feature] = np.log1p(X[feature])

    return X.replace([np.inf, -np.inf], np.nan)


def score_batches(frame, features, imputer, scaler, pca):
    errors = []

    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE]
        X = prepare_features(batch, features)
        X = imputer.transform(X)
        X = scaler.transform(X)

        reconstructed = pca.inverse_transform(pca.transform(X))
        batch_error = np.mean((X - reconstructed) ** 2, axis=1)
        errors.append(batch_error)

    return np.concatenate(errors)


def main():
    start_time = time.time()

    print("Loading benign training data...")
    train = pd.read_parquet(TRAIN_PATH)

    # CIC-IDS2017's Label column is not used as a model feature.
    features = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c != "Label"
    ]

    missing = [c for c in features if c not in train.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    train_sample = train.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(train)),
        random_state=RANDOM_STATE
    )

    X_train = prepare_features(train_sample, features)

    print("Fitting imputer and scaler on benign training sample...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    print("Fitting PCA...")
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)

    print(f"PCA components: {pca.n_components_}")
    print(
        "Explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4%}"
    )

    print("Loading benign calibration data...")
    calibration = pd.read_parquet(CAL_PATH)
    calibration_sample = calibration.sample(
        n=min(CAL_SAMPLE_SIZE, len(calibration)),
        random_state=RANDOM_STATE
    )

    calibration_scores = score_batches(
        calibration_sample, features, imputer, scaler, pca
    )

    threshold = float(np.quantile(calibration_scores, 1 - TARGET_FPR))
    print(f"Threshold at {TARGET_FPR:.1%} calibration FPR: {threshold:.8f}")

    print("Loading test data...")
    test = pd.read_parquet(TEST_PATH)

    test_scores = score_batches(test, features, imputer, scaler, pca)
    predictions = test_scores > threshold

    # Labels are used only for exploratory evaluation, never for fitting,
    # feature selection, transformation selection, or threshold calibration.
    labels = test["Label"].astype(str).str.upper().ne("BENIGN").to_numpy()

    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))

    detection_rate = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    ap = average_precision_score(labels, test_scores)

    metrics = {
        "model": "Transformed PCA",
        "threshold": threshold,
        "target_calibration_fpr": TARGET_FPR,
        "pca_components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detection_rate": detection_rate,
        "false_positive_rate": fpr,
        "precision": precision,
        "average_precision": float(ap),
        "runtime_seconds": time.time() - start_time,
        "note": "Exploratory evaluation; test labels were previously inspected."
    }

    print("\n=== Transformed PCA Results ===")
    print(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
    print(f"Detection rate: {detection_rate:.2%}")
    print(f"False-positive rate: {fpr:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Average precision: {ap:.4f}")
    print(f"Runtime: {metrics['runtime_seconds']:.1f} seconds")

    with open(OUT_DIR / "transformed_pca_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "anomaly_score": test_scores,
        "is_anomaly": predictions
    }).to_parquet(
        OUT_DIR / "transformed_pca_predictions.parquet",
        index=False
    )

    joblib.dump(imputer, MODEL_DIR / "transformed_pca_imputer.joblib")
    joblib.dump(scaler, MODEL_DIR / "transformed_pca_scaler.joblib")
    joblib.dump(pca, MODEL_DIR / "transformed_pca_model.joblib")
    joblib.dump(features, MODEL_DIR / "transformed_pca_feature_columns.joblib")

    with open(MODEL_DIR / "transformed_pca_threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    print(f"\nSaved metrics to: {OUT_DIR / 'transformed_pca_metrics.json'}")


if __name__ == "__main__":
    main()
