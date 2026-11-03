
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
FEATURE_LIST_PATH = ROOT / "outputs/evaluation/correlation_pruned_features.csv"

OUT_DIR = ROOT / "outputs/evaluation"
MODEL_DIR = ROOT / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 100_000
CAL_SAMPLE_SIZE = 50_000
BATCH_SIZE = 50_000
TARGET_FPR = 0.01


def prepare_features(frame, features):
    X = frame[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # These negative values are invalid for these rate/duration features
    # in our inspected benign training data.
    for feature in ["Flow Duration", "Flow Bytes/s", "Flow Packets/s"]:
        if feature in X.columns:
            X.loc[X[feature] < 0, feature] = np.nan

    return X


def score_batches(frame, features, imputer, scaler, pca):
    scores = []

    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE]
        X = prepare_features(batch, features)
        X = imputer.transform(X)
        X = scaler.transform(X)

        reconstructed = pca.inverse_transform(pca.transform(X))
        errors = np.mean((X - reconstructed) ** 2, axis=1)
        scores.append(errors)

    return np.concatenate(scores)


def main():
    start_time = time.time()

    print("Loading feature list...")
    feature_df = pd.read_csv(FEATURE_LIST_PATH)

    if "feature" not in feature_df.columns:
        raise ValueError(
            "Feature CSV must contain a column named 'feature'."
        )

    features = feature_df["feature"].dropna().astype(str).tolist()
    features = list(dict.fromkeys(features))

    print(f"Features selected: {len(features)}")

    print("Loading benign training data...")
    train = pd.read_parquet(TRAIN_PATH)

    missing_features = [f for f in features if f not in train.columns]
    if missing_features:
        raise ValueError(f"Features missing from training data: {missing_features}")

    train_sample = train.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(train)),
        random_state=RANDOM_STATE,
    )

    X_train = prepare_features(train_sample, features)

    print("Fitting median imputer on benign training sample...")
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    print("Fitting standard scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    print("Fitting PCA...")
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)

    print(f"PCA components: {pca.n_components_}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4%}")

    print("Loading benign calibration data...")
    calibration = pd.read_parquet(CAL_PATH)
    missing_cal = [f for f in features if f not in calibration.columns]
    if missing_cal:
        raise ValueError(f"Features missing from calibration data: {missing_cal}")

    cal_sample = calibration.sample(
        n=min(CAL_SAMPLE_SIZE, len(calibration)),
        random_state=RANDOM_STATE,
    )

    calibration_scores = score_batches(
        cal_sample, features, imputer, scaler, pca
    )

    threshold = float(np.quantile(calibration_scores, 1 - TARGET_FPR))
    print(f"Threshold at {TARGET_FPR:.1%} calibration FPR: {threshold:.8f}")

    print("Loading test data...")
    test = pd.read_parquet(TEST_PATH)

    missing_test = [f for f in features if f not in test.columns]
    if missing_test:
        raise ValueError(f"Features missing from test data: {missing_test}")

    test_scores = score_batches(test, features, imputer, scaler, pca)
    predictions = test_scores > threshold

    # Labels are used only for exploratory evaluation, never for fitting
    # the detector or calibrating its threshold.
    labels = (
        test["Label"].astype(str).str.strip().str.upper() != "BENIGN"
    ).to_numpy()

    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))

    detection_rate = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    average_precision = average_precision_score(labels, test_scores)

    metrics = {
        "model": "39-feature PCA",
        "feature_count": len(features),
        "features": features,
        "threshold": threshold,
        "target_calibration_fpr": TARGET_FPR,
        "pca_components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "average_precision": float(average_precision),
        "runtime_seconds": time.time() - start_time,
        "note": "Exploratory test evaluation; test labels were previously inspected.",
    }

    print("\n=== 39-Feature PCA Results ===")
    print(f"Features: {len(features)}")
    print(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
    print(f"Detection rate: {detection_rate:.2%}")
    print(f"False-positive rate: {false_positive_rate:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Average precision: {average_precision:.4f}")
    print(f"Runtime: {metrics['runtime_seconds']:.1f} seconds")

    metrics_path = OUT_DIR / "reduced_pca_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "anomaly_score": test_scores,
        "is_anomaly": predictions,
    }).to_parquet(
        OUT_DIR / "reduced_pca_predictions.parquet",
        index=False,
    )

    joblib.dump(imputer, MODEL_DIR / "reduced_pca_imputer.joblib")
    joblib.dump(scaler, MODEL_DIR / "reduced_pca_scaler.joblib")
    joblib.dump(pca, MODEL_DIR / "reduced_pca_model.joblib")
    joblib.dump(features, MODEL_DIR / "reduced_pca_feature_columns.joblib")

    with open(MODEL_DIR / "reduced_pca_threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    print(f"\nSaved metrics: {metrics_path}")
    print("Saved model artifacts with the reduced_pca_ prefix.")


if __name__ == "__main__":
    main()
