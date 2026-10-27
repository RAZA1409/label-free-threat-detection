
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
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
EVAL = ROOT / "outputs" / "evaluation"
MODELS = ROOT / "models"

RANDOM_STATE = 42
TARGET_FPR = 0.01
MAX_TRAIN_ROWS = 100_000
MAX_CALIBRATION_ROWS = 50_000
BATCH_SIZE = 20_000
N_CLUSTERS = 20

def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)

def distances_in_batches(model, scaler, imputer, frame, features):
    result = []
    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE][features]
        x = scaler.transform(imputer.transform(batch))
        distances = model.transform(x).min(axis=1)
        result.append(distances)
    return np.concatenate(result)

def main():
    started = time.time()
    EVAL.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train = load_frame(DATA / "train_benign.parquet")
    calibration = load_frame(DATA / "calibration_benign.parquet")
    test = load_frame(DATA / "test_tue_fri.parquet")

    excluded = {"Timestamp", "Label"}
    features = [
        c for c in train.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(train[c])
    ]

    train_sample = train.sample(
        n=min(MAX_TRAIN_ROWS, len(train)), random_state=RANDOM_STATE
    )
    calibration_sample = calibration.sample(
        n=min(MAX_CALIBRATION_ROWS, len(calibration)),
        random_state=RANDOM_STATE,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_sample[features]))

    print(f"Training MiniBatchKMeans with {N_CLUSTERS} clusters...")
    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=3,
    )
    model.fit(x_train)

    calibration_scores = distances_in_batches(
        model, scaler, imputer, calibration_sample, features
    )
    threshold = float(np.quantile(calibration_scores, 1 - TARGET_FPR))

    test_scores = distances_in_batches(model, scaler, imputer, test, features)
    y_true = (
        test["Label"].astype(str).str.upper() != "BENIGN"
    ).astype(int).to_numpy()
    y_pred = (test_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "MiniBatchKMeans nearest-centroid distance",
        "n_clusters": N_CLUSTERS,
        "training_rows_used": len(train_sample),
        "calibration_rows_used": len(calibration_sample),
        "test_rows": len(test),
        "target_calibration_fpr": TARGET_FPR,
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "detection_rate": float(recall_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, test_scores)),
        "elapsed_seconds": round(time.time() - started, 2),
        "note": (
            "Exploratory evaluation on a previously examined test split; "
            "independent validation is required."
        ),
    }

    pd.DataFrame({
        "Timestamp": test["Timestamp"].to_numpy(),
        "Label": test["Label"].to_numpy(),
        "anomaly_score": test_scores,
        "predicted_anomaly": y_pred,
    }).to_parquet(EVAL / "kmeans_predictions.parquet", index=False)

    (EVAL / "kmeans_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    joblib.dump(model, MODELS / "kmeans_model.joblib")
    joblib.dump(imputer, MODELS / "kmeans_imputer.joblib")
    joblib.dump(scaler, MODELS / "kmeans_scaler.joblib")
    joblib.dump(features, MODELS / "kmeans_features.joblib")
    (MODELS / "kmeans_threshold.json").write_text(
        json.dumps({"threshold": threshold, "target_calibration_fpr": TARGET_FPR},
                   indent=2),
        encoding="utf-8",
    )

    print("\n=== MiniBatchKMeans Results ===")
    print(f"Threshold:          {threshold:.6f}")
    print(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
    print(f"Detection rate:     {metrics['detection_rate']:.2%}")
    print(f"False-positive rate:{metrics['false_positive_rate']:.2%}")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Average precision:  {metrics['average_precision']:.4f}")
    print(f"Runtime:            {metrics['elapsed_seconds']} seconds")

if __name__ == "__main__":
    main()

