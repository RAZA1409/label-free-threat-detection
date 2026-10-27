
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
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

FEATURES = [
    "PSH Flag Count",
    "Init_Win_bytes_backward",
    "Packet Length Std",
    "Down/Up Ratio",
    "Init_Win_bytes_forward",
]

TARGET_FPR = 0.01
RANDOM_STATE = 42
BATCH_SIZE = 20_000


def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def get_scores(model, imputer, frame):
    scores = []
    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE][FEATURES]
        values = imputer.transform(batch)
        scores.append(-model.decision_function(values))
    return np.concatenate(scores)


def main():
    started = time.time()
    EVAL.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Loading benign training, calibration, and test data...")
    train = load_frame(DATA / "train_benign.parquet")
    calibration = load_frame(DATA / "calibration_benign.parquet")
    test = load_frame(DATA / "test_tue_fri.parquet")

    missing = [f for f in FEATURES if f not in train.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Fit preprocessing on training data only.
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURES])

    print(f"Training rows: {len(train):,}")
    print(f"Features: {FEATURES}")
    print("Fitting targeted Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train)

    print("Calibrating threshold on benign calibration data...")
    calibration_scores = get_scores(model, imputer, calibration)
    threshold = float(np.quantile(calibration_scores, 1 - TARGET_FPR))

    print("Scoring test data...")
    test_scores = get_scores(model, imputer, test)
    y_true = (
        test["Label"].astype(str).str.upper() != "BENIGN"
    ).astype(int).to_numpy()
    y_pred = (test_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "Targeted Isolation Forest",
        "features": FEATURES,
        "training_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "target_calibration_fpr": TARGET_FPR,
        "threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "detection_rate": float(recall_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, test_scores)
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "note": (
            "Exploratory test evaluation. Feature set was motivated by "
            "prior test-set exploration; independent validation is needed."
        ),
    }

    predictions = pd.DataFrame({
        "Timestamp": test["Timestamp"].to_numpy(),
        "Label": test["Label"].to_numpy(),
        "anomaly_score": test_scores,
        "predicted_anomaly": y_pred,
    })
    predictions.to_parquet(
        EVAL / "targeted_if_predictions.parquet", index=False
    )
    (EVAL / "targeted_if_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    joblib.dump(model, MODELS / "targeted_if_model.joblib")
    joblib.dump(imputer, MODELS / "targeted_if_imputer.joblib")
    joblib.dump(FEATURES, MODELS / "targeted_if_features.joblib")
    (MODELS / "targeted_if_threshold.json").write_text(
        json.dumps({
            "threshold": threshold,
            "target_calibration_fpr": TARGET_FPR,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== Targeted Isolation Forest Results ===")
    print(f"Threshold:          {threshold:.8f}")
    print(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
    print(f"Detection rate:     {metrics['detection_rate']:.2%}")
    print(f"False-positive rate:{metrics['false_positive_rate']:.2%}")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Average precision:  {metrics['pr_auc_average_precision']:.4f}")
    print(f"Runtime:            {metrics['elapsed_seconds']} seconds")
    print(f"Metrics:            {EVAL / 'targeted_if_metrics.json'}")


if __name__ == "__main__":
    main()
