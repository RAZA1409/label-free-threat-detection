
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
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
PCA_TRAIN_ROWS = 100_000
LOF_TRAIN_ROWS = 10_000
CALIBRATION_ROWS = 50_000
BATCH_SIZE = 1_000
LOF_NEIGHBORS = 35


def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_parquet(path)


def numeric_features(frame):
    excluded = {"Timestamp", "Label"}
    return [
        col for col in frame.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(frame[col])
    ]


def batched_scores(frame, features, imputer, scaler, pca, lof):
    pca_scores = []
    lof_scores = []

    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start:start + BATCH_SIZE][features]
        values = scaler.transform(imputer.transform(batch))

        reconstructed = pca.inverse_transform(pca.transform(values))
        reconstruction_error = np.mean(
            np.square(values - reconstructed), axis=1
        )

        # Higher scores mean more anomalous for both detectors.
        local_anomaly_score = -lof.decision_function(values)

        pca_scores.append(reconstruction_error)
        lof_scores.append(local_anomaly_score)

    return np.concatenate(pca_scores), np.concatenate(lof_scores)


def percentile_scores(reference_sorted, values):
    # Map scores to empirical percentiles using benign calibration scores.
    # The mapping is monotonic and does not use attack labels.
    return np.searchsorted(
        reference_sorted, values, side="right"
    ) / len(reference_sorted)


def main():
    started = time.time()
    EVAL.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train = load_frame(DATA / "train_benign.parquet")
    calibration = load_frame(DATA / "calibration_benign.parquet")
    test = load_frame(DATA / "test_tue_fri.parquet")

    features = numeric_features(train)
    missing = set(features) - set(calibration.columns)
    missing |= set(features) - set(test.columns)
    if missing:
        raise ValueError(f"Missing features in data splits: {sorted(missing)}")

    pca_train = train.sample(
        n=min(PCA_TRAIN_ROWS, len(train)),
        random_state=RANDOM_STATE,
    )
    lof_train = train.sample(
        n=min(LOF_TRAIN_ROWS, len(train)),
        random_state=RANDOM_STATE + 1,
    )
    calibration_sample = calibration.sample(
        n=min(CALIBRATION_ROWS, len(calibration)),
        random_state=RANDOM_STATE,
    )

    # Fit preprocessing on benign training data only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_pca = scaler.fit_transform(
        imputer.fit_transform(pca_train[features])
    )

    print("Fitting PCA...")
    pca = PCA(
        n_components=0.95,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )
    pca.fit(x_pca)

    # LOF has its own preprocessing fitted only on benign training data.
    lof_imputer = SimpleImputer(strategy="median")
    lof_scaler = StandardScaler()
    x_lof = lof_scaler.fit_transform(
        lof_imputer.fit_transform(lof_train[features])
    )

    print(f"Fitting LOF on {len(lof_train):,} benign flows...")
    lof = LocalOutlierFactor(
        n_neighbors=LOF_NEIGHBORS,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(x_lof)

    # The two detectors use different preprocessing pipelines.
    # Keep them separate when calculating their scores.
    def score_both(frame):
        pca_score, _ = batched_scores(
            frame, features, imputer, scaler, pca,
            # PCA-only placeholder is not used for LOF scoring below.
            lof,
        )
        lof_score_parts = []
        for start in range(0, len(frame), BATCH_SIZE):
            batch = frame.iloc[start:start + BATCH_SIZE][features]
            values = lof_scaler.transform(lof_imputer.transform(batch))
            lof_score_parts.append(-lof.decision_function(values))
        return pca_score, np.concatenate(lof_score_parts)

    print("Scoring benign calibration data...")
    cal_pca, cal_lof = score_both(calibration_sample)

    pca_reference = np.sort(cal_pca)
    lof_reference = np.sort(cal_lof)

    cal_pca_rank = percentile_scores(pca_reference, cal_pca)
    cal_lof_rank = percentile_scores(lof_reference, cal_lof)
    cal_combined = (cal_pca_rank + cal_lof_rank) / 2

    threshold = float(np.quantile(cal_combined, 1 - TARGET_FPR))

    print("Scoring test data...")
    test_pca, test_lof = score_both(test)

    test_combined = (
        percentile_scores(pca_reference, test_pca)
        + percentile_scores(lof_reference, test_lof)
    ) / 2

    y_true = (
        test["Label"].astype(str).str.upper() != "BENIGN"
    ).astype(int).to_numpy()
    y_pred = (test_combined > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "PCA + LOF percentile hybrid",
        "features": len(features),
        "pca_training_rows": len(pca_train),
        "lof_training_rows": len(lof_train),
        "calibration_rows": len(calibration_sample),
        "test_rows": len(test),
        "pca_components": int(pca.n_components_),
        "lof_neighbors": LOF_NEIGHBORS,
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
            "Exploratory test evaluation. Test labels were not used for "
            "training or threshold calibration; independent validation "
            "is required because the test set has been explored previously."
        ),
    }

    pd.DataFrame({
        "Timestamp": test["Timestamp"].to_numpy(),
        "Label": test["Label"].to_numpy(),
        "pca_score": test_pca,
        "lof_score": test_lof,
        "anomaly_score": test_combined,
        "predicted_anomaly": y_pred,
    }).to_parquet(EVAL / "pca_lof_hybrid_predictions.parquet", index=False)

    (EVAL / "pca_lof_hybrid_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    joblib.dump(pca, MODELS / "pca_lof_hybrid_pca.joblib")
    joblib.dump(imputer, MODELS / "pca_lof_hybrid_imputer.joblib")
    joblib.dump(scaler, MODELS / "pca_lof_hybrid_scaler.joblib")
    joblib.dump(lof, MODELS / "pca_lof_hybrid_lof.joblib")
    joblib.dump(lof_imputer, MODELS / "pca_lof_hybrid_lof_imputer.joblib")
    joblib.dump(lof_scaler, MODELS / "pca_lof_hybrid_lof_scaler.joblib")
    joblib.dump(features, MODELS / "pca_lof_hybrid_features.joblib")
    joblib.dump(
        {
            "pca_reference": pca_reference,
            "lof_reference": lof_reference,
            "threshold": threshold,
            "target_calibration_fpr": TARGET_FPR,
        },
        MODELS / "pca_lof_hybrid_calibration.joblib",
    )

    print("\n=== PCA + LOF Hybrid Results ===")
    print(f"Threshold:          {threshold:.6f}")
    print(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
    print(f"Detection rate:     {metrics['detection_rate']:.2%}")
    print(f"False-positive rate:{metrics['false_positive_rate']:.2%}")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Average precision:  {metrics['average_precision']:.4f}")
    print(f"Runtime:            {metrics['elapsed_seconds']} seconds")
    print(f"Metrics: {EVAL / 'pca_lof_hybrid_metrics.json'}")


if __name__ == "__main__":
    main()
