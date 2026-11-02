"""
src/train_oneway_pca_detector.py
Trains and evaluates an unsupervised PCA anomaly detector using STRICTLY ONE-WAY
(forward-only / unidirectional) features, ensuring 100% compliance with SIH26145.
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    ONE_WAY_FEATURES,
    sanitize_labels,
    PreprocessingPipeline,
    calibrate_threshold,
    evaluate_detection_performance,
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
MAX_TRAIN_ROWS = 150_000
MAX_CAL_ROWS = 50_000
BATCH_SIZE = 50_000


def calculate_reconstruction_errors(
    pca_model: PCA,
    scaled_data: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Calculate mean squared reconstruction error in batches."""
    errors = []
    n_samples = len(scaled_data)

    for start in range(0, n_samples, batch_size):
        batch = scaled_data[start : start + batch_size]
        reconstructed = pca_model.inverse_transform(pca_model.transform(batch))
        # Mean squared reconstruction error per row
        row_errors = np.mean(np.square(batch - reconstructed), axis=1)
        errors.append(row_errors)

    return np.concatenate(errors) if errors else np.empty(0)


def main():
    started = time.time()
    print("=" * 70)
    print("TRAINING ONE-WAY (UNIDIRECTIONAL) PCA THREAT DETECTOR")
    print("SIH 2026 — Problem ID: SIH26145")
    print("=" * 70)

    print(f"\n1. Loading datasets...")
    train_df = pd.read_parquet(TRAIN_FILE)
    cal_df = pd.read_parquet(CALIBRATION_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    print(f"   Train rows:       {len(train_df):,}")
    print(f"   Calibration rows: {len(cal_df):,}")
    print(f"   Test rows:        {len(test_df):,}")

    # Subsample training data for speed while retaining statistical variance
    train_sample = train_df.sample(
        n=min(MAX_TRAIN_ROWS, len(train_df)),
        random_state=RANDOM_STATE,
    )
    cal_sample = cal_df.sample(
        n=min(MAX_CAL_ROWS, len(cal_df)),
        random_state=RANDOM_STATE,
    )

    print(f"\n2. Setting up One-Way Feature Preprocessing...")
    print(f"   Unidirectional Features ({len(ONE_WAY_FEATURES)}):")
    for idx, f in enumerate(ONE_WAY_FEATURES, 1):
        print(f"     {idx:02d}. {f}")

    pipeline = PreprocessingPipeline(features=ONE_WAY_FEATURES)
    pipeline.fit(train_sample)

    print("\n3. Transforming Training and Calibration partitions...")
    X_train_scaled = pipeline.transform(train_sample, batch_size=BATCH_SIZE)
    X_cal_scaled = pipeline.transform(cal_sample, batch_size=BATCH_SIZE)

    # Automatically select number of components to explain >= 95% variance
    print("\n4. Fitting PCA on Benign One-Way Traffic...")
    pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)

    explained_var = float(np.sum(pca.explained_variance_ratio_))
    n_components = int(pca.n_components_)
    print(f"   Components selected: {n_components} (explains {explained_var:.2%} variance)")

    print("\n5. Calibrating Anomaly Threshold on Unseen Benign Calibration Data...")
    cal_errors = calculate_reconstruction_errors(pca, X_cal_scaled, batch_size=BATCH_SIZE)
    threshold = calibrate_threshold(cal_errors, target_fpr=TARGET_FPR)
    print(f"   Target Calibration FPR: {TARGET_FPR:.2%}")
    print(f"   Calibrated Threshold:   {threshold:.6f}")

    print("\n6. Transforming and Scoring Test Data (Tuesday–Friday, 2.3M flows)...")
    X_test_scaled = pipeline.transform(test_df, batch_size=BATCH_SIZE)
    test_errors = calculate_reconstruction_errors(pca, X_test_scaled, batch_size=BATCH_SIZE)

    # Binary ground truth & predictions
    test_labels = sanitize_labels(test_df["Label"])
    y_true = (test_labels != "BENIGN").astype(int).to_numpy()
    y_pred = (test_errors >= threshold).astype(int)

    print("\n7. Evaluating Performance...")
    metrics = evaluate_detection_performance(
        y_true_binary=y_true,
        y_pred_binary=y_pred,
        scores=test_errors,
        labels=test_labels.to_numpy(),
    )

    elapsed = round(time.time() - started, 2)
    metrics["model"] = "Unidirectional (One-Way) PCA Detector"
    metrics["feature_set"] = "Strict One-Way (Forward-Only)"
    metrics["feature_count"] = len(ONE_WAY_FEATURES)
    metrics["pca_components"] = n_components
    metrics["explained_variance"] = round(explained_var, 4)
    metrics["threshold"] = round(threshold, 6)
    metrics["target_calibration_fpr"] = TARGET_FPR
    metrics["elapsed_seconds"] = elapsed

    print("\n" + "=" * 70)
    print("ONE-WAY PCA DETECTOR TEST RESULTS")
    print("=" * 70)
    print(f"Detection Rate (Recall):  {metrics['detection_rate_pct']}%")
    print(f"False Positive Rate:      {metrics['false_positive_rate_pct']}%")
    print(f"Precision:                {metrics['precision_pct']}%")
    print(f"Average Precision (AP):   {metrics['average_precision']}")
    print(f"Total True Positives:     {metrics['tp']:,}")
    print(f"Total False Positives:    {metrics['fp']:,}")
    print(f"Runtime:                  {elapsed}s")

    print("\nPer-Attack Category Detection Breakdown:")
    cat_df = pd.DataFrame.from_dict(metrics["per_category"], orient="index")
    cat_df = cat_df.sort_values("detection_rate_pct", ascending=False)
    print(cat_df[["total", "detected", "missed", "detection_rate_pct"]].to_string())

    # Save metrics JSON
    metrics_path = OUTPUTS / "oneway_pca_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Save category comparison CSV
    cat_csv_path = OUTPUTS / "oneway_pca_attack_breakdown.csv"
    cat_df.to_csv(cat_csv_path, index_label="category")
    print(f"Saved category report: {cat_csv_path}")

    # Save model artifacts
    joblib.dump(pca, MODELS / "oneway_pca_model.joblib")
    joblib.dump(pipeline, MODELS / "oneway_pca_pipeline.joblib")
    with open(MODELS / "oneway_pca_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "target_fpr": TARGET_FPR}, f, indent=2)
    print(f"Saved model artifacts to {MODELS}/")

    print("\nDONE!")


if __name__ == "__main__":
    main()

