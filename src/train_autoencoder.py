"""
src/train_autoencoder.py
Deep Neural Network Autoencoder for Label-Free Threat Detection in One-Way Network Traffic.

Implements:
1. Non-linear manifold compression via bottleneck MLP (29 -> 20 -> 8 -> 20 -> 29).
2. Log1p preprocessing to linearize heavy-tailed network rates and durations.
3. Feature-weighted reconstruction error (calibrated residual inverse-variance weights).
4. Threshold calibration targeting 1% FPR on unseen Monday benign traffic.
5. Full evaluation on Tuesday-Friday test set (2,300,825 flows).
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    ONE_WAY_FEATURES,
    sanitize_labels,
    calibrate_threshold,
    evaluate_detection_performance,
)
from train_oneway_enhanced_detector import LogPreprocessingPipeline

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
MAX_CAL_ROWS = 50_000
BATCH_SIZE = 50_000


def calculate_weighted_reconstruction_errors(
    model: MLPRegressor,
    scaled_data: np.ndarray,
    weights: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Computes weighted mean squared reconstruction error across batches."""
    errors = []
    n_samples = len(scaled_data)

    for start in range(0, n_samples, batch_size):
        batch = scaled_data[start : start + batch_size]
        reconstructed = model.predict(batch)
        # Element-wise squared residual weighted by feature sensitivity
        diff_sq = np.square(batch - reconstructed) * weights
        batch_errors = np.mean(diff_sq, axis=1)
        errors.append(batch_errors)

    return np.concatenate(errors) if errors else np.empty(0)


def main():
    started = time.time()
    print("=" * 75)
    print("TRAINING DEEP ONE-WAY AUTOENCODER DETECTOR")
    print("SIH 2026 — Problem ID: SIH26145")
    print("=" * 75)

    print("\n1. Loading experiment partitions...")
    train_df = pd.read_parquet(TRAIN_FILE)
    cal_df = pd.read_parquet(CALIBRATION_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    train_sample = train_df.sample(n=min(MAX_TRAIN_ROWS, len(train_df)), random_state=RANDOM_STATE)
    cal_sample = cal_df.sample(n=min(MAX_CAL_ROWS, len(cal_df)), random_state=RANDOM_STATE)

    print(f"   Train samples used:       {len(train_sample):,}")
    print(f"   Calibration samples used: {len(cal_sample):,}")
    print(f"   Test flows to score:      {len(test_df):,}")

    print("\n2. Fitting Log-Preprocessing Pipeline...")
    pipeline = LogPreprocessingPipeline(features=ONE_WAY_FEATURES)
    pipeline.fit(train_sample)

    X_train_scaled = pipeline.transform(train_sample, batch_size=BATCH_SIZE)
    X_cal_scaled = pipeline.transform(cal_sample, batch_size=BATCH_SIZE)

    print(f"   Features in input space: {X_train_scaled.shape[1]}")

    print("\n3. Training Deep Bottleneck Autoencoder (29 -> 20 -> 8 -> 20 -> 29)...")
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(20, 8, 20),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.003,
        max_iter=40,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    # Self-supervised task: predict X from X through the 8-dim latent bottleneck
    autoencoder.fit(X_train_scaled, X_train_scaled)
    print("   Autoencoder training converged.")

    print("\n4. Computing Calibration Feature Residual Weights...")
    cal_rec = autoencoder.predict(X_cal_scaled)
    cal_residuals = np.square(X_cal_scaled - cal_rec)
    cal_var = np.var(cal_residuals, axis=0)
    # Weights inversely proportional to benign residual noise (stabilized with epsilon)
    weights = 1.0 / (cal_var + 1e-3)
    weights = weights / np.mean(weights)  # Normalize mean weight to 1.0

    print("\n5. Calibrating Anomaly Threshold on Benign Calibration Traffic...")
    cal_errors = np.mean(cal_residuals * weights, axis=1)
    threshold = calibrate_threshold(cal_errors, target_fpr=TARGET_FPR)
    print(f"   Target Calibration FPR: {TARGET_FPR:.2%}")
    print(f"   Calibrated Threshold:   {threshold:.6f}")

    print("\n6. Transforming and Scoring Test Flows (2.3M Tuesday-Friday)...")
    X_test_scaled = pipeline.transform(test_df, batch_size=BATCH_SIZE)
    test_errors = calculate_weighted_reconstruction_errors(
        model=autoencoder,
        scaled_data=X_test_scaled,
        weights=weights,
        batch_size=BATCH_SIZE,
    )

    test_labels = sanitize_labels(test_df["Label"])
    y_true = (test_labels != "BENIGN").astype(int).to_numpy()
    y_pred = (test_errors >= threshold).astype(int)

    print("\n7. Evaluating Autoencoder Performance...")
    metrics = evaluate_detection_performance(
        y_true_binary=y_true,
        y_pred_binary=y_pred,
        scores=test_errors,
        labels=test_labels.to_numpy(),
    )

    elapsed = round(time.time() - started, 2)
    metrics["model"] = "Deep One-Way Autoencoder (8-Dim Latent Bottleneck)"
    metrics["feature_set"] = "Strict One-Way (Forward-Only)"
    metrics["feature_count"] = len(ONE_WAY_FEATURES)
    metrics["architecture"] = [29, 20, 8, 20, 29]
    metrics["threshold"] = round(threshold, 6)
    metrics["target_calibration_fpr"] = TARGET_FPR
    metrics["elapsed_seconds"] = elapsed

    print("\n" + "=" * 75)
    print("DEEP ONE-WAY AUTOENCODER TEST RESULTS")
    print("=" * 75)
    print(f"Detection Rate (Recall):  {metrics['detection_rate_pct']}%")
    print(f"False Positive Rate:      {metrics['false_positive_rate_pct']}%")
    print(f"Precision:                {metrics['precision_pct']}%")
    print(f"Average Precision (AP):   {metrics['average_precision']}")
    print(f"True Positives:           {metrics['tp']:,}")
    print(f"False Positives:          {metrics['fp']:,}")
    print(f"Total Runtime:            {elapsed}s")

    print("\nPer-Attack Category Detection Breakdown:")
    cat_df = pd.DataFrame.from_dict(metrics["per_category"], orient="index")
    cat_df = cat_df.sort_values("detection_rate_pct", ascending=False)
    print(cat_df[["total", "detected", "missed", "detection_rate_pct"]].to_string())

    # Save artifacts
    metrics_path = OUTPUTS / "oneway_autoencoder_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    cat_df.to_csv(OUTPUTS / "oneway_autoencoder_breakdown.csv", index_label="category")

    joblib.dump(autoencoder, MODELS / "oneway_autoencoder_model.joblib")
    joblib.dump(weights, MODELS / "oneway_autoencoder_weights.joblib")
    with open(MODELS / "oneway_autoencoder_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "target_fpr": TARGET_FPR}, f, indent=2)

    print("All Autoencoder artifacts saved to models/.")
    print("DONE!")


if __name__ == "__main__":
    main()

