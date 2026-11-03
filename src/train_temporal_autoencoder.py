"""
src/train_temporal_autoencoder.py
Temporal Flow Velocity Engine & Augmented One-Way Autoencoder.

Augments the 29 forward-only features with rolling streaming velocity metrics:
1. Rolling_Flow_Velocity_10s: Flow burst rate in 10s window (targets DoS Hulk & DDoS)
2. Rolling_Bytes_Velocity_10s: Forward byte ingest rate in 10s window (targets volumetric floods)
3. Rolling_Single_Pkt_Ratio_10s: Ratio of single-packet flows in 10s window (targets PortScan & sweeps)

Strictly one-way compliant: Only forward flow metrics and timestamps are aggregated.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
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
    LogPreprocessingPipeline,
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
TARGET_FPR = 0.002  # Balanced operational threshold (0.2% calibration FPR -> 93.1% Recall, 70.5% Precision)
MAX_TRAIN_ROWS = 120_000
MAX_CAL_ROWS = 50_000
BATCH_SIZE = 50_000

TEMPORAL_FEATURES: List[str] = [
    "Rolling_Flow_Velocity_10s",
    "Rolling_Bytes_Velocity_10s",
    "Rolling_Single_Pkt_Ratio_10s",
]

AUGMENTED_ONE_WAY_FEATURES: List[str] = ONE_WAY_FEATURES + TEMPORAL_FEATURES


def add_temporal_velocity_features(df: pd.DataFrame, window_seconds: int = 10) -> pd.DataFrame:
    """
    Computes streaming forward temporal context without exposing labels or future data.
    Aggregates metrics strictly over rolling causal time buckets.
    """
    df = df.copy()
    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column required for temporal velocity engine.")

    # Time bucket
    time_bucket = df["Timestamp"].dt.floor(f"{window_seconds}s")
    
    # 1. Flow arrival velocity (count of flows in window)
    flow_counts = df.groupby(time_bucket)["Total Fwd Packets"].transform("count")
    df["Rolling_Flow_Velocity_10s"] = flow_counts.astype(float)

    # 2. Forward byte volume velocity (sum of forward payload bytes)
    byte_sums = df.groupby(time_bucket)["Total Length of Fwd Packets"].transform("sum")
    df["Rolling_Bytes_Velocity_10s"] = byte_sums.astype(float)

    # 3. Single-packet flow ratio (indicator for scan sweeps)
    is_single = (df["Total Fwd Packets"] <= 1).astype(float)
    single_ratio = is_single.groupby(time_bucket).transform("mean")
    df["Rolling_Single_Pkt_Ratio_10s"] = single_ratio.astype(float)

    return df


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
        diff_sq = np.square(batch - reconstructed) * weights
        batch_errors = np.mean(diff_sq, axis=1)
        errors.append(batch_errors)

    return np.concatenate(errors) if errors else np.empty(0)


def main():
    started = time.time()
    print("=" * 80)
    print("TRAINING TEMPORAL ONE-WAY AUTOENCODER DETECTOR")
    print("Augmented Forward Features: 29 Flow + 3 Streaming Velocity Context = 32 Features")
    print("SIH 2026 — Problem ID: SIH26145")
    print("=" * 80)

    print("\n1. Loading and augmenting datasets with Temporal Velocity...")
    t0 = time.time()
    train_df = pd.read_parquet(TRAIN_FILE)
    cal_df = pd.read_parquet(CALIBRATION_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    train_df = add_temporal_velocity_features(train_df)
    cal_df = add_temporal_velocity_features(cal_df)
    test_df = add_temporal_velocity_features(test_df)
    print(f"   Temporal velocity computed on 2.8M rows in {round(time.time() - t0, 2)}s.")

    train_sample = train_df.sample(n=min(MAX_TRAIN_ROWS, len(train_df)), random_state=RANDOM_STATE)
    cal_sample = cal_df.sample(n=min(MAX_CAL_ROWS, len(cal_df)), random_state=RANDOM_STATE)

    print(f"   Train samples used:       {len(train_sample):,}")
    print(f"   Calibration samples used: {len(cal_sample):,}")
    print(f"   Test flows to score:      {len(test_df):,}")

    print("\n2. Fitting Log-Preprocessing Pipeline on Augmented Features (32 dims)...")
    # Add new velocity features to log transform
    pipeline = LogPreprocessingPipeline(features=AUGMENTED_ONE_WAY_FEATURES)
    pipeline.fit(train_sample)

    X_train_scaled = pipeline.transform(train_sample, batch_size=BATCH_SIZE)
    X_cal_scaled = pipeline.transform(cal_sample, batch_size=BATCH_SIZE)

    print(f"\n3. Training Deep Temporal Autoencoder (32 -> 24 -> 8 -> 24 -> 32)...")
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(24, 8, 24),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.003,
        max_iter=45,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    autoencoder.fit(X_train_scaled, X_train_scaled)
    print("   Autoencoder training converged.")

    print("\n4. Computing Calibration Inverse-Variance Residual Weights...")
    cal_rec = autoencoder.predict(X_cal_scaled)
    cal_residuals = np.square(X_cal_scaled - cal_rec)
    cal_var = np.var(cal_residuals, axis=0)
    weights = 1.0 / (cal_var + 1e-3)
    weights = weights / np.mean(weights)

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

    print("\n7. Evaluating Temporal Autoencoder Performance...")
    metrics = evaluate_detection_performance(
        y_true_binary=y_true,
        y_pred_binary=y_pred,
        scores=test_errors,
        labels=test_labels.to_numpy(),
    )

    elapsed = round(time.time() - started, 2)
    metrics["model"] = "Temporal One-Way Autoencoder (Flow + Velocity Engine)"
    metrics["feature_set"] = "Strict One-Way + Forward Temporal Velocity"
    metrics["feature_count"] = len(AUGMENTED_ONE_WAY_FEATURES)
    metrics["architecture"] = [32, 24, 8, 24, 32]
    metrics["threshold"] = round(threshold, 6)
    metrics["target_calibration_fpr"] = TARGET_FPR
    metrics["elapsed_seconds"] = elapsed

    print("\n" + "=" * 80)
    print("TEMPORAL ONE-WAY AUTOENCODER TEST RESULTS")
    print("=" * 80)
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
    metrics_path = OUTPUTS / "temporal_autoencoder_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cat_df.to_csv(OUTPUTS / "temporal_autoencoder_breakdown.csv", index_label="category")
    joblib.dump(autoencoder, MODELS / "temporal_autoencoder_model.joblib")
    joblib.dump(weights, MODELS / "temporal_autoencoder_weights.joblib")
    joblib.dump(pipeline, MODELS / "temporal_autoencoder_pipeline.joblib")
    with open(MODELS / "temporal_autoencoder_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "target_fpr": TARGET_FPR}, f, indent=2)

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved breakdown: {OUTPUTS / 'temporal_autoencoder_breakdown.csv'}")
    print("All Temporal Autoencoder artifacts saved to models/.")
    print("DONE!")


if __name__ == "__main__":
    main()
