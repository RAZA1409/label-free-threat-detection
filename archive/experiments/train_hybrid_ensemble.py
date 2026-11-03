"""
src/train_hybrid_ensemble.py
Dual-Head Hybrid One-Way Threat Detector (Deep Autoencoder + Log-PCA Union).

Combines:
1. Deep Autoencoder (excels at DDoS, DoS Hulk, and non-linear manifold violations)
2. Enhanced Log-PCA (excels at Slowloris, GoldenEye, and linear orthogonal outliers)

Uses calibrated ratio normalization:
    Alert = (Score_AE / Thresh_AE >= 1.0) OR (Score_PCA / Thresh_PCA >= 1.0)
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

CALIBRATION_FILE = DATA / "calibration_benign.parquet"
TEST_FILE = DATA / "test_tue_fri.parquet"
BATCH_SIZE = 50_000


def main():
    started = time.time()
    print("=" * 80)
    print("EVALUATING DUAL-HEAD HYBRID DETECTOR (AUTOENCODER + LOG-PCA)")
    print("=" * 80)

    print("\n1. Loading models and pipeline...")
    pipeline = joblib.load(MODELS / "oneway_log_pipeline.joblib")
    pca = joblib.load(MODELS / "oneway_log_pca_model.joblib")
    ae = joblib.load(MODELS / "oneway_autoencoder_model.joblib")
    ae_weights = joblib.load(MODELS / "oneway_autoencoder_weights.joblib")

    with open(MODELS / "oneway_autoencoder_threshold.json") as f:
        ae_thresh = float(json.load(f)["threshold"])

    print(f"   Autoencoder threshold: {ae_thresh:.6f}")

    print("\n2. Scoring Benign Calibration Set to Calibrate Normalized Fusion...")
    cal_df = pd.read_parquet(CALIBRATION_FILE)
    X_cal = pipeline.transform(cal_df, batch_size=BATCH_SIZE)

    # PCA calibration scores
    cal_pca_rec = pca.inverse_transform(pca.transform(X_cal))
    cal_pca_scores = np.mean(np.square(X_cal - cal_pca_rec), axis=1)
    pca_thresh = calibrate_threshold(cal_pca_scores, target_fpr=0.01)
    print(f"   Log-PCA threshold:     {pca_thresh:.6f}")

    # AE calibration scores
    cal_ae_rec = ae.predict(X_cal)
    cal_ae_scores = np.mean(np.square(X_cal - cal_ae_rec) * ae_weights, axis=1)

    # Normalized calibration ratios
    cal_ratio_pca = cal_pca_scores / pca_thresh
    cal_ratio_ae = cal_ae_scores / ae_thresh
    cal_fused = np.maximum(cal_ratio_pca, cal_ratio_ae)

    # Calibrate combined threshold targeting 3-4% test FPR
    fused_threshold = calibrate_threshold(cal_fused, target_fpr=0.015)
    print(f"   Fused Anomaly Ratio Threshold: {fused_threshold:.4f}")

    print("\n3. Scoring Test Traffic (2.3M Tuesday-Friday flows)...")
    test_df = pd.read_parquet(TEST_FILE)
    X_test = pipeline.transform(test_df, batch_size=BATCH_SIZE)

    # Test PCA scores
    test_pca_rec = pca.inverse_transform(pca.transform(X_test))
    test_pca_scores = np.mean(np.square(X_test - test_pca_rec), axis=1)

    # Test AE scores
    test_ae_rec = ae.predict(X_test)
    test_ae_scores = np.mean(np.square(X_test - test_ae_rec) * ae_weights, axis=1)

    # Fused ratio scores
    test_ratio_pca = test_pca_scores / pca_thresh
    test_ratio_ae = test_ae_scores / ae_thresh
    test_fused = np.maximum(test_ratio_pca, test_ratio_ae)

    test_labels = sanitize_labels(test_df["Label"])
    y_true = (test_labels != "BENIGN").astype(int).to_numpy()
    y_pred = (test_fused >= fused_threshold).astype(int)

    metrics = evaluate_detection_performance(y_true, y_pred, test_fused, test_labels.to_numpy())

    elapsed = round(time.time() - started, 2)
    metrics["model"] = "Dual-Head One-Way Hybrid (AE + Log-PCA)"
    metrics["feature_set"] = "Strict One-Way (Forward-Only)"
    metrics["threshold"] = round(fused_threshold, 4)
    metrics["elapsed_seconds"] = elapsed

    print("\n" + "=" * 80)
    print("DUAL-HEAD HYBRID DETECTOR RESULTS")
    print("=" * 80)
    print(f"Detection Rate (Recall):  {metrics['detection_rate_pct']}%")
    print(f"False Positive Rate:      {metrics['false_positive_rate_pct']}%")
    print(f"Precision:                {metrics['precision_pct']}%")
    print(f"Average Precision (AP):   {metrics['average_precision']}")
    print(f"True Positives:           {metrics['tp']:,}")
    print(f"False Positives:          {metrics['fp']:,}")
    print(f"Runtime:                  {elapsed}s")

    print("\nPer-Attack Category Detection Breakdown:")
    cat_df = pd.DataFrame.from_dict(metrics["per_category"], orient="index")
    cat_df = cat_df.sort_values("detection_rate_pct", ascending=False)
    print(cat_df[["total", "detected", "missed", "detection_rate_pct"]].to_string())

    # Save metrics
    with open(OUTPUTS / "hybrid_ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    cat_df.to_csv(OUTPUTS / "hybrid_ensemble_breakdown.csv", index_label="category")

    print(f"\nSaved metrics: {OUTPUTS / 'hybrid_ensemble_metrics.json'}")
    print(f"Saved breakdown: {OUTPUTS / 'hybrid_ensemble_breakdown.csv'}")
    print("DONE!")


if __name__ == "__main__":
    main()

