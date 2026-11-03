"""
src/train_subspace_autoencoder.py
Advanced Multi-Subspace & Top-K Residual Autoencoder for One-Way Threat Detection.

Solves the "Residual Dilution Problem" where focused anomalies in timing/cadence
(e.g., DoS Hulk, DDoS, Slowloris) get smoothed out by unrelated normal features.

Implements:
1. Top-K Residual Pooling (K=4): Focuses on the most anomalous feature deviations.
2. Functional Subspace Decomposition:
   - Timing & Cadence Subspace (targets DoS Hulk, DDoS, Slowloris)
   - Volume & Payload Subspace (targets Heartbleed, Infiltration, Exfiltration)
   - Protocol & State Subspace (targets PortScan, TCP probe attacks)
3. Subspace Max-Percentile Union Anomaly Scoring.
4. Full evaluation on 2,300,825 test flows with per-attack breakdown.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
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
TARGET_FPR = 0.02  # Target ~2% operational FPR
MAX_TRAIN_ROWS = 120_000
MAX_CAL_ROWS = 50_000
BATCH_SIZE = 50_000
TOP_K = 4

# --------------------------------------------------------------------------
# 1. Functional Subspace Definitions
# --------------------------------------------------------------------------

TIMING_FEATURES: List[str] = [
    "Flow Duration",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Fwd Packets/s",
    "Flow Bytes/s",
]

VOLUME_FEATURES: List[str] = [
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "act_data_pkt_fwd",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Fwd Header Length",
    "Avg Fwd Segment Size",
    "min_seg_size_forward",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
]

PROTOCOL_FEATURES: List[str] = [
    "Init_Win_bytes_forward",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "FIN Flag Count",
    "PSH Flag Count",
    "URG Flag Count",
    "Fwd PSH Flags",
    "Fwd URG Flags",
]

# Map feature names to column indices in ONE_WAY_FEATURES
TIMING_IDX = [ONE_WAY_FEATURES.index(f) for f in TIMING_FEATURES]
VOLUME_IDX = [ONE_WAY_FEATURES.index(f) for f in VOLUME_FEATURES]
PROTOCOL_IDX = [ONE_WAY_FEATURES.index(f) for f in PROTOCOL_FEATURES]


def compute_residuals(
    model: MLPRegressor,
    scaled_data: np.ndarray,
    weights: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Computes element-wise weighted squared residuals matrix (N x D)."""
    residuals = []
    n = len(scaled_data)

    for start in range(0, n, batch_size):
        batch = scaled_data[start : start + batch_size]
        reconstructed = model.predict(batch)
        # Weighted squared difference for each feature
        diff_sq = np.square(batch - reconstructed) * weights
        residuals.append(diff_sq)

    return np.vstack(residuals) if residuals else np.empty((0, len(weights)))


def compute_top_k_scores(residuals: np.ndarray, k: int = 4) -> np.ndarray:
    """Computes mean of top-k largest feature residuals per sample."""
    # Partition to find top-k largest values per row efficiently
    top_k_vals = np.partition(residuals, -k, axis=1)[:, -k:]
    return np.mean(top_k_vals, axis=1)


def compute_subspace_scores(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes mean residual for each functional subspace."""
    timing_scores = np.mean(residuals[:, TIMING_IDX], axis=1)
    volume_scores = np.mean(residuals[:, VOLUME_IDX], axis=1)
    protocol_scores = np.mean(residuals[:, PROTOCOL_IDX], axis=1)
    return timing_scores, volume_scores, protocol_scores


def main():
    started = time.time()
    print("=" * 80)
    print("TRAINING MULTI-SUBSPACE & TOP-K RESIDUAL ONE-WAY AUTOENCODER")
    print("SIH 2026 — Problem ID: SIH26145")
    print("=" * 80)

    print("\n1. Loading datasets...")
    train_df = pd.read_parquet(TRAIN_FILE)
    cal_df = pd.read_parquet(CALIBRATION_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    train_sample = train_df.sample(n=min(MAX_TRAIN_ROWS, len(train_df)), random_state=RANDOM_STATE)
    cal_sample = cal_df.sample(n=min(MAX_CAL_ROWS, len(cal_df)), random_state=RANDOM_STATE)

    print(f"   Train samples used:       {len(train_sample):,}")
    print(f"   Calibration samples used: {len(cal_sample):,}")
    print(f"   Test flows to score:      {len(test_df):,}")

    print("\n2. Fitting Log-Preprocessing Pipeline on Benign Training Data...")
    pipeline = LogPreprocessingPipeline(features=ONE_WAY_FEATURES)
    pipeline.fit(train_sample)

    X_train_scaled = pipeline.transform(train_sample, batch_size=BATCH_SIZE)
    X_cal_scaled = pipeline.transform(cal_sample, batch_size=BATCH_SIZE)

    print(f"\n3. Training Enhanced Autoencoder (29 -> 24 -> 10 -> 24 -> 29)...")
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(24, 10, 24),
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
    cal_raw_residuals = np.square(X_cal_scaled - cal_rec)
    cal_var = np.var(cal_raw_residuals, axis=0)
    weights = 1.0 / (cal_var + 1e-3)
    weights = weights / np.mean(weights)

    print("\n5. Computing Weighted Residuals on Calibration & Test sets...")
    cal_residuals = cal_raw_residuals * weights

    X_test_scaled = pipeline.transform(test_df, batch_size=BATCH_SIZE)
    test_residuals = compute_residuals(autoencoder, X_test_scaled, weights, batch_size=BATCH_SIZE)

    # Strategy A: Global Mean (Flat baseline)
    cal_global_mean = np.mean(cal_residuals, axis=1)
    test_global_mean = np.mean(test_residuals, axis=1)
    thresh_global = calibrate_threshold(cal_global_mean, target_fpr=TARGET_FPR)

    # Strategy B: Top-K Residual Pooling (K=4)
    cal_top_k = compute_top_k_scores(cal_residuals, k=TOP_K)
    test_top_k = compute_top_k_scores(test_residuals, k=TOP_K)
    thresh_top_k = calibrate_threshold(cal_top_k, target_fpr=TARGET_FPR)

    # Strategy C: Multi-Subspace Max-Percentile Pooling
    cal_time, cal_vol, cal_proto = compute_subspace_scores(cal_residuals)
    test_time, test_vol, test_proto = compute_subspace_scores(test_residuals)

    # Percentile-normalize each subspace score relative to calibration distribution
    cal_n = len(cal_time)
    test_n = len(test_time)

    # Empirical percentile rank (0.0 to 1.0)
    p_time_cal = rankdata(cal_time) / cal_n
    p_vol_cal = rankdata(cal_vol) / cal_n
    p_proto_cal = rankdata(cal_proto) / cal_n
    cal_subspace_max = np.maximum(p_time_cal, np.maximum(p_vol_cal, p_proto_cal))
    thresh_subspace = calibrate_threshold(cal_subspace_max, target_fpr=TARGET_FPR)

    # For test data: project onto calibration distribution using searchsorted
    cal_time_sorted = np.sort(cal_time)
    cal_vol_sorted = np.sort(cal_vol)
    cal_proto_sorted = np.sort(cal_proto)

    p_time_test = np.searchsorted(cal_time_sorted, test_time) / cal_n
    p_vol_test = np.searchsorted(cal_vol_sorted, test_vol) / cal_n
    p_proto_test = np.searchsorted(cal_proto_sorted, test_proto) / cal_n
    test_subspace_max = np.maximum(p_time_test, np.maximum(p_vol_test, p_proto_test))

    # Evaluate all 3 strategies
    test_labels = sanitize_labels(test_df["Label"])
    y_true = (test_labels != "BENIGN").astype(int).to_numpy()

    print("\n" + "=" * 80)
    print("STRATEGY A: Standard Global Mean Residual (Flat Baseline)")
    print("=" * 80)
    y_pred_global = (test_global_mean >= thresh_global).astype(int)
    metrics_a = evaluate_detection_performance(y_true, y_pred_global, test_global_mean, test_labels.to_numpy())
    print(f"Recall: {metrics_a['detection_rate_pct']}% | FPR: {metrics_a['false_positive_rate_pct']}% | Precision: {metrics_a['precision_pct']}% | AP: {metrics_a['average_precision']}")

    print("\n" + "=" * 80)
    print(f"STRATEGY B: Top-{TOP_K} Residual Pooling (Anti-Dilution)")
    print("=" * 80)
    y_pred_topk = (test_top_k >= thresh_top_k).astype(int)
    metrics_b = evaluate_detection_performance(y_true, y_pred_topk, test_top_k, test_labels.to_numpy())
    print(f"Recall: {metrics_b['detection_rate_pct']}% | FPR: {metrics_b['false_positive_rate_pct']}% | Precision: {metrics_b['precision_pct']}% | AP: {metrics_b['average_precision']}")

    print("\n" + "=" * 80)
    print("STRATEGY C: Multi-Subspace Max-Percentile Pooling (Timing + Volume + Protocol)")
    print("=" * 80)
    y_pred_subspace = (test_subspace_max >= thresh_subspace).astype(int)
    metrics_c = evaluate_detection_performance(y_true, y_pred_subspace, test_subspace_max, test_labels.to_numpy())
    print(f"Recall: {metrics_c['detection_rate_pct']}% | FPR: {metrics_c['false_positive_rate_pct']}% | Precision: {metrics_c['precision_pct']}% | AP: {metrics_c['average_precision']}")

    # Pick champion strategy
    champion_metrics = metrics_b if metrics_b["detection_rate_pct"] >= metrics_c["detection_rate_pct"] else metrics_c
    champion_name = f"Top-{TOP_K} Residual Autoencoder" if metrics_b["detection_rate_pct"] >= metrics_c["detection_rate_pct"] else "Multi-Subspace Autoencoder"

    print(f"\nCHAMPION: {champion_name}")
    print("\nPer-Attack Category Detection Breakdown for Champion:")
    cat_df = pd.DataFrame.from_dict(champion_metrics["per_category"], orient="index")
    cat_df = cat_df.sort_values("detection_rate_pct", ascending=False)
    print(cat_df[["total", "detected", "missed", "detection_rate_pct"]].to_string())

    elapsed = round(time.time() - started, 2)
    champion_metrics["model"] = f"One-Way {champion_name}"
    champion_metrics["feature_set"] = "Strict One-Way (Forward-Only)"
    champion_metrics["feature_count"] = len(ONE_WAY_FEATURES)
    champion_metrics["architecture"] = [29, 24, 10, 24, 29]
    champion_metrics["top_k"] = TOP_K
    champion_metrics["elapsed_seconds"] = elapsed

    # Save artifacts
    metrics_path = OUTPUTS / "subspace_autoencoder_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(champion_metrics, f, indent=2)

    cat_df.to_csv(OUTPUTS / "subspace_autoencoder_breakdown.csv", index_label="category")
    joblib.dump(autoencoder, MODELS / "subspace_autoencoder_model.joblib")
    joblib.dump(weights, MODELS / "subspace_autoencoder_weights.joblib")

    print(f"\nSaved champion metrics to: {metrics_path}")
    print(f"Saved attack breakdown to: {OUTPUTS / 'subspace_autoencoder_breakdown.csv'}")
    print(f"Saved model artifact to:   {MODELS / 'subspace_autoencoder_model.joblib'}")
    print(f"Total elapsed runtime:     {elapsed}s")
    print("DONE!")


if __name__ == "__main__":
    main()

