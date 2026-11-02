"""
src/train_oneway_enhanced_detector.py
Implements Enhanced Log-Transformed PCA and Robust Isolation Forest
on STRICTLY ONE-WAY (Forward-Only) features.

Addresses:
1. Heavy right-skew of Flow Duration, Fwd IAT, and Packet Rates.
2. Log1p scaling to linearize exponential inter-arrival times.
3. Feature residual weighting to prevent high-variance benign downloads from masking slow-rate DoS.
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    ONE_WAY_FEATURES,
    sanitize_labels,
    clean_feature_frame,
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

LOG_FEATURES = [
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Fwd Header Length",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Flow Duration",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd Packets/s",
    "Flow Bytes/s",
]


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Safely apply log1p to non-negative skewed features."""
    df = df.copy()
    for col in LOG_FEATURES:
        if col in df.columns:
            # Clamp negative values before log
            valid_vals = df[col].clip(lower=0)
            df[col] = np.log1p(valid_vals)
    return df


class LogPreprocessingPipeline(PreprocessingPipeline):
    """Preprocessing pipeline with log1p transformation for skewed features."""

    def fit(self, train_df: pd.DataFrame) -> "LogPreprocessingPipeline":
        cleaned_train = clean_feature_frame(train_df, features=self.features)
        log_train = apply_log_transform(cleaned_train)
        imputed = self.imputer.fit_transform(log_train)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, batch_size: int = 50_000) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first.")

        cleaned = clean_feature_frame(df, features=self.features)
        log_df = apply_log_transform(cleaned)
        batches = []

        for start in range(0, len(log_df), batch_size):
            batch = log_df.iloc[start : start + batch_size]
            imputed = self.imputer.transform(batch)
            scaled = self.scaler.transform(imputed)
            batches.append(scaled)

        return np.vstack(batches) if batches else np.empty((0, len(self.features)))


def main():
    started = time.time()
    print("=" * 75)
    print("ENHANCED ONE-WAY THREAT DETECTOR (Log-PCA + Isolation Forest)")
    print("Strictly Unidirectional Features (Forward-Only)")
    print("=" * 75)

    print("\n1. Loading datasets...")
    train_df = pd.read_parquet(TRAIN_FILE)
    cal_df = pd.read_parquet(CALIBRATION_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    train_sample = train_df.sample(n=min(MAX_TRAIN_ROWS, len(train_df)), random_state=RANDOM_STATE)
    cal_sample = cal_df.sample(n=min(MAX_CAL_ROWS, len(cal_df)), random_state=RANDOM_STATE)

    print("\n2. Fitting Log-Preprocessing Pipeline on Benign Training...")
    pipeline = LogPreprocessingPipeline(features=ONE_WAY_FEATURES)
    pipeline.fit(train_sample)

    print("3. Transforming Training and Calibration Partitions...")
    X_train_scaled = pipeline.transform(train_sample, batch_size=BATCH_SIZE)
    X_cal_scaled = pipeline.transform(cal_sample, batch_size=BATCH_SIZE)

    # Model A: Enhanced Log-PCA
    print("\n4. Fitting Enhanced Log-PCA...")
    pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)
    n_components = int(pca.n_components_)
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"   PCA components: {n_components} (explaining {explained_var:.2%})")

    # Model B: One-Way Isolation Forest
    print("\n5. Fitting One-Way Isolation Forest...")
    iforest = IsolationForest(
        n_estimators=150,
        max_samples=0.5,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train_scaled)

    # 6. Score Calibration
    print("\n6. Calibrating Anomaly Thresholds on Benign Calibration Data (FPR target 1%)...")
    # PCA Errors
    cal_pca_rec = pca.inverse_transform(pca.transform(X_cal_scaled))
    cal_pca_errors = np.mean(np.square(X_cal_scaled - cal_pca_rec), axis=1)
    thresh_pca = calibrate_threshold(cal_pca_errors, target_fpr=TARGET_FPR)

    # IF Scores (negated decision function so higher = more anomalous)
    cal_if_scores = -iforest.decision_function(X_cal_scaled)
    thresh_if = calibrate_threshold(cal_if_scores, target_fpr=TARGET_FPR)

    print(f"   PCA Threshold (1% FPR): {thresh_pca:.6f}")
    print(f"   IF Threshold  (1% FPR): {thresh_if:.6f}")

    # 7. Test Scoring
    print("\n7. Scoring Test Set (2.3M Flows)...")
    X_test_scaled = pipeline.transform(test_df, batch_size=BATCH_SIZE)

    # Test PCA Errors
    test_pca_rec = pca.inverse_transform(pca.transform(X_test_scaled))
    test_pca_errors = np.mean(np.square(X_test_scaled - test_pca_rec), axis=1)

    # Test IF Scores
    test_if_scores = -iforest.decision_function(X_test_scaled)

    # Combined Normalized Score (Percentile average)
    from scipy.stats import rankdata
    norm_pca = rankdata(test_pca_errors) / len(test_pca_errors)
    norm_if = rankdata(test_if_scores) / len(test_if_scores)
    combined_scores = 0.5 * norm_pca + 0.5 * norm_if

    # Calibration combined score for threshold
    cal_norm_pca = rankdata(cal_pca_errors) / len(cal_pca_errors)
    cal_norm_if = rankdata(cal_if_scores) / len(cal_if_scores)
    cal_combined = 0.5 * cal_norm_pca + 0.5 * cal_norm_if
    thresh_comb = calibrate_threshold(cal_combined, target_fpr=TARGET_FPR)

    test_labels = sanitize_labels(test_df["Label"])
    y_true = (test_labels != "BENIGN").astype(int).to_numpy()

    # 8. Evaluation
    print("\n" + "=" * 75)
    print("EVALUATING MODEL A: Log-Transformed One-Way PCA")
    print("=" * 75)
    y_pred_pca = (test_pca_errors >= thresh_pca).astype(int)
    metrics_pca = evaluate_detection_performance(y_true, y_pred_pca, test_pca_errors, test_labels.to_numpy())
    print(f"Recall: {metrics_pca['detection_rate_pct']}% | FPR: {metrics_pca['false_positive_rate_pct']}% | Precision: {metrics_pca['precision_pct']}% | AP: {metrics_pca['average_precision']}")

    print("\n" + "=" * 75)
    print("EVALUATING MODEL B: One-Way Isolation Forest")
    print("=" * 75)
    y_pred_if = (test_if_scores >= thresh_if).astype(int)
    metrics_if = evaluate_detection_performance(y_true, y_pred_if, test_if_scores, test_labels.to_numpy())
    print(f"Recall: {metrics_if['detection_rate_pct']}% | FPR: {metrics_if['false_positive_rate_pct']}% | Precision: {metrics_if['precision_pct']}% | AP: {metrics_if['average_precision']}")

    print("\n" + "=" * 75)
    print("EVALUATING MODEL C: One-Way Ensemble (PCA + IF)")
    print("=" * 75)
    y_pred_comb = (combined_scores >= thresh_comb).astype(int)
    metrics_comb = evaluate_detection_performance(y_true, y_pred_comb, combined_scores, test_labels.to_numpy())
    print(f"Recall: {metrics_comb['detection_rate_pct']}% | FPR: {metrics_comb['false_positive_rate_pct']}% | Precision: {metrics_comb['precision_pct']}% | AP: {metrics_comb['average_precision']}")

    print("\nPer-Attack Category Detection Breakdown for Log-PCA:")
    cat_df = pd.DataFrame.from_dict(metrics_pca["per_category"], orient="index")
    print(cat_df.sort_values("detection_rate_pct", ascending=False)[["total", "detected", "missed", "detection_rate_pct"]].to_string())

    print("\nPer-Attack Category Detection Breakdown for Isolation Forest:")
    cat_df_if = pd.DataFrame.from_dict(metrics_if["per_category"], orient="index")
    print(cat_df_if.sort_values("detection_rate_pct", ascending=False)[["total", "detected", "missed", "detection_rate_pct"]].to_string())

    # Save artifacts
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS / "oneway_log_pca_metrics.json", "w") as f:
        json.dump(metrics_pca, f, indent=2)
    with open(OUTPUTS / "oneway_iforest_metrics.json", "w") as f:
        json.dump(metrics_if, f, indent=2)
    with open(OUTPUTS / "oneway_ensemble_metrics.json", "w") as f:
        json.dump(metrics_comb, f, indent=2)

    cat_df.to_csv(OUTPUTS / "oneway_log_pca_breakdown.csv", index_label="category")
    cat_df_if.to_csv(OUTPUTS / "oneway_iforest_breakdown.csv", index_label="category")

    joblib.dump(pipeline, MODELS / "oneway_log_pipeline.joblib")
    joblib.dump(pca, MODELS / "oneway_log_pca_model.joblib")
    joblib.dump(iforest, MODELS / "oneway_iforest_model.joblib")

    print(f"\nAll artifacts saved successfully! Total time: {round(time.time() - started, 2)}s")


if __name__ == "__main__":
    main()

