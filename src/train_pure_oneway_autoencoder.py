"""
src/train_pure_oneway_autoencoder.py
Phase 2 Stage 2: Pure One-Way Autoencoder Training & Calibration Pipeline.

Model Architecture:
    26 -> 20 -> 8 -> 20 -> 26 Bottleneck Autoencoder (MLPRegressor)

Features:
    PURE_ONE_WAY_FEATURES (26 dimensions strictly forward-only)
    PureOneWayLogPreprocessingPipeline (median imputation, log1p, scaling)

Training Policy:
    - Monday benign training partition ONLY (train_benign.parquet)
    - 120,000 observations sampled deterministically with random_state=42
    - Chronological validation segment (Timestamp >= 2017-07-03 09:45:00)
    - Strict unsupervised self-reconstruction (X -> X); zero attack labels exposed
    - No preprocessing fit on validation, calibration, or test data

Scoring & Calibration:
    - Unweighted Mean Squared Error (MSE): score(x) = mean((x - x_hat) ** 2)
    - Thresholds calibrated strictly on Monday benign calibration partition (calibration_benign.parquet)
    - Primary alpha = 0.002 (0.2% target FPR), Reference alpha = 0.01 (1.0% target FPR)
    - Quantile method = 'higher', decision rule = (score > threshold)

Artifact Isolation:
    - All assets saved exclusively to models/pure_one_way_v2/
    - Legacy 32-dim champion artifacts in models/ remain completely untouched
"""

import os
import sys
import json
import time
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "outputs" / "experiment"
MODELS_DIR = ROOT_DIR / "models"
V2_ARTIFACTS_DIR = MODELS_DIR / "pure_one_way_v2"

sys.path.insert(0, str(SRC_DIR))
from features import (
    PURE_ONE_WAY_FEATURES,
    PureOneWayLogPreprocessingPipeline,
    add_temporal_velocity_features,
    check_model_compatibility,
)

TRAIN_FILE = DATA_DIR / "train_benign.parquet"
CALIBRATION_FILE = DATA_DIR / "calibration_benign.parquet"

# Training configuration
RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 120_000
VAL_SAMPLE_SIZE = 25_000
CHRONOLOGICAL_VAL_CUTOFF = pd.Timestamp("2017-07-03 09:45:00")
BATCH_SIZE = 256
MAX_ITER = 45
LEARNING_RATE_INIT = 0.003
ALPHA = 0.0001
SCORING_BATCH_SIZE = 50_000

# Calibration configuration
PRIMARY_ALPHA = 0.002   # 0.2% calibration target FPR
REFERENCE_ALPHA = 0.01  # 1.0% calibration target FPR


def compute_unweighted_reconstruction_errors(
    model: MLPRegressor,
    scaled_data: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    Computes unweighted Mean Squared Error (MSE) reconstruction loss:
        score(x) = mean((x - reconstruction(x)) ** 2)
    across the standardized feature dimensions.
    """
    if len(scaled_data) == 0:
        return np.empty(0, dtype=float)

    errors = []
    n_samples = len(scaled_data)

    for start in range(0, n_samples, batch_size):
        batch = scaled_data[start : start + batch_size]
        reconstructed = model.predict(batch)
        diff_sq = np.square(batch - reconstructed)
        batch_mse = np.mean(diff_sq, axis=1)
        errors.append(batch_mse)

    return np.concatenate(errors) if errors else np.empty(0, dtype=float)


def prepare_training_and_validation_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partitions the Monday benign training data into a 120,000-flow training sample
    and a separate chronological validation segment.
    """
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training partition not found at {TRAIN_FILE}")

    print(f"Loading Monday benign training partition from {TRAIN_FILE.name}...")
    full_train_df = pd.read_parquet(TRAIN_FILE)
    full_train_df["Timestamp"] = pd.to_datetime(full_train_df["Timestamp"])

    # Verify 100% benign labels
    if not full_train_df["Label"].eq("BENIGN").all():
        raise ValueError("Training partition contains unexpected non-benign labels!")

    # Augment full Monday training partition with causal temporal velocity features
    print("Computing causal temporal velocity features on Monday training partition...")
    t0 = time.time()
    augmented_train_df = add_temporal_velocity_features(full_train_df, window_seconds=10)
    print(f"Temporal velocity computed in {time.time() - t0:.2f}s.")

    # Time-aware split: training pool before cutoff, validation pool at or after cutoff
    train_pool = augmented_train_df[augmented_train_df["Timestamp"] < CHRONOLOGICAL_VAL_CUTOFF]
    val_pool = augmented_train_df[augmented_train_df["Timestamp"] >= CHRONOLOGICAL_VAL_CUTOFF]

    if len(train_pool) < TRAIN_SAMPLE_SIZE:
        raise ValueError(
            f"Insufficient training observations ({len(train_pool)}) before cutoff to sample {TRAIN_SAMPLE_SIZE}"
        )

    # Deterministic sampling to preserve diversity and guarantee exact reproducibility
    train_sample = train_pool.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE)
    val_sample = val_pool.sample(n=min(VAL_SAMPLE_SIZE, len(val_pool)), random_state=RANDOM_STATE)

    split_meta = {
        "full_train_partition_rows": len(full_train_df),
        "train_pool_rows_before_cutoff": len(train_pool),
        "val_pool_rows_at_or_after_cutoff": len(val_pool),
        "train_sample_size": len(train_sample),
        "val_sample_size": len(val_sample),
        "chronological_cutoff": str(CHRONOLOGICAL_VAL_CUTOFF),
        "train_time_min": str(train_sample["Timestamp"].min()),
        "train_time_max": str(train_sample["Timestamp"].max()),
        "val_time_min": str(val_sample["Timestamp"].min()),
        "val_time_max": str(val_sample["Timestamp"].max()),
        "temporal_overlap_rows": len(set(train_sample.index) & set(val_sample.index)),
    }

    print(f"   Train sample: {len(train_sample):,} flows [{split_meta['train_time_min']} -> {split_meta['train_time_max']}]")
    print(f"   Validation:   {len(val_sample):,} flows [{split_meta['val_time_min']} -> {split_meta['val_time_max']}]")

    return train_sample, val_sample, split_meta


def train_autoencoder(
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
) -> Tuple[MLPRegressor, Dict[str, Any]]:
    """
    Trains the 26 -> 20 -> 8 -> 20 -> 26 MLPRegressor autoencoder.
    Tracks iterations, convergence warnings, and reconstruction losses.
    """
    print("\nInitializing MLPRegressor (26 -> 20 -> 8 -> 20 -> 26)...")
    model = MLPRegressor(
        hidden_layer_sizes=(20, 8, 20),
        activation="relu",
        solver="adam",
        learning_rate_init=LEARNING_RATE_INIT,
        learning_rate="adaptive",
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        early_stopping=False,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    print(f"Training on {len(X_train_scaled):,} observations (max_iter={MAX_ITER}, batch_size={BATCH_SIZE})...")
    t0 = time.time()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_scaled, X_train_scaled)

    training_duration = time.time() - t0
    print(f"Training finished in {training_duration:.2f}s across {model.n_iter_} iterations.")

    # Check convergence status accurately
    has_conv_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught_warnings)
    reached_max = model.n_iter_ >= model.max_iter
    converged = (not has_conv_warning) and (not reached_max)

    # Compute training and validation reconstruction MSE losses
    print("Computing reconstruction loss diagnostics...")
    train_errors = compute_unweighted_reconstruction_errors(model, X_train_scaled, batch_size=SCORING_BATCH_SIZE)
    val_errors = compute_unweighted_reconstruction_errors(model, X_val_scaled, batch_size=SCORING_BATCH_SIZE)

    train_loss = float(np.mean(train_errors))
    val_loss = float(np.mean(val_errors))

    print(f"   Train Reconstruction MSE:      {train_loss:.6f}")
    print(f"   Validation Reconstruction MSE: {val_loss:.6f}")
    print(f"   Convergence Status:            {'CONVERGED' if converged else 'STOPPED_AT_MAX_ITER'}")

    convergence_meta = {
        "converged": bool(converged),
        "actual_iterations": int(model.n_iter_),
        "max_iter": int(model.max_iter),
        "reached_max_iter": bool(reached_max),
        "convergence_warning_raised": bool(has_conv_warning),
        "convergence_status": (
            "Converged within loss tolerance before max_iter"
            if converged
            else "Reached max_iter limit (45 iterations)"
        ),
        "training_duration_seconds": round(training_duration, 2),
        "final_loss_objective": float(model.loss_),
        "train_reconstruction_mse": round(train_loss, 6),
        "val_reconstruction_mse": round(val_loss, 6),
    }

    return model, convergence_meta


def calibrate_thresholds(
    model: MLPRegressor,
    pipeline: PureOneWayLogPreprocessingPipeline,
    historical_context_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calibrates anomaly thresholds on the Monday benign calibration partition.
    Ensures calibration data is NEVER used to fit preprocessing or model parameters.
    """
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(f"Calibration partition not found at {CALIBRATION_FILE}")

    print(f"\nLoading Monday benign calibration partition from {CALIBRATION_FILE.name}...")
    cal_df = pd.read_parquet(CALIBRATION_FILE)
    cal_df["Timestamp"] = pd.to_datetime(cal_df["Timestamp"])

    if not cal_df["Label"].eq("BENIGN").all():
        raise ValueError("Calibration partition contains unexpected non-benign labels!")

    # Augment calibration with causal temporal velocity, supplying historical training tail for continuity
    print(f"Augmenting {len(cal_df):,} calibration flows with causal temporal velocity...")
    t0 = time.time()
    augmented_cal = add_temporal_velocity_features(
        cal_df,
        window_seconds=10,
        historical_context=historical_context_df,
    )
    print(f"Calibration velocity computed in {time.time() - t0:.2f}s.")

    # Transform through pipeline fitted STRICTLY on training data
    print("Transforming calibration flows through fitted training pipeline...")
    X_cal_scaled = pipeline.transform(augmented_cal, batch_size=SCORING_BATCH_SIZE)

    # Compute unweighted reconstruction scores
    print("Computing unweighted MSE calibration scores...")
    cal_scores = compute_unweighted_reconstruction_errors(model, X_cal_scaled, batch_size=SCORING_BATCH_SIZE)

    # Quantile threshold calibration with method='higher'
    primary_thresh = float(np.quantile(cal_scores, 1.0 - PRIMARY_ALPHA, method="higher"))
    ref_thresh = float(np.quantile(cal_scores, 1.0 - REFERENCE_ALPHA, method="higher"))

    # Empirical validation on calibration partition (score > threshold)
    primary_fp = int(np.sum(cal_scores > primary_thresh))
    primary_emp_fpr = float(primary_fp / len(cal_scores))

    ref_fp = int(np.sum(cal_scores > ref_thresh))
    ref_emp_fpr = float(ref_fp / len(cal_scores))

    print("\n=== CALIBRATION THRESHOLD RESULTS ===")
    print(f"Calibration Sample Count:    {len(cal_scores):,}")
    print(f"Primary Threshold (0.2%):    {primary_thresh:.6f} -> FP: {primary_fp}/{len(cal_scores)} (Empirical FPR: {primary_emp_fpr:.4%})")
    print(f"Reference Threshold (1.0%):  {ref_thresh:.6f} -> FP: {ref_fp}/{len(cal_scores)} (Empirical FPR: {ref_emp_fpr:.4%})")

    cal_meta = {
        "calibration_rows": len(cal_scores),
        "primary_alpha": PRIMARY_ALPHA,
        "primary_threshold": round(primary_thresh, 6),
        "primary_false_positives": primary_fp,
        "primary_empirical_fpr": round(primary_emp_fpr, 6),
        "reference_alpha": REFERENCE_ALPHA,
        "reference_threshold": round(ref_thresh, 6),
        "reference_false_positives": ref_fp,
        "reference_empirical_fpr": round(ref_emp_fpr, 6),
        "quantile_method": "higher",
        "scoring_formula": "mean((x - x_hat) ** 2)",
        "decision_rule": "score > threshold",
        "calibration_score_min": round(float(np.min(cal_scores)), 6),
        "calibration_score_median": round(float(np.median(cal_scores)), 6),
        "calibration_score_mean": round(float(np.mean(cal_scores)), 6),
        "calibration_score_p99": round(float(np.percentile(cal_scores, 99)), 6),
        "calibration_score_max": round(float(np.max(cal_scores)), 6),
    }

    return cal_scores, cal_meta


def save_v2_artifacts(
    model: MLPRegressor,
    pipeline: PureOneWayLogPreprocessingPipeline,
    split_meta: Dict[str, Any],
    convergence_meta: Dict[str, Any],
    cal_meta: Dict[str, Any],
) -> Path:
    """
    Saves all Phase 2 assets atomically into models/pure_one_way_v2/.
    Guarantees existing champion models in models/ are untouched.
    """
    # Safe directory creation
    V2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = V2_ARTIFACTS_DIR / ".tmp_save"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPersisting v2 artifacts to {V2_ARTIFACTS_DIR}...")

    # 1. Model weights
    joblib.dump(model, tmp_dir / "model.joblib")

    # 2. Fitted preprocessing pipeline
    joblib.dump(pipeline, tmp_dir / "pipeline.joblib")

    # 3. Feature manifest
    feature_manifest = {
        "model_version": "2.0-pure-oneway",
        "manifest_identifier": "PURE_ONE_WAY_FEATURES_v2",
        "feature_count": len(PURE_ONE_WAY_FEATURES),
        "features": list(PURE_ONE_WAY_FEATURES),
        "pipeline_class": pipeline.__class__.__name__,
        "imputer_strategy": "median",
        "scaling": "StandardScaler",
        "log_transform_features": list(pipeline.features),
        "provenance": "Strict unidirectional forward sensor (SIH26145 compliant)",
    }
    with open(tmp_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, indent=2)

    # 4. Training configuration and diagnostics
    train_config = {
        "model_version": "2.0-pure-oneway",
        "architecture": [26, 20, 8, 20, 26],
        "hidden_layer_sizes": [20, 8, 20],
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": LEARNING_RATE_INIT,
        "learning_rate": "adaptive",
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "max_iter": MAX_ITER,
        "early_stopping": False,
        "random_state": RANDOM_STATE,
        "dataset_split": split_meta,
        "training_convergence": convergence_meta,
    }
    with open(tmp_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    # 5. Threshold configuration
    threshold_config = {
        "model_version": "2.0-pure-oneway",
        "calibration": cal_meta,
    }
    with open(tmp_dir / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)

    # Atomic move to final files
    for fname in ["model.joblib", "pipeline.joblib", "feature_manifest.json", "train_config.json", "threshold_config.json"]:
        dest = V2_ARTIFACTS_DIR / fname
        shutil.move(str(tmp_dir / fname), str(dest))

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("All Phase 2 artifacts successfully persisted:")
    for f in V2_ARTIFACTS_DIR.glob("*"):
        if f.is_file():
            print(f"   [SAVED] {f.name} ({f.stat().st_size:,} bytes)")

    return V2_ARTIFACTS_DIR


def main():
    print("=" * 80)
    print("AEGIS PHASE 2 STAGE 2: PURE ONE-WAY AUTOENCODER TRAINING")
    print("Model: 26 -> 20 -> 8 -> 20 -> 26 | Input: PURE_ONE_WAY_FEATURES (v2.0)")
    print("SIH Problem ID: SIH26145 | Artifact Isolation: models/pure_one_way_v2/")
    print("=" * 80)

    # 1. Prepare data partitions
    train_sample, val_sample, split_meta = prepare_training_and_validation_data()

    # 2. Fit preprocessing pipeline strictly on training sample
    print("\nFitting PureOneWayLogPreprocessingPipeline strictly on training sample...")
    pipeline = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    pipeline.fit(train_sample)

    # 3. Transform training and validation data
    print("Transforming training observations...")
    X_train_scaled = pipeline.transform(train_sample, batch_size=SCORING_BATCH_SIZE)
    print("Transforming validation observations...")
    X_val_scaled = pipeline.transform(val_sample, batch_size=SCORING_BATCH_SIZE)

    # 4. Verify feature dimensions
    assert X_train_scaled.shape[1] == 26, f"Expected 26 features, got {X_train_scaled.shape[1]}"
    assert X_val_scaled.shape[1] == 26, f"Expected 26 features, got {X_val_scaled.shape[1]}"

    # 5. Train autoencoder
    model, convergence_meta = train_autoencoder(X_train_scaled, X_val_scaled)

    # 6. Compatibility verification
    is_compat, reason = check_model_compatibility(model, pipeline, PURE_ONE_WAY_FEATURES)
    if not is_compat:
        raise RuntimeError(f"Compatibility verification failed: {reason}")
    print(f"\nModel and Pipeline Compatibility Check: [PASS] ({reason})")

    # 7. Calibrate thresholds on Monday benign calibration partition
    _, cal_meta = calibrate_thresholds(
        model=model,
        pipeline=pipeline,
        historical_context_df=train_sample,
    )

    # 8. Persist artifacts to models/pure_one_way_v2/
    save_v2_artifacts(
        model=model,
        pipeline=pipeline,
        split_meta=split_meta,
        convergence_meta=convergence_meta,
        cal_meta=cal_meta,
    )

    print("\n" + "=" * 80)
    print("STAGE 2 TRAINING & CALIBRATION COMPLETE — ARTIFACTS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
