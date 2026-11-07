"""
src/evaluate_operational.py
AEGIS Phase 2 — Stage 3: Full Operational Evaluation

Evaluates and compares:
1. Pure One-Way 26-Feature Autoencoder (v2) [models/pure_one_way_v2/]
   - Primary Threshold: 0.314036 (target FPR: 0.2%)
   - Reference Threshold: 0.099018 (target FPR: 1.0%)
   - Unweighted Mean Squared Error (MSE)
2. Legacy 32-Feature Champion Model (v1) [models/temporal_autoencoder_*]
   - Established Threshold: 0.19088475755009496 (target FPR: 0.2%)
   - Residual-Weighted Mean Squared Error (MSE)

Dataset: outputs/experiment/test_tue_fri.parquet (2,300,825 flows, Tuesday–Friday, 14 attacks)

STRICT EVALUATION SCOPE:
- Zero model retraining or parameter adaptation
- Zero threshold recalculation or operational label contamination
- Leakage-free, reproducible batch evaluation
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

# Repository root and path setup
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "outputs" / "experiment"
MODELS_DIR = ROOT_DIR / "models"
V2_MODELS_DIR = MODELS_DIR / "pure_one_way_v2"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "phase_2_operational_results"

sys.path.insert(0, str(SRC_DIR))
from features import (
    PURE_ONE_WAY_FEATURES,
    add_temporal_velocity_features,
    sanitize_labels,
)

TEST_DATA_FILE = DATA_DIR / "test_tue_fri.parquet"
BATCH_SIZE = 50_000

# Legacy champion calendar-bucket temporal velocity feature computation
def add_temporal_velocity_features_legacy(df: pd.DataFrame, window_seconds: int = 10) -> pd.DataFrame:
    """Computes legacy calendar-bucketed temporal velocity features as used by v1 champion."""
    df = df.copy()
    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column required for legacy temporal velocity calculation.")
    time_bucket = df["Timestamp"].dt.floor(f"{window_seconds}s")
    flow_counts = df.groupby(time_bucket)["Total Fwd Packets"].transform("count")
    df["Rolling_Flow_Velocity_10s"] = flow_counts.astype(float)
    byte_sums = df.groupby(time_bucket)["Total Length of Fwd Packets"].transform("sum")
    df["Rolling_Bytes_Velocity_10s"] = byte_sums.astype(float)
    is_single = (df["Total Fwd Packets"] <= 1).astype(float)
    single_ratio = is_single.groupby(time_bucket).transform("mean")
    df["Rolling_Single_Pkt_Ratio_10s"] = single_ratio.astype(float)
    return df


def compute_unweighted_mse_scores(model: Any, scaled_data: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Computes unweighted MSE scores across batches."""
    errors = []
    n_samples = len(scaled_data)
    for start in range(0, n_samples, batch_size):
        batch = scaled_data[start : start + batch_size]
        reconstructed = model.predict(batch)
        batch_mse = np.mean(np.square(batch - reconstructed), axis=1)
        errors.append(batch_mse)
    return np.concatenate(errors) if errors else np.empty(0, dtype=float)


def compute_weighted_mse_scores(
    model: Any, scaled_data: np.ndarray, weights: np.ndarray, batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """Computes residual-weighted MSE scores across batches."""
    errors = []
    n_samples = len(scaled_data)
    for start in range(0, n_samples, batch_size):
        batch = scaled_data[start : start + batch_size]
        reconstructed = model.predict(batch)
        batch_mse = np.mean(np.square(batch - reconstructed) * weights, axis=1)
        errors.append(batch_mse)
    return np.concatenate(errors) if errors else np.empty(0, dtype=float)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    labels: pd.Series,
    dates: pd.Series,
    model_name: str,
    threshold: float,
    threshold_name: str,
    timing_meta: Dict[str, float],
) -> Dict[str, Any]:
    """
    Computes complete statistical detection metrics, per-category breakdown,
    and daily breakdown.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)

    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Continuous curve metrics
    print(f"      Computing ROC-AUC and PR-AUC for {model_name} ({threshold_name})...")
    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    # Per-category metrics across all 15 categories (BENIGN + 14 attacks)
    category_metrics = {}
    unique_categories = sorted(labels.unique())
    for cat in unique_categories:
        mask = (labels == cat).to_numpy()
        cat_total = int(np.sum(mask))
        cat_detected = int(np.sum(y_pred[mask] == 1))
        cat_missed = cat_total - cat_detected
        cat_recall = float(cat_detected / cat_total) if cat_total > 0 else 0.0

        # Calculate per-class precision and F1 if it's an attack
        if cat == "BENIGN":
            # For benign, detection means false positive
            cat_precision = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            cat_fpr = float(cat_detected / cat_total) if cat_total > 0 else 0.0
            category_metrics[cat] = {
                "total": cat_total,
                "detected_as_anomaly": cat_detected,
                "classified_benign": cat_missed,
                "false_positive_rate_pct": round(cat_fpr * 100, 4),
                "accuracy_pct": round((1.0 - cat_fpr) * 100, 4),
            }
        else:
            cat_y_true = mask.astype(int)
            # Binary precision treating only this attack as positive vs all other flows
            cat_tp = cat_detected
            cat_fp = int(np.sum((y_pred == 1) & (~mask)))
            cat_prec = float(cat_tp / (cat_tp + cat_fp)) if (cat_tp + cat_fp) > 0 else 0.0
            cat_f1 = (
                float(2 * cat_prec * cat_recall / (cat_prec + cat_recall))
                if (cat_prec + cat_recall) > 0
                else 0.0
            )
            category_metrics[cat] = {
                "total": cat_total,
                "detected": cat_detected,
                "missed": cat_missed,
                "recall_pct": round(cat_recall * 100, 4),
                "specific_precision_pct": round(cat_prec * 100, 4),
                "specific_f1_score": round(cat_f1, 4),
            }

    # Per-day metrics
    day_metrics = {}
    day_mapping = {
        "2017-07-04": "Tuesday",
        "2017-07-05": "Wednesday",
        "2017-07-06": "Thursday",
        "2017-07-07": "Friday",
    }
    for date_str, day_name in day_mapping.items():
        date_mask = (dates.astype(str) == date_str).to_numpy()
        d_y_true = y_true[date_mask]
        d_y_pred = y_pred[date_mask]
        d_tn, d_fp, d_fn, d_tp = confusion_matrix(d_y_true, d_y_pred, labels=[0, 1]).ravel()
        d_total = len(d_y_true)
        d_recall = float(recall_score(d_y_true, d_y_pred, zero_division=0))
        d_prec = float(precision_score(d_y_true, d_y_pred, zero_division=0))
        d_f1 = float(f1_score(d_y_true, d_y_pred, zero_division=0))
        d_fpr = float(d_fp / (d_fp + d_tn)) if (d_fp + d_tn) > 0 else 0.0

        day_metrics[day_name] = {
            "date": date_str,
            "total_flows": int(d_total),
            "benign_flows": int(d_tn + d_fp),
            "attack_flows": int(d_tp + d_fn),
            "tp": int(d_tp),
            "fp": int(d_fp),
            "tn": int(d_tn),
            "fn": int(d_fn),
            "recall_pct": round(d_recall * 100, 4),
            "false_positive_rate_pct": round(d_fpr * 100, 4),
            "precision_pct": round(d_prec * 100, 4),
            "f1_score": round(d_f1, 4),
        }

    return {
        "model_name": model_name,
        "threshold_name": threshold_name,
        "threshold_value": float(threshold),
        "total_flows": int(total),
        "benign_flows": int(tn + fp),
        "attack_flows": int(tp + fn),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "rates": {
            "detection_rate_pct": round(recall * 100, 4),
            "false_positive_rate_pct": round(fpr * 100, 4),
            "false_negative_rate_pct": round(fnr * 100, 4),
            "specificity_pct": round(specificity * 100, 4),
        },
        "classification_metrics": {
            "precision_pct": round(precision * 100, 4),
            "f1_score": round(f1, 4),
            "accuracy_pct": round(accuracy * 100, 4),
            "balanced_accuracy_pct": round(bal_acc * 100, 4),
        },
        "continuous_curves": {
            "roc_auc": round(roc_auc, 6),
            "pr_auc": round(pr_auc, 6),
        },
        "timing_and_throughput": timing_meta,
        "per_category": category_metrics,
        "per_day": day_metrics,
    }


def main():
    total_start = time.time()
    print("=" * 80)
    print("AEGIS PHASE 2 — STAGE 3: FULL OPERATIONAL EVALUATION")
    print("Repository: RAZA1409/label-free-threat-detection")
    print("SIH Problem ID: SIH26145")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load operational test dataset
    print(f"\n1. Ingesting operational test partition: {TEST_DATA_FILE}...")
    t0 = time.time()
    test_df = pd.read_parquet(TEST_DATA_FILE)
    ingest_time = time.time() - t0
    n_flows = len(test_df)
    print(f"   Loaded {n_flows:,} flows ({len(test_df.columns)} columns) in {ingest_time:.2f}s.")

    # 2. Extract labels and temporal attributes
    print("\n2. Sanitizing labels and extracting operational dates...")
    cleaned_labels = sanitize_labels(test_df["Label"])
    y_true = (cleaned_labels != "BENIGN").astype(int).to_numpy()
    flow_dates = test_df["Timestamp"].dt.date
    n_benign = int(np.sum(y_true == 0))
    n_attacks = int(np.sum(y_true == 1))
    print(f"   Benign flows: {n_benign:,} ({n_benign / n_flows:.2%})")
    print(f"   Attack flows: {n_attacks:,} ({n_attacks / n_flows:.2%}) across 14 categories.")

    # 3. Evaluate Pure One-Way Model (v2)
    print("\n" + "=" * 80)
    print("3. EVALUATING PURE ONE-WAY 26-FEATURE AUTOENCODER (v2)")
    print("=" * 80)

    # Load v2 artifacts
    v2_model_path = V2_MODELS_DIR / "model.joblib"
    v2_pipe_path = V2_MODELS_DIR / "pipeline.joblib"
    v2_thresh_path = V2_MODELS_DIR / "threshold_config.json"

    print(f"   Loading v2 model from:    {v2_model_path.name}")
    v2_model = joblib.load(v2_model_path)
    print(f"   Loading v2 pipeline from: {v2_pipe_path.name}")
    v2_pipeline = joblib.load(v2_pipe_path)
    with open(v2_thresh_path, "r", encoding="utf-8") as f:
        v2_thresh_cfg = json.load(f)["calibration"]

    primary_thresh = float(v2_thresh_cfg["primary_threshold"])
    ref_thresh = float(v2_thresh_cfg["reference_threshold"])
    print(f"   Primary Threshold (alpha=0.002):   {primary_thresh:.6f}")
    print(f"   Reference Threshold (alpha=0.010): {ref_thresh:.6f}")

    # Feature engineering for v2
    print("\n   [v2] Computing causal backward-looking rolling features (10s window)...")
    t0 = time.time()
    v2_aug_df = add_temporal_velocity_features(test_df, window_seconds=10)
    v2_fe_time = time.time() - t0
    v2_fe_throughput = n_flows / v2_fe_time if v2_fe_time > 0 else 0
    print(f"   [v2] Feature engineering completed in {v2_fe_time:.2f}s ({v2_fe_throughput:,.0f} flows/sec).")

    # Transformation for v2
    print("   [v2] Transforming 26 features through training-fitted pipeline...")
    t0 = time.time()
    X_v2_scaled = v2_pipeline.transform(v2_aug_df, batch_size=BATCH_SIZE)
    v2_trans_time = time.time() - t0
    v2_trans_throughput = n_flows / v2_trans_time if v2_trans_time > 0 else 0
    print(f"   [v2] Transformation completed in {v2_trans_time:.2f}s ({v2_trans_throughput:,.0f} flows/sec).")

    # Scoring for v2
    print("   [v2] Scoring unweighted MSE reconstruction error...")
    t0 = time.time()
    scores_v2 = compute_unweighted_mse_scores(v2_model, X_v2_scaled, batch_size=BATCH_SIZE)
    v2_score_time = time.time() - t0
    v2_score_throughput = n_flows / v2_score_time if v2_score_time > 0 else 0
    v2_total_time = v2_fe_time + v2_trans_time + v2_score_time
    v2_total_throughput = n_flows / v2_total_time if v2_total_time > 0 else 0
    print(f"   [v2] Scoring completed in {v2_score_time:.2f}s ({v2_score_throughput:,.0f} flows/sec).")
    print(f"   [v2] Total processing time: {v2_total_time:.2f}s ({v2_total_throughput:,.0f} flows/sec).")

    v2_timing_meta = {
        "feature_engineering_seconds": round(v2_fe_time, 2),
        "transformation_seconds": round(v2_trans_time, 2),
        "scoring_inference_seconds": round(v2_score_time, 2),
        "total_processing_seconds": round(v2_total_time, 2),
        "total_throughput_flows_per_second": round(v2_total_throughput, 1),
        "latency_microseconds_per_flow": round((v2_total_time / n_flows) * 1e6, 2),
    }

    # Evaluate v2 primary threshold
    print("\n   Evaluating v2 at Primary Threshold (0.314036)...")
    y_pred_v2_primary = (scores_v2 > primary_thresh).astype(int)
    res_v2_primary = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred_v2_primary,
        scores=scores_v2,
        labels=cleaned_labels,
        dates=flow_dates,
        model_name="Pure One-Way Autoencoder (v2)",
        threshold=primary_thresh,
        threshold_name="Primary (alpha=0.002)",
        timing_meta=v2_timing_meta,
    )

    # Evaluate v2 reference threshold
    print("   Evaluating v2 at Reference Threshold (0.099018)...")
    y_pred_v2_ref = (scores_v2 > ref_thresh).astype(int)
    res_v2_ref = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred_v2_ref,
        scores=scores_v2,
        labels=cleaned_labels,
        dates=flow_dates,
        model_name="Pure One-Way Autoencoder (v2)",
        threshold=ref_thresh,
        threshold_name="Reference (alpha=0.010)",
        timing_meta=v2_timing_meta,
    )

    # Clean up large v2 intermediate dataframes from memory
    del v2_aug_df, X_v2_scaled

    # 4. Evaluate Legacy Champion Model (v1)
    print("\n" + "=" * 80)
    print("4. EVALUATING LEGACY 32-FEATURE CHAMPION MODEL (v1)")
    print("=" * 80)

    leg_model_path = MODELS_DIR / "temporal_autoencoder_model.joblib"
    leg_pipe_path = MODELS_DIR / "temporal_autoencoder_pipeline.joblib"
    leg_weights_path = MODELS_DIR / "temporal_autoencoder_weights.joblib"
    leg_thresh_path = MODELS_DIR / "temporal_autoencoder_threshold.json"

    print(f"   Loading legacy model from:    {leg_model_path.name}")
    leg_model = joblib.load(leg_model_path)
    print(f"   Loading legacy pipeline from: {leg_pipe_path.name}")
    leg_pipeline = joblib.load(leg_pipe_path)
    print(f"   Loading legacy weights from:  {leg_weights_path.name}")
    leg_weights = joblib.load(leg_weights_path)
    with open(leg_thresh_path, "r", encoding="utf-8") as f:
        leg_thresh = float(json.load(f)["threshold"])
    print(f"   Legacy Established Threshold: {leg_thresh:.6f}")

    # Feature engineering for legacy
    print("\n   [v1] Computing legacy calendar-bucketed temporal features...")
    t0 = time.time()
    leg_aug_df = add_temporal_velocity_features_legacy(test_df, window_seconds=10)
    leg_fe_time = time.time() - t0
    leg_fe_throughput = n_flows / leg_fe_time if leg_fe_time > 0 else 0
    print(f"   [v1] Feature engineering completed in {leg_fe_time:.2f}s ({leg_fe_throughput:,.0f} flows/sec).")

    # Transformation for legacy
    print("   [v1] Transforming 32 features through legacy pipeline...")
    t0 = time.time()
    X_leg_scaled = leg_pipeline.transform(leg_aug_df, batch_size=BATCH_SIZE)
    leg_trans_time = time.time() - t0
    leg_trans_throughput = n_flows / leg_trans_time if leg_trans_time > 0 else 0
    print(f"   [v1] Transformation completed in {leg_trans_time:.2f}s ({leg_trans_throughput:,.0f} flows/sec).")

    # Scoring for legacy
    print("   [v1] Scoring residual-weighted MSE reconstruction error...")
    t0 = time.time()
    scores_leg = compute_weighted_mse_scores(leg_model, X_leg_scaled, leg_weights, batch_size=BATCH_SIZE)
    leg_score_time = time.time() - t0
    leg_score_throughput = n_flows / leg_score_time if leg_score_time > 0 else 0
    leg_total_time = leg_fe_time + leg_trans_time + leg_score_time
    leg_total_throughput = n_flows / leg_total_time if leg_total_time > 0 else 0
    print(f"   [v1] Scoring completed in {leg_score_time:.2f}s ({leg_score_throughput:,.0f} flows/sec).")
    print(f"   [v1] Total processing time: {leg_total_time:.2f}s ({leg_total_throughput:,.0f} flows/sec).")

    leg_timing_meta = {
        "feature_engineering_seconds": round(leg_fe_time, 2),
        "transformation_seconds": round(leg_trans_time, 2),
        "scoring_inference_seconds": round(leg_score_time, 2),
        "total_processing_seconds": round(leg_total_time, 2),
        "total_throughput_flows_per_second": round(leg_total_throughput, 1),
        "latency_microseconds_per_flow": round((leg_total_time / n_flows) * 1e6, 2),
    }

    # Evaluate legacy champion
    print("\n   Evaluating legacy champion at Established Threshold (0.190885)...")
    y_pred_leg = (scores_leg >= leg_thresh).astype(int)
    res_leg = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred_leg,
        scores=scores_leg,
        labels=cleaned_labels,
        dates=flow_dates,
        model_name="Legacy Temporal Autoencoder (v1)",
        threshold=leg_thresh,
        threshold_name="Established (alpha=0.002)",
        timing_meta=leg_timing_meta,
    )

    del leg_aug_df, X_leg_scaled

    # 5. Comparative Performance Summary Table
    print("\n" + "=" * 80)
    print("5. OPERATIONAL EVALUATION BENCHMARK SUMMARY")
    print("=" * 80)

    summary_rows = [
        {
            "Model": "Pure One-Way (v2) [Primary]",
            "Features": "26 (Pure One-Way)",
            "Threshold": f"{primary_thresh:.6f}",
            "Detection Rate (Recall)": f"{res_v2_primary['rates']['detection_rate_pct']:.2f}%",
            "FPR": f"{res_v2_primary['rates']['false_positive_rate_pct']:.2f}%",
            "Precision": f"{res_v2_primary['classification_metrics']['precision_pct']:.2f}%",
            "F1-Score": f"{res_v2_primary['classification_metrics']['f1_score']:.4f}",
            "Balanced Acc": f"{res_v2_primary['classification_metrics']['balanced_accuracy_pct']:.2f}%",
            "PR-AUC (AP)": f"{res_v2_primary['continuous_curves']['pr_auc']:.4f}",
            "ROC-AUC": f"{res_v2_primary['continuous_curves']['roc_auc']:.4f}",
            "Throughput": f"{res_v2_primary['timing_and_throughput']['total_throughput_flows_per_second']:,.0f} fl/s",
        },
        {
            "Model": "Pure One-Way (v2) [Reference]",
            "Features": "26 (Pure One-Way)",
            "Threshold": f"{ref_thresh:.6f}",
            "Detection Rate (Recall)": f"{res_v2_ref['rates']['detection_rate_pct']:.2f}%",
            "FPR": f"{res_v2_ref['rates']['false_positive_rate_pct']:.2f}%",
            "Precision": f"{res_v2_ref['classification_metrics']['precision_pct']:.2f}%",
            "F1-Score": f"{res_v2_ref['classification_metrics']['f1_score']:.4f}",
            "Balanced Acc": f"{res_v2_ref['classification_metrics']['balanced_accuracy_pct']:.2f}%",
            "PR-AUC (AP)": f"{res_v2_ref['continuous_curves']['pr_auc']:.4f}",
            "ROC-AUC": f"{res_v2_ref['continuous_curves']['roc_auc']:.4f}",
            "Throughput": f"{res_v2_ref['timing_and_throughput']['total_throughput_flows_per_second']:,.0f} fl/s",
        },
        {
            "Model": "Legacy Champion (v1)",
            "Features": "32 (Bidirectional leak)",
            "Threshold": f"{leg_thresh:.6f}",
            "Detection Rate (Recall)": f"{res_leg['rates']['detection_rate_pct']:.2f}%",
            "FPR": f"{res_leg['rates']['false_positive_rate_pct']:.2f}%",
            "Precision": f"{res_leg['classification_metrics']['precision_pct']:.2f}%",
            "F1-Score": f"{res_leg['classification_metrics']['f1_score']:.4f}",
            "Balanced Acc": f"{res_leg['classification_metrics']['balanced_accuracy_pct']:.2f}%",
            "PR-AUC (AP)": f"{res_leg['continuous_curves']['pr_auc']:.4f}",
            "ROC-AUC": f"{res_leg['continuous_curves']['roc_auc']:.4f}",
            "Throughput": f"{res_leg['timing_and_throughput']['total_throughput_flows_per_second']:,.0f} fl/s",
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # 6. Save Machine-Readable Artifacts
    print("\n" + "=" * 80)
    print("6. PERSISTING EVALUATION ARTIFACTS TO REPORTS DIRECTORY")
    print("=" * 80)

    # A. overall_metrics.json
    overall_path = RESULTS_DIR / "overall_metrics.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation_metadata": {
                    "dataset": "outputs/experiment/test_tue_fri.parquet",
                    "total_flows": n_flows,
                    "benign_flows": n_benign,
                    "attack_flows": n_attacks,
                    "attack_categories_count": 14,
                    "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                },
                "models": {
                    "pure_one_way_v2_primary": res_v2_primary,
                    "pure_one_way_v2_reference": res_v2_ref,
                    "legacy_champion_v1": res_leg,
                },
            },
            f,
            indent=2,
        )
    print(f"   [OK] Saved overall metrics to: {overall_path}")

    # B. confusion_matrices.json
    cm_path = RESULTS_DIR / "confusion_matrices.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pure_one_way_v2_primary": res_v2_primary["confusion_matrix"],
                "pure_one_way_v2_reference": res_v2_ref["confusion_matrix"],
                "legacy_champion_v1": res_leg["confusion_matrix"],
            },
            f,
            indent=2,
        )
    print(f"   [OK] Saved confusion matrices to: {cm_path}")

    # C. per_attack_metrics.csv
    # Build a combined CSV comparing per-attack recall across models
    attack_rows = []
    unique_categories = sorted(cleaned_labels.unique())
    for cat in unique_categories:
        v2_prim_cat = res_v2_primary["per_category"][cat]
        v2_ref_cat = res_v2_ref["per_category"][cat]
        leg_cat = res_leg["per_category"][cat]

        total_cnt = v2_prim_cat["total"]
        if cat == "BENIGN":
            attack_rows.append({
                "Category": cat,
                "Total_Count": total_cnt,
                "v2_Primary_Detected": v2_prim_cat["detected_as_anomaly"],
                "v2_Primary_Rate_Pct": v2_prim_cat["false_positive_rate_pct"],
                "v2_Ref_Detected": v2_ref_cat["detected_as_anomaly"],
                "v2_Ref_Rate_Pct": v2_ref_cat["false_positive_rate_pct"],
                "v1_Legacy_Detected": leg_cat["detected_as_anomaly"],
                "v1_Legacy_Rate_Pct": leg_cat["false_positive_rate_pct"],
                "Metric_Type": "False Positive Rate (FPR)",
            })
        else:
            attack_rows.append({
                "Category": cat,
                "Total_Count": total_cnt,
                "v2_Primary_Detected": v2_prim_cat["detected"],
                "v2_Primary_Rate_Pct": v2_prim_cat["recall_pct"],
                "v2_Ref_Detected": v2_ref_cat["detected"],
                "v2_Ref_Rate_Pct": v2_ref_cat["recall_pct"],
                "v1_Legacy_Detected": leg_cat["detected"],
                "v1_Legacy_Rate_Pct": leg_cat["recall_pct"],
                "Metric_Type": "Detection Rate (Recall)",
            })

    attack_df = pd.DataFrame(attack_rows)
    per_attack_path = RESULTS_DIR / "per_attack_metrics.csv"
    attack_df.to_csv(per_attack_path, index=False)
    print(f"   [OK] Saved per-attack metrics to: {per_attack_path}")

    # Print Attack Detection Breakdown Table
    print("\nPER-ATTACK DETECTION BREAKDOWN (All 14 Attack Categories):")
    print(attack_df.to_string(index=False))

    # D. per_day_metrics.csv
    day_rows = []
    for day_name in ["Tuesday", "Wednesday", "Thursday", "Friday"]:
        d_v2_prim = res_v2_primary["per_day"][day_name]
        d_v2_ref = res_v2_ref["per_day"][day_name]
        d_leg = res_leg["per_day"][day_name]

        day_rows.append({
            "Day": day_name,
            "Date": d_v2_prim["date"],
            "Total_Flows": d_v2_prim["total_flows"],
            "Benign_Flows": d_v2_prim["benign_flows"],
            "Attack_Flows": d_v2_prim["attack_flows"],
            "v2_Prim_Recall_Pct": d_v2_prim["recall_pct"],
            "v2_Prim_FPR_Pct": d_v2_prim["false_positive_rate_pct"],
            "v2_Prim_Prec_Pct": d_v2_prim["precision_pct"],
            "v2_Prim_F1": d_v2_prim["f1_score"],
            "v2_Ref_Recall_Pct": d_v2_ref["recall_pct"],
            "v2_Ref_FPR_Pct": d_v2_ref["false_positive_rate_pct"],
            "v2_Ref_Prec_Pct": d_v2_ref["precision_pct"],
            "v2_Ref_F1": d_v2_ref["f1_score"],
            "v1_Leg_Recall_Pct": d_leg["recall_pct"],
            "v1_Leg_FPR_Pct": d_leg["false_positive_rate_pct"],
            "v1_Leg_Prec_Pct": d_leg["precision_pct"],
            "v1_Leg_F1": d_leg["f1_score"],
        })

    day_df = pd.DataFrame(day_rows)
    per_day_path = RESULTS_DIR / "per_day_metrics.csv"
    day_df.to_csv(per_day_path, index=False)
    print(f"   [OK] Saved per-day metrics to: {per_day_path}")

    print("\nPER-DAY OPERATIONAL PERFORMANCE (Tuesday - Friday):")
    print(day_df.to_string(index=False))

    total_duration = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"STAGE 3 OPERATIONAL EVALUATION COMPLETE IN {total_duration:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
