"""
src/features.py
Unified Feature Engineering, Cleaning, and Hygiene Module for One-Way Threat Detection.

This module resolves:
1. Label encoding and mojibake normalization (e.g. em-dash issues).
2. De-duplication of raw features (removes duplicate 'Fwd Header Length.1').
3. Sanitization of negative sentinels (-1 in TCP window, negative durations).
4. Strict separation between:
   - ONE_WAY_FEATURES: Unidirectional (forward-only) flow features for SIH26145 compliance.
   - BIDIRECTIONAL_FEATURES: Full reference feature set.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score,
)

# --------------------------------------------------------------------------
# 1. Feature Dictionaries
# --------------------------------------------------------------------------

# Features strictly observable in Unidirectional (One-Way) network traffic
# (e.g., optical data diode, receive-only mirror tap, unidirectional gateway).
# No backward packets, backward timing, or reverse TCP parameters exist.
ONE_WAY_FEATURES: List[str] = [
    # Volume & Counts
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "act_data_pkt_fwd",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    # Header & Segment Metrics
    "Fwd Header Length",
    "min_seg_size_forward",
    "Avg Fwd Segment Size",
    # Packet Size Statistics
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    # Timing & Rates
    "Flow Duration",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Fwd Packets/s",
    "Flow Bytes/s",
    # Forward TCP Flags & State
    "Init_Win_bytes_forward",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "SYN Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "PSH Flag Count",
]

# Physical features that can never realistically be negative.
# In CIC-IDS2017, negative values arise from clock skew or integer overflow sentinels.
NON_NEGATIVE_PHYSICAL_FEATURES: List[str] = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Min",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Min",
]

# TCP window features where -1 represents no window / non-TCP flow.
TCP_WINDOW_FEATURES: List[str] = [
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

# Duplicate or non-feature identifier columns to strip
COLUMNS_TO_DROP: List[str] = [
    "Fwd Header Length.1",
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "CaptureDate",
]

# --------------------------------------------------------------------------
# 2. Sanitization Functions
# --------------------------------------------------------------------------

def sanitize_labels(series: pd.Series) -> pd.Series:
    """Normalize label strings and fix Unicode mojibake / corrupted dashes."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\x00-\x7F]+", "-", regex=True)
        .str.replace(r"-+", "-", regex=True)
        .str.strip()
    )
    # Standardize known variations
    cleaned = cleaned.replace({
        "Web Attack - Brute Force": "Web Attack - Brute Force",
        "Web Attack - XSS": "Web Attack - XSS",
        "Web Attack - Sql Injection": "Web Attack - Sql Injection",
    })
    return cleaned


def clean_feature_frame(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    clamp_negatives: bool = True,
) -> pd.DataFrame:
    """
    Clean the feature dataframe:
    - Drops duplicate columns (e.g. Fwd Header Length.1)
    - Replaces infinities with NaN
    - Handles negative sentinels on physical non-negative features
    """
    df = df.copy()

    # Drop duplicate column names if any exist
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in COLUMNS_TO_DROP:
        if col in df.columns and (features is None or col not in features):
            df.drop(columns=[col], inplace=True)

    target_features = features if features is not None else [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c.lower() != "label"
    ]

    # Replace infinities
    df[target_features] = df[target_features].replace([np.inf, -np.inf], np.nan)

    if clamp_negatives:
        for col in NON_NEGATIVE_PHYSICAL_FEATURES:
            if col in df.columns and col in target_features:
                df.loc[df[col] < 0, col] = np.nan

        # For TCP window features: -1 is an indicator for no-TCP / uninitialized
        for col in TCP_WINDOW_FEATURES:
            if col in df.columns and col in target_features:
                df.loc[df[col] < 0, col] = np.nan

    return df[target_features]


class PreprocessingPipeline:
    """
    Leak-free feature preprocessing pipeline.
    Fits SimpleImputer and StandardScaler strictly on normal/benign training data.
    """

    def __init__(self, features: List[str], imputer_strategy: str = "median"):
        self.features = list(features)
        self.imputer = SimpleImputer(strategy=imputer_strategy)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "PreprocessingPipeline":
        cleaned_train = clean_feature_frame(train_df, features=self.features)
        imputed = self.imputer.fit_transform(cleaned_train)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, batch_size: int = 50_000) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        cleaned = clean_feature_frame(df, features=self.features)
        transformed_batches = []

        for start in range(0, len(cleaned), batch_size):
            batch = cleaned.iloc[start : start + batch_size]
            imputed = self.imputer.transform(batch)
            scaled = self.scaler.transform(imputed)
            transformed_batches.append(scaled)

        return np.vstack(transformed_batches) if transformed_batches else np.empty((0, len(self.features)))


# Features that follow heavy right-skewed power-law distributions
LOG_FEATURES: List[str] = [
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


# --------------------------------------------------------------------------
# 3. Anomaly Evaluation & Reporting Helpers
# --------------------------------------------------------------------------

def calibrate_threshold(
    calibration_scores: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Calibrate anomaly score threshold targeting a specified FPR on benign calibration."""
    return float(np.quantile(calibration_scores, 1.0 - target_fpr))


def evaluate_detection_performance(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate comprehensive intrusion detection metrics:
    - Overall: Confusion matrix, Detection Rate (Recall), FPR, Precision, PR-AUC
    - Per-attack breakdown: Detection rate for each attack class
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true_binary, y_pred_binary, labels=[0, 1]
    ).ravel()

    detection_rate = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    average_precision = float(average_precision_score(y_true_binary, scores))

    # Per-category breakdown
    category_results = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        total_cat = int(mask.sum())
        detected_cat = int(y_pred_binary[mask].sum())
        rate = float(detected_cat / total_cat) if total_cat > 0 else 0.0
        category_results[str(label)] = {
            "total": total_cat,
            "detected": detected_cat,
            "missed": total_cat - detected_cat,
            "detection_rate_pct": round(rate * 100, 2),
        }

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "detection_rate_pct": round(detection_rate * 100, 2),
        "false_positive_rate_pct": round(false_positive_rate * 100, 2),
        "precision_pct": round(precision * 100, 2),
        "average_precision": round(average_precision, 4),
        "per_category": category_results,
    }

