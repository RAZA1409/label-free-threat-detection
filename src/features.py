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

# Legacy v1.0 Features (maintained for backward compatibility with v1 models)
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

# Streaming Temporal Velocity Features (Rolling causal context)
# Computed strictly over forward-only traffic metrics
TEMPORAL_FEATURES: List[str] = [
    "Rolling_Flow_Velocity_10s",
    "Rolling_Bytes_Velocity_10s",
    "Rolling_Single_Pkt_Ratio_10s",
]

AUGMENTED_ONE_WAY_FEATURES: List[str] = ONE_WAY_FEATURES + TEMPORAL_FEATURES

# --------------------------------------------------------------------------
# 1B. STRICT PURE ONE-WAY FEATURE SPECIFICATION (SIH26145 v2.0 COMPLIANT)
# --------------------------------------------------------------------------
# Explicitly removes all reverse-direction dependencies identified during audit:
# - Flow Bytes/s: EXCLUDED (CICFlowMeter adds Bwd Bytes in numerator)
# - Flow Duration: EXCLUDED (Depends on reverse packet timestamp in 77.4% of flows)
# - Fwd Packets/s: EXCLUDED (CICFlowMeter divides by bidirectional Flow Duration)
# - ACK Flag Count: EXCLUDED (Flow-wide flag capturing return ACKs)
# - PSH Flag Count: EXCLUDED (Flow-wide flag, superseded by Fwd PSH Flags)
# - URG Flag Count: EXCLUDED (Flow-wide flag, superseded by Fwd URG Flags)
# - FIN Flag Count: EXCLUDED (Flow-wide flag capturing return teardowns)
# - RST Flag Count: EXCLUDED (Flow-wide flag)
#
# Adds pure forward derivations:
# - Fwd_Bytes_Per_Sec: Forward payload bytes / (Fwd IAT Total in seconds)
# - Fwd_Packets_Per_Sec: Forward packet count / (Fwd IAT Total in seconds)

PURE_ONE_WAY_BASE_FEATURES: List[str] = [
    # Forward Volume & Packet Counts (6)
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "act_data_pkt_fwd",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Fwd Header Length",
    # Forward Segment & TCP Parameters (3)
    "min_seg_size_forward",
    "Avg Fwd Segment Size",
    "Init_Win_bytes_forward",
    # Forward Packet Length Distributions (4)
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    # Forward Timing & Inter-Arrival Times (5)
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    # Pure Forward Directional Rates (2)
    "Fwd_Bytes_Per_Sec",
    "Fwd_Packets_Per_Sec",
    # Forward Directional TCP Flags (3)
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "SYN Flag Count",
]

PURE_ONE_WAY_TEMPORAL_FEATURES: List[str] = [
    "Rolling_Flow_Velocity_10s",
    "Rolling_Bytes_Velocity_10s",
    "Rolling_Single_Pkt_Ratio_10s",
]

# Total 26 pure one-way forward features (23 base forward + 3 rolling forward velocity)
PURE_ONE_WAY_FEATURES: List[str] = PURE_ONE_WAY_BASE_FEATURES + PURE_ONE_WAY_TEMPORAL_FEATURES

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

    cols_to_remove = [
        col for col in COLUMNS_TO_DROP
        if col in df.columns and (features is None or col not in features)
    ]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)

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


def add_temporal_velocity_features(df: pd.DataFrame, window_seconds: int = 10) -> pd.DataFrame:
    """
    Computes streaming forward temporal context without exposing labels or future data.
    Aggregates metrics strictly over causal, backward-looking rolling time windows (t - window, t].

    Guarantees:
    - Pure temporal causality: An event at timestamp t is never affected by any events with timestamps > t.
    - Preserves input order: Preserves the original DataFrame index and row sequence.
    - Robust edge case handling: Empty dataframes, duplicate timestamps, missing/infinite values.
    """
    if len(df) == 0:
        res = df.copy()
        res["Rolling_Flow_Velocity_10s"] = pd.Series(dtype=float)
        res["Rolling_Bytes_Velocity_10s"] = pd.Series(dtype=float)
        res["Rolling_Single_Pkt_Ratio_10s"] = pd.Series(dtype=float)
        return res

    df = df.copy()
    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column required for temporal velocity engine.")

    # Convert Timestamp to datetime if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Track original row sequence using stable position identifier
    df["_row_pos"] = np.arange(len(df))
    sorted_df = df.sort_values(["Timestamp", "_row_pos"], kind="stable").copy()

    # Pre-clean inputs for numerical aggregation
    fwd_pkts = sorted_df.get("Total Fwd Packets", pd.Series(0, index=sorted_df.index))
    fwd_pkts = fwd_pkts.replace([np.inf, -np.inf], np.nan).fillna(0)

    fwd_bytes = sorted_df.get("Total Length of Fwd Packets", pd.Series(0, index=sorted_df.index))
    fwd_bytes = fwd_bytes.replace([np.inf, -np.inf], np.nan).fillna(0)

    sorted_df["_fwd_pkts"] = fwd_pkts
    sorted_df["_fwd_bytes"] = fwd_bytes
    sorted_df["_is_single"] = (fwd_pkts <= 1).astype(float)

    # Strictly causal rolling window (t - window_seconds, t]
    rolling_obj = sorted_df.rolling(f"{window_seconds}s", on="Timestamp", closed="right")

    sorted_df["_rf"] = rolling_obj["_fwd_pkts"].count().astype(float)
    sorted_df["_rb"] = rolling_obj["_fwd_bytes"].sum().astype(float)
    sorted_df["_rs"] = rolling_obj["_is_single"].mean().astype(float)

    # For simultaneous events occurring at the exact same timestamp t, all such events
    # share the full causal aggregate accumulated up to timestamp t (the last row of timestamp t).
    # Events occurring at t' > t strictly do not affect timestamp t.
    sorted_df["Rolling_Flow_Velocity_10s"] = sorted_df.groupby("Timestamp")["_rf"].transform("last")
    sorted_df["Rolling_Bytes_Velocity_10s"] = sorted_df.groupby("Timestamp")["_rb"].transform("last")
    sorted_df["Rolling_Single_Pkt_Ratio_10s"] = sorted_df.groupby("Timestamp")["_rs"].transform("last")

    # Restore original row ordering and clean up temporary helper columns
    result = sorted_df.sort_values("_row_pos").drop(
        columns=["_row_pos", "_fwd_pkts", "_fwd_bytes", "_is_single", "_rf", "_rb", "_rs"]
    )
    return result


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
# 2B. Pure One-Way Forward Rate Derivation & Preprocessing Pipelines (v2.0)
# --------------------------------------------------------------------------

def add_pure_forward_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives pure forward transmission rates strictly from unidirectional forward metrics:
    - Fwd_Bytes_Per_Sec: Total Length of Fwd Packets / (Fwd IAT Total in seconds)
    - Fwd_Packets_Per_Sec: Total Fwd Packets / (Fwd IAT Total in seconds)

    Zero-duration and invalid handling:
    When Fwd IAT Total == 0 (single packet flows or instantaneous micro-bursts),
    or when inputs contain NaNs/infinities, rates are cleanly assigned 0.0 to
    prevent division by zero, overflow, or NaN propagation.
    """
    df = df.copy()
    if "Fwd IAT Total" not in df.columns:
        raise ValueError("Column 'Fwd IAT Total' required to compute pure forward rates.")

    fwd_iat = df["Fwd IAT Total"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    fwd_iat_sec = (fwd_iat / 1e6).to_numpy(dtype=float)

    fwd_bytes = df.get("Total Length of Fwd Packets", pd.Series(0, index=df.index))
    fwd_bytes = fwd_bytes.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(dtype=float)

    fwd_pkts = df.get("Total Fwd Packets", pd.Series(0, index=df.index))
    fwd_pkts = fwd_pkts.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(dtype=float)

    valid_mask = fwd_iat_sec > 0
    fwd_bytes_rate = np.zeros(len(df), dtype=float)
    fwd_pkts_rate = np.zeros(len(df), dtype=float)

    fwd_bytes_rate[valid_mask] = fwd_bytes[valid_mask] / fwd_iat_sec[valid_mask]
    fwd_pkts_rate[valid_mask] = fwd_pkts[valid_mask] / fwd_iat_sec[valid_mask]

    # Guard against numeric overflow or NaN
    fwd_bytes_rate = np.nan_to_num(fwd_bytes_rate, nan=0.0, posinf=0.0, neginf=0.0)
    fwd_pkts_rate = np.nan_to_num(fwd_pkts_rate, nan=0.0, posinf=0.0, neginf=0.0)

    df["Fwd_Bytes_Per_Sec"] = fwd_bytes_rate
    df["Fwd_Packets_Per_Sec"] = fwd_pkts_rate
    return df


PURE_LOG_FEATURES: List[str] = [
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Fwd Header Length",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd_Bytes_Per_Sec",
    "Fwd_Packets_Per_Sec",
]


def apply_pure_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Safely apply log1p to non-negative skewed pure forward features."""
    df = df.copy()
    for col in PURE_LOG_FEATURES:
        if col in df.columns:
            valid_vals = df[col].clip(lower=0)
            df[col] = np.log1p(valid_vals)
    return df


class PureOneWayPreprocessingPipeline(PreprocessingPipeline):
    """
    Leak-free feature preprocessing pipeline strictly for PURE_ONE_WAY_FEATURES (v2.0).
    Automatically augments forward rates before median imputation and standard scaling.
    """
    def __init__(self, features: Optional[List[str]] = None, imputer_strategy: str = "median"):
        selected = features if features is not None else PURE_ONE_WAY_FEATURES
        super().__init__(features=selected, imputer_strategy=imputer_strategy)
        self.imputer = SimpleImputer(strategy=imputer_strategy, keep_empty_features=True)
        self.version = "2.0-pure-oneway"

    def fit(self, train_df: pd.DataFrame) -> "PureOneWayPreprocessingPipeline":
        augmented = add_pure_forward_rates(train_df)
        cleaned_train = clean_feature_frame(augmented, features=self.features)
        imputed = self.imputer.fit_transform(cleaned_train)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, batch_size: int = 50_000) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PureOneWayPreprocessingPipeline must be fitted first.")
        augmented = add_pure_forward_rates(df)
        cleaned = clean_feature_frame(augmented, features=self.features)
        batches = []
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned.iloc[start : start + batch_size]
            imputed = self.imputer.transform(batch)
            scaled = self.scaler.transform(imputed)
            batches.append(scaled)
        return np.vstack(batches) if batches else np.empty((0, len(self.features)))


class PureOneWayLogPreprocessingPipeline(LogPreprocessingPipeline):
    """
    Preprocessing pipeline with log1p transformation strictly for PURE_ONE_WAY_FEATURES (v2.0).
    Applies log1p transformation to power-law skewed pure forward rates and durations.
    """
    def __init__(self, features: Optional[List[str]] = None, imputer_strategy: str = "median"):
        selected = features if features is not None else PURE_ONE_WAY_FEATURES
        super().__init__(features=selected, imputer_strategy=imputer_strategy)
        self.imputer = SimpleImputer(strategy=imputer_strategy, keep_empty_features=True)
        self.version = "2.0-pure-oneway"

    def fit(self, train_df: pd.DataFrame) -> "PureOneWayLogPreprocessingPipeline":
        augmented = add_pure_forward_rates(train_df)
        cleaned_train = clean_feature_frame(augmented, features=self.features)
        log_train = apply_pure_log_transform(cleaned_train)
        imputed = self.imputer.fit_transform(log_train)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, batch_size: int = 50_000) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PureOneWayLogPreprocessingPipeline must be fitted first.")
        augmented = add_pure_forward_rates(df)
        cleaned = clean_feature_frame(augmented, features=self.features)
        log_df = apply_pure_log_transform(cleaned)
        batches = []
        for start in range(0, len(log_df), batch_size):
            batch = log_df.iloc[start : start + batch_size]
            imputed = self.imputer.transform(batch)
            scaled = self.scaler.transform(imputed)
            batches.append(scaled)
        return np.vstack(batches) if batches else np.empty((0, len(self.features)))


def check_model_compatibility(model: Any, pipeline: Any, expected_features: List[str]) -> Tuple[bool, str]:
    """
    Validates model and pipeline compatibility with expected feature specification.
    Prevents silent bugs where modified feature counts (e.g., 26-dim) are fed into old models (e.g., 32-dim).
    """
    pipe_feats = getattr(pipeline, "features", [])
    model_in = getattr(model, "n_features_in_", None)

    if len(pipe_feats) != len(expected_features):
        return False, f"Feature count mismatch: pipeline has {len(pipe_feats)}, expected {len(expected_features)}"
    if list(pipe_feats) != list(expected_features):
        return False, f"Feature ordering or naming mismatch between pipeline and expected specification"
    if model_in is not None and model_in != len(expected_features):
        return False, f"Model expects {model_in} features, but specification has {len(expected_features)}"
    return True, "Compatible"


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

