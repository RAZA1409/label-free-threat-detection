"""
src/streaming_features.py
AEGIS Phase 3 Stage 4A & Stage 4B.1: Pure One-Way Causal Streaming Feature Engine & Safe Buffer Capacity.

SIH Problem ID: SIH26145
Compliance: 26-dimensional forward-feature-only design; reverse-direction feature information excluded.

Provides:
1. Exact Forward Rate Derivation:
   - Fwd_Bytes_Per_Sec: Total Length of Fwd Packets / (Fwd IAT Total in seconds)
   - Fwd_Packets_Per_Sec: Total Fwd Packets / (Fwd IAT Total in seconds)
   - Zero-duration and invalid sentinel handling with strict non-negative clamping.
2. Chronological Stream Preparation:
   - Stable deterministic sorting by (Timestamp, original_index).
   - Documented duplicate, missing, and late-arrival timestamp policies.
3. Causal Rolling Context Engine (StreamingCausalContext):
   - Sliding causal window: (t - 10s, t] closed on right.
   - Microsecond deque-based eviction of expired entries (t_k <= t - 10s).
   - Configurable memory safety ceiling (max_buffer_capacity, default: 10,000 entries).
   - Buffer overflow detection and explicit degraded state transition (is_degraded, is_valid).
   - Safe event-time recovery: t > last_overflow_timestamp + 10s.
   - Multi-batch continuity, simultaneous timestamp group finalization, and warm-up buffer.
4. High-Performance Streaming Feature Engine (StreamingFeatureEngine):
   - Single-flow microsecond feature extraction and fast NumPy standardization.
   - Vectorized batch feature extraction matching offline features.py bit-for-bit.
   - Explicit feature validity and context state signaling for downstream inference safety.
"""

import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from features import (
    PURE_ONE_WAY_FEATURES,
    PURE_ONE_WAY_BASE_FEATURES,
    PURE_ONE_WAY_TEMPORAL_FEATURES,
    PURE_LOG_FEATURES,
    NON_NEGATIVE_PHYSICAL_FEATURES,
    TCP_WINDOW_FEATURES,
    add_pure_forward_rates,
    clean_feature_frame,
    apply_pure_log_transform,
    PureOneWayLogPreprocessingPipeline,
)

STREAMING_WINDOW_SECONDS: int = 10
DEFAULT_EVENT_TIME_HORIZON: pd.Timestamp = pd.Timestamp("2030-01-01 00:00:00")
DEFAULT_MAX_FUTURE_DRIFT_SECONDS: float = 30.0


def _normalize_tz_comparison(
    ts_left: pd.Timestamp,
    ts_right: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Ensures two pd.Timestamp objects have compatible tz for comparison."""
    if ts_left.tz is None and ts_right.tz is not None:
        return ts_left, ts_right.tz_localize(None)
    elif ts_left.tz is not None and ts_right.tz is None:
        return ts_left.tz_localize(None), ts_right
    return ts_left, ts_right


def validate_timestamp_strict(
    ts: Any,
    clock_mode: str = "event_time",
    max_future_drift_seconds: Optional[float] = None,
    max_timestamp_horizon: Optional[Union[pd.Timestamp, str]] = None,
    current_time_fn: Optional[Any] = None,
) -> Tuple[bool, Optional[str], Optional[pd.Timestamp]]:
    """
    Strictly validates, parses, and bounds an incoming flow timestamp.

    Guarantees:
    - Rejects None: 'MISSING_TIMESTAMP'
    - Rejects booleans, NaN, Inf, non-numeric unparseable strings, NaT: 'INVALID_TIMESTAMP'
    - Rejects future drift / horizon exceedance: 'TIMESTAMP_FUTURE_DRIFT'
    - In 'wall_clock' mode: rejects timestamps > current_time + max_future_drift_seconds (default: 30s)
    - In 'event_time' mode: rejects timestamps > max_timestamp_horizon (default: 2030-01-01)
    - Handles timezone-aware and timezone-naive comparisons safely without raising TypeError.

    Returns:
        (is_valid: bool, reason: Optional[str], parsed_timestamp: Optional[pd.Timestamp])
    """
    if ts is None:
        return False, "MISSING_TIMESTAMP", None

    # In Python, isinstance(True, (int, float)) is True, so check bool explicitly first
    if isinstance(ts, bool):
        return False, "INVALID_TIMESTAMP", None

    if isinstance(ts, (float, np.number)):
        if math.isnan(ts) or math.isinf(ts):
            return False, "INVALID_TIMESTAMP", None

    if pd.isna(ts):
        return False, "INVALID_TIMESTAMP", None

    try:
        ts_pd = pd.Timestamp(ts)
        if pd.isna(ts_pd):
            return False, "INVALID_TIMESTAMP", None
    except Exception:
        return False, "INVALID_TIMESTAMP", None

    # Horizon and clock drift enforcement
    if clock_mode == "wall_clock":
        drift_sec = (
            float(max_future_drift_seconds)
            if max_future_drift_seconds is not None
            else DEFAULT_MAX_FUTURE_DRIFT_SECONDS
        )
        if current_time_fn is not None:
            wall_now = pd.Timestamp(current_time_fn())
        else:
            wall_now = pd.Timestamp.now(tz=ts_pd.tz)

        cmp_ts, cmp_wall = _normalize_tz_comparison(ts_pd, wall_now)
        wall_ceiling = cmp_wall + pd.Timedelta(seconds=drift_sec)
        if cmp_ts > wall_ceiling:
            return False, "TIMESTAMP_FUTURE_DRIFT", None

    elif clock_mode == "event_time":
        horizon = (
            pd.Timestamp(max_timestamp_horizon)
            if max_timestamp_horizon is not None
            else DEFAULT_EVENT_TIME_HORIZON
        )
        cmp_ts, cmp_horizon = _normalize_tz_comparison(ts_pd, horizon)
        if cmp_ts > cmp_horizon:
            return False, "TIMESTAMP_FUTURE_DRIFT", None
    else:
        raise ValueError(f"Unknown clock_mode '{clock_mode}'. Expected 'event_time' or 'wall_clock'.")

    return True, None, ts_pd


# --------------------------------------------------------------------------
# 1. Forward Rate Derivation (Scalar & Batch)
# --------------------------------------------------------------------------

def compute_pure_forward_rates_scalar(
    fwd_iat_total: Any,
    total_fwd_bytes: Any,
    total_fwd_pkts: Any,
) -> Tuple[float, float]:
    """
    Computes pure forward transmission rates for a single flow record.
    Reproduces the exact offline mathematical formula from add_pure_forward_rates():
        - Fwd_Bytes_Per_Sec  = total_fwd_bytes / (fwd_iat_total / 1e6)
        - Fwd_Packets_Per_Sec = total_fwd_pkts  / (fwd_iat_total / 1e6)

    Zero-Duration and Sentinel Handling:
    When fwd_iat_total <= 0 (e.g. single-packet flows, probe bursts, negative sentinels),
    or when inputs are None, NaN, or infinite, rates are strictly assigned 0.0.
    """
    try:
        iat = float(fwd_iat_total)
        if math.isnan(iat) or math.isinf(iat) or iat <= 0.0:
            return 0.0, 0.0
    except (TypeError, ValueError):
        return 0.0, 0.0

    try:
        b = float(total_fwd_bytes)
        if math.isnan(b) or math.isinf(b) or b < 0.0:
            b = 0.0
    except (TypeError, ValueError):
        b = 0.0

    try:
        p = float(total_fwd_pkts)
        if math.isnan(p) or math.isinf(p) or p < 0.0:
            p = 0.0
    except (TypeError, ValueError):
        p = 0.0

    iat_sec = iat / 1e6
    if iat_sec <= 0.0:
        return 0.0, 0.0

    bytes_rate = b / iat_sec
    pkts_rate = p / iat_sec

    if math.isnan(bytes_rate) or math.isinf(bytes_rate) or bytes_rate < 0.0:
        bytes_rate = 0.0
    if math.isnan(pkts_rate) or math.isinf(pkts_rate) or pkts_rate < 0.0:
        pkts_rate = 0.0

    return float(bytes_rate), float(pkts_rate)


def compute_pure_forward_rates_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized forward rate calculation across an entire DataFrame.
    Delegates to the canonical offline add_pure_forward_rates() implementation.
    """
    return add_pure_forward_rates(df)


# --------------------------------------------------------------------------
# 2. Chronological Stream Preparation & Timestamp Policy
# --------------------------------------------------------------------------

def find_contiguous_replay_window(
    sorted_df: pd.DataFrame,
    max_events: int,
    label_col: str = "Label",
    preferred_attack: Optional[str] = "DDoS",
    target_ratio: float = 0.25,
) -> Tuple[int, str]:
    """
    Identifies a deterministic, contiguous chronological window containing real
    attack flows and surrounding baseline/benign traffic for realistic temporal replay.

    Algorithm:
    1. Detects contiguous attack runs using vectorized diff.
    2. Evaluates candidate windows anchored around attack onsets.
       For bursts longer than max_events, deterministically leads in with pre-attack
       benign traffic (20% of max_events) to capture baseline-to-attack transition dynamics.
    3. Scores candidate windows based on preferred attack match, presence of both
       classes (attack and benign boundary), and natural proximity to target_ratio.
    4. Deterministically breaks score ties by selecting the earliest chronological occurrence.

    Returns:
        (start_idx, selection_note)
    """
    n_total = len(sorted_df)
    if n_total <= max_events:
        return 0, f"Dataset length ({n_total}) <= requested window ({max_events}); full dataset selected."

    # Identify label column
    target_col = None
    for cand in [label_col, "Clean_Label", "Label"]:
        if cand in sorted_df.columns:
            target_col = cand
            break

    if target_col is None:
        return 0, f"No label column found; defaulting to initial {max_events}-flow contiguous window."

    is_attack = ~sorted_df[target_col].astype(str).str.upper().str.contains("BENIGN")
    if not is_attack.any():
        return 0, f"No attack flows present in dataset; defaulting to initial {max_events}-flow contiguous window."

    # Contiguous attack runs via vectorized diff
    diff = np.diff(is_attack.to_numpy().astype(np.int8), prepend=0, append=0)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0] - 1

    if len(run_starts) == 0:
        return 0, f"No attack runs detected; defaulting to initial {max_events}-flow contiguous window."

    candidates = []
    for r_start, r_end in zip(run_starts, run_ends):
        burst_len = int(r_end - r_start + 1)
        run_label = str(sorted_df[target_col].iloc[r_start])

        # Deterministic anchor & lead-in policy
        if burst_len <= max_events:
            lead_in = min(r_start, max(1, (max_events - burst_len) // 2))
        else:
            lead_in = min(r_start, max(1, int(max_events * 0.20)))

        cand_start = max(0, r_start - lead_in)
        if cand_start + max_events > n_total:
            cand_start = max(0, n_total - max_events)

        n_atk = int(is_attack.iloc[cand_start : cand_start + max_events].sum())
        cand_ratio = n_atk / float(max_events)

        # Multi-criterion scoring:
        # Prioritize matching preferred attack (+1000), retaining both baseline & attack (+500),
        # authentic natural burst volume/density (+burst_len to favor genuine volumetric floods
        # over stray 1-flow trickles), and target ratio alignment (-ratio_penalty * 10).
        is_pref = 1.0 if (preferred_attack and preferred_attack.lower() in run_label.lower()) else 0.0
        has_both = 1.0 if (0 < n_atk < max_events) else 0.0
        ratio_penalty = abs(cand_ratio - float(target_ratio))

        score = (is_pref * 1000.0) + (has_both * 500.0) + (burst_len * 1.0) - (ratio_penalty * 10.0)
        candidates.append((score, cand_start, run_label, burst_len, n_atk, cand_ratio))

    # Highest score first, then earliest index
    candidates.sort(key=lambda c: (-c[0], c[1]))
    best = candidates[0]
    note = (
        f"Selected contiguous window at index {best[1]} containing {best[2]} attack "
        f"(natural burst: {best[3]} flows, window attacks: {best[4]}/{max_events} = {best[5]*100:.1f}%)"
    )
    return best[1], note


def prepare_chronological_stream(
    df: pd.DataFrame,
    max_events: Optional[int] = None,
    inject_attacks: bool = False,
    attack_ratio: float = 0.25,
    random_state: int = 42,
    preferred_attack: Optional[str] = "DDoS",
) -> pd.DataFrame:
    """
    Prepares a chronologically ordered streaming event pool from historical captures.

    Timestamp & Selection Policy:
    1. Valid timestamps: Parsed to datetime64[ns] and sorted monotonically.
    2. Duplicate timestamps: Preserved in stable original sequence order using
       '_orig_pos' as secondary sort key.
    3. Missing / Invalid timestamps (NaT / None): Rejected with ValueError.
    4. Deterministic contiguous window selection:
       - If max_events is requested, selects a deterministic CONTIGUOUS chronological
         window rather than sparse random sampling across multi-day captures.
       - If inject_attacks is True and label information exists, identifies a contiguous
         window containing a real attack burst (e.g. DDoS) and surrounding baseline traffic.
       - Guarantees authentic inter-arrival times, physical flow velocities, and temporal fidelity.
       - Never randomly drops intermediate records or synthesizes artificial timestamps.

    Metadata Attached to attrs:
    - min_timestamp: Earliest timestamp in stream.
    - max_timestamp: Latest timestamp in stream.
    - trusted_horizon: Replay safety horizon (max_ts + 10s).
    - selection_mode: "full_dataset", "contiguous_attack_burst", or "contiguous_head".
    - window_start_idx: Start offset in the sorted capture.
    - window_end_idx: End offset in the sorted capture.
    - selection_note: Diagnostic explanation of selected window.
    """
    if "Timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'Timestamp' column for chronological streaming.")

    pool = df.copy()

    # Reject null timestamps
    if pool["Timestamp"].isna().any():
        n_missing = int(pool["Timestamp"].isna().sum())
        raise ValueError(
            f"Chronological stream rejected: {n_missing} events have missing or invalid timestamps. "
            "Silent timestamp imputation is prohibited by AEGIS streaming policy."
        )

    if not pd.api.types.is_datetime64_any_dtype(pool["Timestamp"]):
        pool["Timestamp"] = pd.to_datetime(pool["Timestamp"])

    # Retain original index position for stable tie-breaking
    pool["_orig_pos"] = np.arange(len(pool))

    # Strictly stable chronological sort across entire input pool first
    sorted_df = pool.sort_values(by=["Timestamp", "_orig_pos"], kind="stable").reset_index(drop=True)
    n_total = len(sorted_df)

    # Window selection policy
    if max_events is not None and max_events > 0 and n_total > max_events:
        label_col = "Label" if "Label" in sorted_df.columns else ("Clean_Label" if "Clean_Label" in sorted_df.columns else None)
        has_attacks = False
        if label_col is not None:
            has_attacks = bool((~sorted_df[label_col].astype(str).str.upper().str.contains("BENIGN")).any())

        if inject_attacks and label_col is not None and has_attacks:
            start_idx, note = find_contiguous_replay_window(
                sorted_df,
                max_events=max_events,
                label_col=label_col,
                preferred_attack=preferred_attack,
                target_ratio=attack_ratio,
            )
            mode = "contiguous_attack_burst"
        else:
            start_idx = 0
            mode = "contiguous_head"
            note = f"Contiguous window of {max_events} flows from start of chronological capture"

        end_idx = start_idx + max_events
        result_df = sorted_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        w_start, w_end = start_idx, end_idx
    else:
        result_df = sorted_df
        mode = "full_dataset"
        w_start, w_end = 0, n_total
        note = f"Full chronological dataset selected ({n_total} flows)"

    result_df = result_df.drop(columns=["_orig_pos"])

    # Metadata attachment for trusted replay bounds
    if len(result_df) > 0:
        max_ts = result_df["Timestamp"].max()
        min_ts = result_df["Timestamp"].min()
        result_df.attrs["max_timestamp"] = max_ts
        result_df.attrs["min_timestamp"] = min_ts
        result_df.attrs["trusted_horizon"] = max_ts + pd.Timedelta(seconds=10.0)
        result_df.attrs["selection_mode"] = mode
        result_df.attrs["window_start_idx"] = w_start
        result_df.attrs["window_end_idx"] = w_end
        result_df.attrs["selection_note"] = note

    return result_df


# --------------------------------------------------------------------------
# 2.5 Input Sanitization & Malformed-Flow Quarantine (Stage 4B.3A & 4B.3B)
# --------------------------------------------------------------------------

def validate_and_sanitize_flow_record(
    record: Any,
    clock_mode: str = "event_time",
    max_future_drift_seconds: Optional[float] = None,
    max_timestamp_horizon: Optional[Union[pd.Timestamp, str]] = None,
    current_time_fn: Optional[Any] = None,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates and sanitizes a raw network flow record before ingestion into
    the causal streaming feature context.

    Rejection Policies:
    - Non-mapping types: Rejection reason 'INVALID_RECORD_TYPE'.
    - Missing or null timestamp: Rejection reason 'MISSING_TIMESTAMP'.
    - Invalid, NaT, or unparseable timestamp: Rejection reason 'INVALID_TIMESTAMP'.
    - Future drift or horizon exceedance: Rejection reason 'TIMESTAMP_FUTURE_DRIFT'.
    - Missing packet or byte counters: Rejection reason 'MISSING_REQUIRED_FIELD'.
    - Non-numeric or unparseable counter: Rejection reason 'NON_NUMERIC_FEATURE'.
    - NaN counter: Rejection reason 'INVALID_COUNTER'.
    - Infinite counter (+inf, -inf): Rejection reason 'INFINITE_COUNTER'.
    - Negative packet or byte counter: Rejection reason 'NEGATIVE_COUNTER'.
      CRITICAL: Strictly rejects without silently clamping negative counters to zero!
    - Non-numeric corrupted strings in features: Rejection reason 'NON_NUMERIC_FEATURE'.
    - Unexpected nested data structures (list, dict, set, tuple): Rejection reason 'UNEXPECTED_DATA_TYPE'.

    Sanitization Guarantees:
    - Coerces parseable numeric string values (e.g. "123.45") to float.
    - Replaces infinite values in non-counter features with np.nan for pipeline imputation.
    - Preserves metadata fields (Timestamp, Label, Clean_Label, etc.).

    Returns:
        (is_valid: bool, rejection_reason: Optional[str], sanitized_record: Optional[Dict[str, Any]])
    """
    if not isinstance(record, (dict, pd.Series)):
        return False, "INVALID_RECORD_TYPE", None

    rec: Dict[str, Any] = record.to_dict() if isinstance(record, pd.Series) else dict(record)

    # 1. Timestamp validation via strict parser and horizon boundary check
    raw_ts = rec.get("Timestamp")
    is_ts_valid, ts_reason, ts_pd = validate_timestamp_strict(
        raw_ts,
        clock_mode=clock_mode,
        max_future_drift_seconds=max_future_drift_seconds,
        max_timestamp_horizon=max_timestamp_horizon,
        current_time_fn=current_time_fn,
    )
    if not is_ts_valid:
        return False, ts_reason, None

    # 2. Required packet and byte counters
    for counter_field in ["Total Fwd Packets", "Total Length of Fwd Packets"]:
        if counter_field not in rec or rec[counter_field] is None:
            return False, "MISSING_REQUIRED_FIELD", None

        raw_val = rec[counter_field]
        if isinstance(raw_val, bool) or isinstance(raw_val, (list, dict, set, tuple)):
            return False, "UNEXPECTED_DATA_TYPE", None

        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            return False, "NON_NUMERIC_FEATURE", None

        if math.isnan(val):
            return False, "INVALID_COUNTER", None
        if math.isinf(val):
            return False, "INFINITE_COUNTER", None
        if val < 0.0:
            return False, "NEGATIVE_COUNTER", None

        rec[counter_field] = val

    rec["Timestamp"] = ts_pd

    # 3. Numeric feature validation and sanitization for other features
    non_numeric_fields = {
        "Timestamp", "Label", "Clean_Label", "Flow ID", "Source IP",
        "Destination IP", "_orig_pos", "_is_valid", "_context_state",
        "_overflow_events", "original_index", "flow_id", "protocol",
    }

    for k, v in list(rec.items()):
        if k in non_numeric_fields:
            continue

        if isinstance(v, bool) or isinstance(v, (list, dict, set, tuple)):
            return False, "UNEXPECTED_DATA_TYPE", None

        if isinstance(v, str):
            try:
                parsed_val = float(v)
                if math.isinf(parsed_val):
                    rec[k] = np.nan
                else:
                    rec[k] = parsed_val
            except (ValueError, TypeError):
                return False, "NON_NUMERIC_FEATURE", None
        elif isinstance(v, (int, float, np.number)):
            val = float(v)
            if math.isinf(val):
                rec[k] = np.nan
            else:
                rec[k] = val
        elif v is None:
            rec[k] = np.nan

    return True, None, rec


class FlowQuarantineManager:
    """
    Manages quarantine metrics, rejection reason categorization, and diagnostic storage
    for malformed network flow records rejected by AEGIS input sanitization.
    """

    def __init__(self, max_diagnostic_samples: int = 100):
        self.max_diagnostic_samples: int = int(max_diagnostic_samples)
        self.total_quarantined_records: int = 0
        self.quarantine_reason_counts: Dict[str, int] = {}
        self.invalid_timestamp_count: int = 0
        self.invalid_numeric_feature_count: int = 0
        self.invalid_packet_byte_counter_count: int = 0
        self.missing_required_field_count: int = 0
        self.recent_quarantined_samples: deque = deque(maxlen=self.max_diagnostic_samples)

    def quarantine(self, record: Any, reason: str) -> None:
        """Records a quarantined flow record and increments diagnostic metrics."""
        self.total_quarantined_records += 1
        self.quarantine_reason_counts[reason] = self.quarantine_reason_counts.get(reason, 0) + 1

        # Categorize
        if "TIMESTAMP" in reason:
            self.invalid_timestamp_count += 1
        elif "COUNTER" in reason:
            self.invalid_packet_byte_counter_count += 1
        elif "FEATURE" in reason or "TYPE" in reason:
            self.invalid_numeric_feature_count += 1
        else:
            self.missing_required_field_count += 1

        # Store diagnostic sample in bounded ring buffer
        ts_str = None
        if isinstance(record, dict):
            ts_str = str(record.get("Timestamp"))
            rec_summary = {k: str(v) for k, v in list(record.items())[:10]}
        elif isinstance(record, pd.Series):
            ts_str = str(record.get("Timestamp"))
            rec_summary = {k: str(v) for k, v in list(record.to_dict().items())[:10]}
        else:
            rec_summary = str(record)[:200]

        self.recent_quarantined_samples.append({
            "reason": reason,
            "timestamp": ts_str,
            "record_summary": rec_summary,
        })

    def get_metrics(self) -> Dict[str, Any]:
        """Returns diagnostic metrics and breakdown of quarantine reasons."""
        return {
            "total_quarantined_records": self.total_quarantined_records,
            "quarantine_reason_counts": dict(self.quarantine_reason_counts),
            "invalid_timestamp_count": self.invalid_timestamp_count,
            "invalid_numeric_feature_count": self.invalid_numeric_feature_count,
            "invalid_packet_byte_counter_count": self.invalid_packet_byte_counter_count,
            "missing_required_field_count": self.missing_required_field_count,
            "recent_quarantined_samples_count": len(self.recent_quarantined_samples),
        }

    def reset(self) -> None:
        """Resets all quarantine metrics and clears diagnostic sample storage."""
        self.total_quarantined_records = 0
        self.quarantine_reason_counts.clear()
        self.invalid_timestamp_count = 0
        self.invalid_numeric_feature_count = 0
        self.invalid_packet_byte_counter_count = 0
        self.missing_required_field_count = 0
        self.recent_quarantined_samples.clear()


# --------------------------------------------------------------------------
# 3. Causal Rolling Context Engine
# --------------------------------------------------------------------------

class StreamingCausalContext:
    """
    Maintains a sliding causal window (t - window_seconds, t] over network flows.

    Window Semantics:
    - Strictly open on left: (t - window_seconds
    - Strictly closed on right: , t]
    - An event at exactly (t - window_seconds) is excluded.
    - An event at exactly t is included.

    Duplicate Timestamp Handling:
    - When multiple events arrive with the exact same timestamp t, they represent
      simultaneous flows occurring within the same time epoch.
      In 'simultaneous' mode, all events sharing timestamp t receive the complete
      aggregate accumulated through all events of timestamp t, matching offline
      transform("last") bit-for-bit.
    - In 'sequential' mode, each event progressively receives the cumulative
      aggregate up to its individual arrival.

    Cross-Batch Continuity:
    - The buffer persists across batch invocations.
    - When flush=False, an open timestamp group spanning batch boundaries is
      retained until the timestamp strictly advances, guaranteeing 100% numerical
      parity across arbitrary chunk boundaries.
    """

    def __init__(
        self,
        window_seconds: int = STREAMING_WINDOW_SECONDS,
        max_late_seconds: float = 10.0,
        max_buffer_capacity: int = 10_000,
        clock_mode: str = "event_time",
        max_future_drift_seconds: Optional[float] = None,
        max_timestamp_horizon: Optional[Union[pd.Timestamp, str]] = None,
        current_time_fn: Optional[Any] = None,
        quarantine_manager: Optional[FlowQuarantineManager] = None,
    ):
        self.window_seconds: int = int(window_seconds)
        self.window_delta: pd.Timedelta = pd.Timedelta(seconds=self.window_seconds)
        self.max_late_delta: pd.Timedelta = pd.Timedelta(seconds=float(max_late_seconds))
        self.max_buffer_capacity: int = int(max_buffer_capacity)
        self.clock_mode: str = str(clock_mode)
        self.max_future_drift_seconds: Optional[float] = (
            float(max_future_drift_seconds) if max_future_drift_seconds is not None else None
        )
        self.max_timestamp_horizon: pd.Timestamp = (
            pd.Timestamp(max_timestamp_horizon)
            if max_timestamp_horizon is not None
            else DEFAULT_EVENT_TIME_HORIZON
        )
        self.current_time_fn: Optional[Any] = current_time_fn
        self.quarantine_manager: FlowQuarantineManager = (
            quarantine_manager if quarantine_manager is not None else FlowQuarantineManager()
        )

        # Deque of (timestamp: pd.Timestamp, fwd_pkts: float, fwd_bytes: float, is_single: float)
        self.buffer: deque = deque()

        # Watermark tracking
        self.latest_timestamp: Optional[pd.Timestamp] = None
        self.total_processed: int = 0
        self.total_late_events: int = 0

        # Memory safety and degradation tracking
        self.is_degraded: bool = False
        self.last_overflow_timestamp: Optional[pd.Timestamp] = None
        self.total_dropped_overflow_events: int = 0

        # Pending group buffer for multi-batch simultaneous group finalization
        # List of (row_idx, timestamp_pd, fwd_pkts, fwd_bytes, is_single, raw_dict)
        self.pending_group: List[Tuple[int, pd.Timestamp, float, float, float, Dict[str, Any]]] = []

    @property
    def is_valid(self) -> bool:
        """Returns True if the causal window is complete and uncompromised by buffer overflow."""
        return not self.is_degraded

    @property
    def context_state(self) -> str:
        """Returns 'HEALTHY' if buffer has full rolling integrity, or 'DEGRADED' if capacity overflowed."""
        return "DEGRADED" if self.is_degraded else "HEALTHY"

    def reset(self) -> None:
        """Clears all historical buffer state and degradation status."""
        self.buffer.clear()
        self.pending_group.clear()
        self.latest_timestamp = None
        self.total_processed = 0
        self.total_late_events = 0
        self.is_degraded = False
        self.last_overflow_timestamp = None
        self.total_dropped_overflow_events = 0
        self.quarantine_manager.reset()

    def warm_up(self, historical_df: pd.DataFrame) -> int:
        """
        Pre-loads historical events into the sliding context buffer.
        Eliminates startup cold-start artifacts by populating events within
        window_seconds of the historical tail.

        Rejection & Horizon Safeguards:
        - Drops null, NaT, and unparseable timestamps.
        - Discards timestamps exceeding max_timestamp_horizon.

        Returns number of events buffered.
        """
        if historical_df is None or len(historical_df) == 0:
            return 0

        hist = historical_df.copy()
        if "Timestamp" not in hist.columns:
            raise ValueError("Historical context must contain a 'Timestamp' column.")

        if not pd.api.types.is_datetime64_any_dtype(hist["Timestamp"]):
            hist["Timestamp"] = pd.to_datetime(hist["Timestamp"], errors="coerce")

        # Filter out NaT and invalid timestamps
        hist = hist.dropna(subset=["Timestamp"]).copy()
        if len(hist) == 0:
            return 0

        # Filter out timestamps beyond max_timestamp_horizon
        if self.max_timestamp_horizon is not None:
            first_ts = hist["Timestamp"].iloc[0]
            _, cmp_horizon = _normalize_tz_comparison(first_ts, self.max_timestamp_horizon)
            if hist["Timestamp"].dt.tz is not None and cmp_horizon.tz is None:
                hist = hist[hist["Timestamp"].dt.tz_localize(None) <= cmp_horizon].copy()
            elif hist["Timestamp"].dt.tz is None and cmp_horizon.tz is not None:
                hist = hist[hist["Timestamp"] <= cmp_horizon.tz_localize(None)].copy()
            else:
                hist = hist[hist["Timestamp"] <= cmp_horizon].copy()

        if len(hist) == 0:
            return 0

        hist = hist.sort_values("Timestamp", kind="stable")
        max_ts = hist["Timestamp"].max()

        if self.latest_timestamp is not None and self.latest_timestamp > max_ts:
            effective_ts = self.latest_timestamp
        else:
            effective_ts = max_ts
            self.latest_timestamp = max_ts

        cutoff = effective_ts - self.window_delta

        # Evict expired entries from existing buffer first
        self._evict_expired(effective_ts)

        # Only retain tail within window_seconds
        tail = hist[hist["Timestamp"] > cutoff].copy()
        if len(tail) == 0:
            self._enforce_capacity(effective_ts)
            return len(self.buffer)

        # If tail alone exceeds capacity, truncate tail to capacity first
        if len(tail) > self.max_buffer_capacity:
            excess = len(tail) - self.max_buffer_capacity
            tail = tail.iloc[excess:].copy()
            self.total_dropped_overflow_events += excess
            self.last_overflow_timestamp = effective_ts
            self.is_degraded = True

        pkts = tail.get("Total Fwd Packets", pd.Series(0, index=tail.index)).replace([np.inf, -np.inf], np.nan).fillna(0).values
        bytes_ = tail.get("Total Length of Fwd Packets", pd.Series(0, index=tail.index)).replace([np.inf, -np.inf], np.nan).fillna(0).values
        singles = (pkts <= 1).astype(float)
        ts_vals = tail["Timestamp"].values

        for i in range(len(tail)):
            ts_pd = pd.Timestamp(ts_vals[i])
            self.buffer.append((ts_pd, float(pkts[i]), float(bytes_[i]), float(singles[i])))

        # Ensure chronological ordering if buffer already had entries
        if len(self.buffer) > len(tail):
            sorted_buf = sorted(self.buffer, key=lambda x: x[0])
            self.buffer = deque(sorted_buf)

        # Enforce capacity across all retained structures
        self._enforce_capacity(effective_ts)

        return len(self.buffer)

    def _check_recovery(self, current_ts: pd.Timestamp) -> None:
        """
        Evaluates event-time safe recovery from degraded buffer state.
        Recovery condition:
        The context can only clear degraded status when the current event timestamp
        strictly exceeds last_overflow_timestamp + window_delta.
        At that point, the entire 10-second causal window consists purely of
        uncompromised post-overflow events.
        """
        if self.is_degraded and self.last_overflow_timestamp is not None:
            if current_ts > self.last_overflow_timestamp + self.window_delta:
                self.is_degraded = False

    def _evict_expired(self, current_ts: pd.Timestamp) -> None:
        """Evicts buffer entries strictly <= (current_ts - window_delta)."""
        cutoff = current_ts - self.window_delta
        while self.buffer and self.buffer[0][0] <= cutoff:
            self.buffer.popleft()

    def _enforce_capacity(
        self,
        current_ts: pd.Timestamp,
        emitted_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Enforces max_buffer_capacity across all retained internal memory structures.
        Total retained entries = len(self.buffer) + len(self.pending_group).

        When capacity is exceeded:
        1. Evicts oldest entries from self.buffer (FIFO).
        2. If self.buffer is empty and pending_group still exceeds capacity, evicts
           oldest entries from pending_group.
        3. Increments total_dropped_overflow_events for each discarded in-window event.
        4. Records last_overflow_timestamp = current_ts.
        5. Sets is_degraded = True.
        """
        total_retained = len(self.buffer) + len(self.pending_group)
        excess = total_retained - self.max_buffer_capacity
        if excess <= 0:
            return

        # Buffer capacity exceeded -> enter degraded state
        self.is_degraded = True
        self.last_overflow_timestamp = current_ts

        # 1. Evict from self.buffer first
        if self.buffer:
            evict_from_buffer = min(excess, len(self.buffer))
            for _ in range(evict_from_buffer):
                self.buffer.popleft()
            self.total_dropped_overflow_events += evict_from_buffer
            excess -= evict_from_buffer

        # 2. If excess remains, evict from self.pending_group
        if excess > 0 and self.pending_group:
            evict_from_pending = min(excess, len(self.pending_group))
            for k in range(evict_from_pending):
                item = self.pending_group[k]
                if emitted_rows is not None:
                    row_dict = dict(item[5])
                    row_dict["Rolling_Flow_Velocity_10s"] = float(len(self.buffer) + len(self.pending_group))
                    row_dict["Rolling_Bytes_Velocity_10s"] = float(
                        sum(b[2] for b in self.buffer) + sum(p[3] for p in self.pending_group)
                    )
                    row_dict["Rolling_Single_Pkt_Ratio_10s"] = 0.0
                    row_dict["_is_valid"] = False
                    row_dict["_context_state"] = "DEGRADED"
                    row_dict["_overflow_events"] = self.total_dropped_overflow_events + k + 1
                    emitted_rows.append(row_dict)
            self.pending_group = self.pending_group[evict_from_pending:]
            self.total_dropped_overflow_events += evict_from_pending
            excess -= evict_from_pending

    def update(
        self,
        timestamp: Any,
        fwd_pkts: float,
        fwd_bytes: float,
    ) -> Tuple[float, float, float]:
        """
        Single-event incremental update in sequential mode.

        Policy:
        - Strict timestamp validation: rejects None, NaT, NaN, Inf, and future drift.
          If rejected, quarantines flow and returns isolated metrics (1.0, fwd_bytes, single_ratio)
          without mutating self.buffer or self.latest_timestamp.
        - Monotonic watermark update: self.latest_timestamp = max(self.latest_timestamp, ts_pd).
        - If timestamp is later or equal to latest_timestamp: evicts entries older
          than (t - window_seconds), adds event, and returns current window metrics.
        - If timestamp is earlier than latest_timestamp:
          - In-window late (latest_timestamp - max_late_delta < t < latest_timestamp):
            inserted into sorted position, evicts relative to latest_timestamp, returns metric.
          - Expired late (t <= latest_timestamp - max_late_delta): increments late counter,
            returns isolated single-event metrics without corrupting streaming buffer.
        """
        is_ts_ok, ts_reason, ts_pd = validate_timestamp_strict(
            timestamp,
            clock_mode=self.clock_mode,
            max_future_drift_seconds=self.max_future_drift_seconds,
            max_timestamp_horizon=self.max_timestamp_horizon,
            current_time_fn=self.current_time_fn,
        )
        p = float(fwd_pkts) if not math.isnan(fwd_pkts) and not math.isinf(fwd_pkts) and fwd_pkts >= 0 else 0.0
        b = float(fwd_bytes) if not math.isnan(fwd_bytes) and not math.isinf(fwd_bytes) and fwd_bytes >= 0 else 0.0
        s = 1.0 if p <= 1.0 else 0.0

        if not is_ts_ok:
            rec_diag = {
                "Timestamp": timestamp,
                "Total Fwd Packets": fwd_pkts,
                "Total Length of Fwd Packets": fwd_bytes,
            }
            self.quarantine_manager.quarantine(rec_diag, ts_reason)
            return 1.0, b, s

        # Safe event-time recovery check
        self._check_recovery(ts_pd)

        if self.latest_timestamp is not None and ts_pd < self.latest_timestamp:
            # Late arrival check
            if ts_pd <= (self.latest_timestamp - self.max_late_delta):
                self.total_late_events += 1
                self.total_processed += 1
                # Expired event: evaluated in isolation, never enters buffer
                return 1.0, b, s
            else:
                self.total_late_events += 1
                # In-window out-of-order: insert into chronological position
                self.buffer.append((ts_pd, p, b, s))
                sorted_buf = sorted(self.buffer, key=lambda x: x[0])
                self.buffer = deque(sorted_buf)
                # Watermark is NOT moved backward
        else:
            # Monotonic advance: W = max(W, ts_pd)
            if self.latest_timestamp is None or ts_pd > self.latest_timestamp:
                self.latest_timestamp = ts_pd
            self.buffer.append((ts_pd, p, b, s))

        # Evict expired flows relative to latest watermark
        self._evict_expired(self.latest_timestamp)
        self._enforce_capacity(self.latest_timestamp)
        self.total_processed += 1

        cnt = float(len(self.buffer))
        tot_bytes = float(sum(item[2] for item in self.buffer))
        ratio = float(sum(item[3] for item in self.buffer) / cnt if cnt > 0 else 0.0)

        return cnt, tot_bytes, ratio

    def process_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single flow record dictionary, augmenting it with pure forward rates
        and rolling temporal velocities.
        Returns a dictionary containing all 26 PURE_ONE_WAY_FEATURES plus metadata.
        """
        # 1. Validation & Quarantine
        is_val, reason, sanitized = validate_and_sanitize_flow_record(
            flow,
            clock_mode=self.clock_mode,
            max_future_drift_seconds=self.max_future_drift_seconds,
            max_timestamp_horizon=self.max_timestamp_horizon,
            current_time_fn=self.current_time_fn,
        )
        if not is_val:
            self.quarantine_manager.quarantine(flow, reason)
            out = dict(flow)
            out["Fwd_Bytes_Per_Sec"] = 0.0
            out["Fwd_Packets_Per_Sec"] = 0.0
            out["Rolling_Flow_Velocity_10s"] = float(len(self.buffer))
            out["Rolling_Bytes_Velocity_10s"] = float(sum(b[2] for b in self.buffer))
            cnt = float(len(self.buffer))
            out["Rolling_Single_Pkt_Ratio_10s"] = float(sum(b[3] for b in self.buffer) / cnt if cnt > 0 else 0.0)
            out["_is_valid"] = False
            out["_context_state"] = "QUARANTINED"
            out["_quarantine_reason"] = reason
            out["_overflow_events"] = self.total_dropped_overflow_events
            return out

        out = dict(sanitized)

        # 2. Forward rates
        iat = sanitized.get("Fwd IAT Total", 0.0)
        bytes_ = float(sanitized["Total Length of Fwd Packets"])
        pkts = float(sanitized["Total Fwd Packets"])
        b_rate, p_rate = compute_pure_forward_rates_scalar(iat, bytes_, pkts)
        out["Fwd_Bytes_Per_Sec"] = b_rate
        out["Fwd_Packets_Per_Sec"] = p_rate

        # 3. Temporal velocities
        ts = sanitized["Timestamp"]
        rf, rb, rs = self.update(ts, pkts, bytes_)
        out["Rolling_Flow_Velocity_10s"] = rf
        out["Rolling_Bytes_Velocity_10s"] = rb
        out["Rolling_Single_Pkt_Ratio_10s"] = rs

        # 4. Buffer health metadata
        out["_is_valid"] = self.is_valid
        out["_context_state"] = self.context_state
        out["_overflow_events"] = self.total_dropped_overflow_events

        return out

    def _handle_late_simultaneous_event(
        self,
        orig_idx: int,
        ts: pd.Timestamp,
        p: float,
        b: float,
        s: float,
        row_dict: Dict[str, Any],
        watermark: pd.Timestamp,
        emitted_rows: List[Dict[str, Any]],
    ) -> None:
        """
        Handles an out-of-order late arrival during simultaneous batch processing.
        Does NOT alter active pending_group.
        """
        self.total_late_events += 1
        self.total_processed += 1
        out_dict = dict(row_dict)

        if ts <= (watermark - self.max_late_delta):
            # Expired late event: strictly outside max_late_delta window
            out_dict["Rolling_Flow_Velocity_10s"] = 1.0
            out_dict["Rolling_Bytes_Velocity_10s"] = b
            out_dict["Rolling_Single_Pkt_Ratio_10s"] = s
        else:
            # In-window late arrival: insert into buffer in chronological sorted position
            self.buffer.append((ts, p, b, s))
            sorted_buf = sorted(self.buffer, key=lambda x: x[0])
            self.buffer = deque(sorted_buf)
            # Enforce capacity
            self._enforce_capacity(watermark)
            cnt = float(len(self.buffer))
            tot_bytes = float(sum(item[2] for item in self.buffer))
            ratio = float(sum(item[3] for item in self.buffer) / cnt if cnt > 0 else 0.0)
            out_dict["Rolling_Flow_Velocity_10s"] = cnt
            out_dict["Rolling_Bytes_Velocity_10s"] = tot_bytes
            out_dict["Rolling_Single_Pkt_Ratio_10s"] = ratio

        out_dict["_is_valid"] = self.is_valid
        out_dict["_context_state"] = self.context_state
        out_dict["_overflow_events"] = self.total_dropped_overflow_events
        emitted_rows.append(out_dict)

    def process_batch(
        self,
        df: pd.DataFrame,
        causal_mode: str = "simultaneous",
        flush: bool = True,
    ) -> pd.DataFrame:
        """
        Processes a DataFrame of flows with exact offline feature parity.

        causal_mode:
            - 'simultaneous' (default): Groups events by timestamp. All events sharing
              timestamp t receive the full aggregate accumulated through the last event
              of timestamp t, exactly reproducing offline transform("last").
            - 'sequential': Processes events strictly one-by-one in DataFrame order.

        flush:
            - True (default): Immediately finalizes and emits all events in the batch.
            - False: Retains the open timestamp group across batch boundaries, ensuring
              bit-exact multi-batch parity even if a batch cut splits identical timestamps.
        """
        if len(df) == 0:
            res = df.copy()
            res["Fwd_Bytes_Per_Sec"] = pd.Series(dtype=float)
            res["Fwd_Packets_Per_Sec"] = pd.Series(dtype=float)
            res["Rolling_Flow_Velocity_10s"] = pd.Series(dtype=float)
            res["Rolling_Bytes_Velocity_10s"] = pd.Series(dtype=float)
            res["Rolling_Single_Pkt_Ratio_10s"] = pd.Series(dtype=float)
            res["_is_valid"] = pd.Series(dtype=bool)
            res["_context_state"] = pd.Series(dtype=str)
            res["_overflow_events"] = pd.Series(dtype=int)
            return res

        df = df.copy()
        if "Timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'Timestamp' column.")

        if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        # Compute forward rates via canonical vectorized function if Fwd IAT Total is present
        if "Fwd IAT Total" in df.columns:
            df = add_pure_forward_rates(df)
        elif "Fwd_Bytes_Per_Sec" not in df.columns:
            df["Fwd_Bytes_Per_Sec"] = 0.0
            df["Fwd_Packets_Per_Sec"] = 0.0

        n = len(df)

        if causal_mode == "sequential":
            emitted_rows: List[Dict[str, Any]] = []

            for i in range(n):
                row_dict = df.iloc[i].to_dict()
                is_val, reason, sanitized_row = validate_and_sanitize_flow_record(
                    row_dict,
                    clock_mode=self.clock_mode,
                    max_future_drift_seconds=self.max_future_drift_seconds,
                    max_timestamp_horizon=self.max_timestamp_horizon,
                    current_time_fn=self.current_time_fn,
                )
                if not is_val:
                    self.quarantine_manager.quarantine(row_dict, reason)
                    continue

                curr_ts_pd = sanitized_row["Timestamp"]
                p = float(sanitized_row["Total Fwd Packets"])
                b = float(sanitized_row["Total Length of Fwd Packets"])

                r_flow, r_bytes, r_single = self.update(curr_ts_pd, p, b)
                res_dict = dict(sanitized_row)
                res_dict["Rolling_Flow_Velocity_10s"] = r_flow
                res_dict["Rolling_Bytes_Velocity_10s"] = r_bytes
                res_dict["Rolling_Single_Pkt_Ratio_10s"] = r_single
                res_dict["_is_valid"] = self.is_valid
                res_dict["_context_state"] = self.context_state
                res_dict["_overflow_events"] = self.total_dropped_overflow_events
                emitted_rows.append(res_dict)

            if not emitted_rows:
                cols = list(df.columns)
                for col in [
                    "Rolling_Flow_Velocity_10s",
                    "Rolling_Bytes_Velocity_10s",
                    "Rolling_Single_Pkt_Ratio_10s",
                    "_is_valid",
                    "_context_state",
                    "_overflow_events",
                ]:
                    if col not in cols:
                        cols.append(col)
                return pd.DataFrame(columns=cols)

            return pd.DataFrame(emitted_rows)

        elif causal_mode == "simultaneous":
            emitted_rows: List[Dict[str, Any]] = []

            # 1. Validate and quarantine malformed rows before any state mutation
            valid_entries: List[Tuple[int, pd.Timestamp, float, float, float, Dict[str, Any]]] = []
            for i in range(n):
                row_dict = df.iloc[i].to_dict()
                is_val, reason, sanitized_row = validate_and_sanitize_flow_record(
                    row_dict,
                    clock_mode=self.clock_mode,
                    max_future_drift_seconds=self.max_future_drift_seconds,
                    max_timestamp_horizon=self.max_timestamp_horizon,
                    current_time_fn=self.current_time_fn,
                )
                if not is_val:
                    self.quarantine_manager.quarantine(row_dict, reason)
                    continue

                curr_ts_pd = sanitized_row["Timestamp"]
                p = float(sanitized_row["Total Fwd Packets"])
                b = float(sanitized_row["Total Length of Fwd Packets"])
                s = 1.0 if p <= 1.0 else 0.0
                valid_entries.append((i, curr_ts_pd, p, b, s, sanitized_row))

            if not valid_entries and not self.pending_group:
                cols = list(df.columns)
                for col in [
                    "Rolling_Flow_Velocity_10s",
                    "Rolling_Bytes_Velocity_10s",
                    "Rolling_Single_Pkt_Ratio_10s",
                    "_is_valid",
                    "_context_state",
                    "_overflow_events",
                ]:
                    if col not in cols:
                        cols.append(col)
                return pd.DataFrame(columns=cols)

            # 2. Stably pre-sort valid entries by (curr_ts_pd, orig_idx) to guarantee chronological ordering
            is_sorted = all(valid_entries[k][1] <= valid_entries[k + 1][1] for k in range(len(valid_entries) - 1))
            if not is_sorted:
                valid_entries = sorted(valid_entries, key=lambda x: (x[1], x[0]))

            # 3. Process entries through simultaneous state machine
            for orig_idx, curr_ts_pd, p, b, s, sanitized_row in valid_entries:
                watermark = self.latest_timestamp
                if self.pending_group:
                    pend_ts = self.pending_group[0][1]
                    if watermark is None or pend_ts > watermark:
                        watermark = pend_ts

                if self.pending_group:
                    active_ts = self.pending_group[0][1]
                    if curr_ts_pd > active_ts:
                        # Timestamp strictly advanced: finalize previous group
                        self._finalize_pending(emitted_rows)
                        self._evict_expired(curr_ts_pd)
                        self._check_recovery(curr_ts_pd)
                        self.pending_group.append((orig_idx, curr_ts_pd, p, b, s, sanitized_row))
                        self._enforce_capacity(curr_ts_pd, emitted_rows=emitted_rows)
                    elif curr_ts_pd == active_ts:
                        # Simultaneous flow: accumulate in active pending group
                        self._evict_expired(curr_ts_pd)
                        self._check_recovery(curr_ts_pd)
                        self.pending_group.append((orig_idx, curr_ts_pd, p, b, s, sanitized_row))
                        self._enforce_capacity(curr_ts_pd, emitted_rows=emitted_rows)
                    else:
                        # curr_ts_pd < active_ts: Out-of-order late arrival
                        self._handle_late_simultaneous_event(
                            orig_idx, curr_ts_pd, p, b, s, sanitized_row, watermark, emitted_rows
                        )
                else:
                    if watermark is not None and curr_ts_pd < watermark:
                        self._handle_late_simultaneous_event(
                            orig_idx, curr_ts_pd, p, b, s, sanitized_row, watermark, emitted_rows
                        )
                    else:
                        self._evict_expired(curr_ts_pd)
                        self._check_recovery(curr_ts_pd)
                        self.pending_group.append((orig_idx, curr_ts_pd, p, b, s, sanitized_row))
                        self._enforce_capacity(curr_ts_pd, emitted_rows=emitted_rows)

            if flush and self.pending_group:
                self._finalize_pending(emitted_rows)

            if not emitted_rows:
                # Return empty frame with expected columns if all rows are still pending
                cols = list(df.columns)
                for col in [
                    "Rolling_Flow_Velocity_10s",
                    "Rolling_Bytes_Velocity_10s",
                    "Rolling_Single_Pkt_Ratio_10s",
                    "_is_valid",
                    "_context_state",
                    "_overflow_events",
                ]:
                    if col not in cols:
                        cols.append(col)
                return pd.DataFrame(columns=cols)

            res_df = pd.DataFrame(emitted_rows)
            return res_df
        else:
            raise ValueError(f"Unknown causal_mode '{causal_mode}'. Expected 'simultaneous' or 'sequential'.")

    def _finalize_pending(self, emitted_rows: List[Dict[str, Any]]) -> None:
        """Finalizes the active pending timestamp group and appends to emitted_rows."""
        if not self.pending_group:
            return

        group_ts = self.pending_group[0][1]
        cutoff = group_ts - self.window_delta

        # Detach pending group items so capacity enforcement only evaluates buffer
        group_items = self.pending_group
        self.pending_group = []

        # Add all simultaneous events in group to sliding buffer
        for _, ts, pkts, bytes_, single, _ in group_items:
            self.buffer.append((ts, pkts, bytes_, single))
            self.total_processed += 1

        # Monotonic watermark advance
        if self.latest_timestamp is None or group_ts > self.latest_timestamp:
            self.latest_timestamp = group_ts

        # Evict entries <= cutoff (strictly open left boundary)
        while self.buffer and self.buffer[0][0] <= cutoff:
            self.buffer.popleft()

        # Enforce buffer capacity if buffer alone exceeds max_buffer_capacity
        self._enforce_capacity(group_ts)

        cnt = float(len(self.buffer))
        tot_bytes = float(sum(item[2] for item in self.buffer))
        ratio = float(sum(item[3] for item in self.buffer) / cnt if cnt > 0 else 0.0)

        for _, _, _, _, _, orig_dict in group_items:
            row_dict = dict(orig_dict)
            row_dict["Rolling_Flow_Velocity_10s"] = cnt
            row_dict["Rolling_Bytes_Velocity_10s"] = tot_bytes
            row_dict["Rolling_Single_Pkt_Ratio_10s"] = ratio
            row_dict["_is_valid"] = self.is_valid
            row_dict["_context_state"] = self.context_state
            row_dict["_overflow_events"] = self.total_dropped_overflow_events
            emitted_rows.append(row_dict)

    def flush(self) -> pd.DataFrame:
        """
        Forces emission of any unfinalized pending timestamp group.
        Useful when multi-batch streaming terminates.
        """
        emitted_rows: List[Dict[str, Any]] = []
        if self.pending_group:
            self._finalize_pending(emitted_rows)
        if not emitted_rows:
            cols = [
                "Rolling_Flow_Velocity_10s",
                "Rolling_Bytes_Velocity_10s",
                "Rolling_Single_Pkt_Ratio_10s",
                "_is_valid",
                "_context_state",
                "_overflow_events",
            ]
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(emitted_rows)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns diagnostic buffer metrics and degradation status."""
        total_retained = len(self.buffer) + len(self.pending_group)
        return {
            "window_seconds": self.window_seconds,
            "max_buffer_capacity": self.max_buffer_capacity,
            "clock_mode": self.clock_mode,
            "max_timestamp_horizon": str(self.max_timestamp_horizon) if self.max_timestamp_horizon is not None else None,
            "max_future_drift_seconds": self.max_future_drift_seconds,
            "buffer_length": len(self.buffer),
            "pending_group_length": len(self.pending_group),
            "total_retained_entries": total_retained,
            "is_degraded": self.is_degraded,
            "is_valid": self.is_valid,
            "context_state": self.context_state,
            "latest_timestamp": str(self.latest_timestamp) if self.latest_timestamp else None,
            "last_overflow_timestamp": str(self.last_overflow_timestamp) if self.last_overflow_timestamp else None,
            "total_processed": self.total_processed,
            "total_late_events": self.total_late_events,
            "total_dropped_overflow_events": self.total_dropped_overflow_events,
            "total_quarantined_records": self.quarantine_manager.total_quarantined_records,
            "quarantine_reason_counts": dict(self.quarantine_manager.quarantine_reason_counts),
            "estimated_memory_bytes": total_retained * 64,  # Approx 64 bytes per tuple
        }

    def export_state(self) -> Dict[str, Any]:
        """
        Exports internal causal buffer, pending simultaneous group, watermarks,
        and degradation metrics for checkpointing and state persistence.
        """
        return {
            "latest_timestamp": self.latest_timestamp.isoformat() if self.latest_timestamp is not None else None,
            "window_seconds": self.window_seconds,
            "max_buffer_capacity": self.max_buffer_capacity,
            "is_degraded": self.is_degraded,
            "last_overflow_timestamp": self.last_overflow_timestamp.isoformat() if self.last_overflow_timestamp is not None else None,
            "total_dropped_overflow_events": self.total_dropped_overflow_events,
            "total_processed": self.total_processed,
            "total_late_events": self.total_late_events,
            "rolling_buffer": [
                [
                    item[0].isoformat() if hasattr(item[0], "isoformat") else str(item[0]),
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                ]
                for item in self.buffer
            ],
            "pending_group": [
                [
                    int(item[0]),
                    item[1].isoformat() if hasattr(item[1], "isoformat") else str(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    dict(item[5]) if hasattr(item[5], "items") else {},
                ]
                for item in self.pending_group
            ],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restores causal buffer, pending simultaneous group, watermarks, and
        degradation state from a validated checkpoint dictionary.
        """
        self.reset()
        if not isinstance(state, dict):
            return

        buf_raw = state.get("rolling_buffer", [])
        for item in buf_raw:
            ts_pd = pd.Timestamp(item[0])
            self.buffer.append((ts_pd, float(item[1]), float(item[2]), float(item[3])))

        pend_raw = state.get("pending_group", [])
        for item in pend_raw:
            idx = int(item[0])
            ts_pd = pd.Timestamp(item[1])
            p = float(item[2])
            b = float(item[3])
            s = float(item[4])
            orig = dict(item[5])
            self.pending_group.append((idx, ts_pd, p, b, s, orig))

        latest_str = state.get("latest_timestamp")
        self.latest_timestamp = pd.Timestamp(latest_str) if latest_str else None

        overflow_str = state.get("last_overflow_timestamp")
        self.last_overflow_timestamp = pd.Timestamp(overflow_str) if overflow_str else None

        self.is_degraded = bool(state.get("is_degraded", False))
        self.total_dropped_overflow_events = int(state.get("total_dropped_overflow_events", 0))
        self.total_processed = int(state.get("total_processed", 0))
        self.total_late_events = int(state.get("total_late_events", 0))


# --------------------------------------------------------------------------
# 4. High-Performance Streaming Feature Engine
# --------------------------------------------------------------------------

class StreamingFeatureEngine:
    """
    Complete end-to-end Pure One-Way Streaming Feature Engine.

    Coordinates:
    - StreamingCausalContext: Rolling velocity generation and buffer capacity safety.
    - Pure Rate Calculation: Fwd_Bytes_Per_Sec, Fwd_Packets_Per_Sec.
    - Feature Extraction: Strictly 26 PURE_ONE_WAY_FEATURES in canonical order.
    - High-Performance NumPy Standardization: Pre-extracts fitted pipeline parameters
      (medians, means, scales, log indices) to execute sub-millisecond transforms
      without Pandas DataFrame construction overhead.
    """

    def __init__(
        self,
        pipeline: Optional[PureOneWayLogPreprocessingPipeline] = None,
        window_seconds: int = STREAMING_WINDOW_SECONDS,
        max_late_seconds: float = 10.0,
        max_buffer_capacity: int = 10_000,
        clock_mode: str = "event_time",
        max_future_drift_seconds: Optional[float] = None,
        max_timestamp_horizon: Optional[Union[pd.Timestamp, str]] = None,
        current_time_fn: Optional[Any] = None,
        quarantine_manager: Optional[FlowQuarantineManager] = None,
    ):
        self.context: StreamingCausalContext = StreamingCausalContext(
            window_seconds=window_seconds,
            max_late_seconds=max_late_seconds,
            max_buffer_capacity=max_buffer_capacity,
            clock_mode=clock_mode,
            max_future_drift_seconds=max_future_drift_seconds,
            max_timestamp_horizon=max_timestamp_horizon,
            current_time_fn=current_time_fn,
            quarantine_manager=quarantine_manager,
        )
        self.pipeline: Optional[PureOneWayLogPreprocessingPipeline] = pipeline
        self.feature_names: List[str] = list(PURE_ONE_WAY_FEATURES)

        # Pre-extracted parameters for microsecond NumPy transforms
        self._medians: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._scales: Optional[np.ndarray] = None
        self._log_indices: List[int] = []

        if self.pipeline is not None and getattr(self.pipeline, "is_fitted", False):
            self._compile_fast_transforms()

    @property
    def quarantine_manager(self) -> FlowQuarantineManager:
        """Access the underlying FlowQuarantineManager."""
        return self.context.quarantine_manager

    @property
    def total_quarantined_records(self) -> int:
        """Cumulative total of quarantined records."""
        return self.context.quarantine_manager.total_quarantined_records

    @property
    def is_degraded(self) -> bool:
        """Returns True if engine causal context is degraded due to capacity overflow."""
        return self.context.is_degraded

    @property
    def is_valid(self) -> bool:
        """Returns True if engine causal context is complete and valid."""
        return self.context.is_valid

    @property
    def context_state(self) -> str:
        """Returns 'HEALTHY' or 'DEGRADED'."""
        return self.context.context_state

    @property
    def total_dropped_overflow_events(self) -> int:
        """Total cumulative dropped events due to buffer capacity overflow."""
        return self.context.total_dropped_overflow_events

    @property
    def max_buffer_capacity(self) -> int:
        """Configured maximum buffer capacity."""
        return self.context.max_buffer_capacity

    def get_metrics(self) -> Dict[str, Any]:
        """Returns diagnostic metrics from underlying causal context."""
        return self.context.get_metrics()

    def _compile_fast_transforms(self) -> None:
        """Pre-extracts pipeline parameters into contiguous NumPy arrays."""
        if not getattr(self.pipeline, "is_fitted", False):
            return

        self._medians = np.asarray(self.pipeline.imputer.statistics_, dtype=np.float64)
        self._means = np.asarray(self.pipeline.scaler.mean_, dtype=np.float64)
        self._scales = np.asarray(self.pipeline.scaler.scale_, dtype=np.float64)
        pipe_features = list(self.pipeline.features)
        self._log_indices = [i for i, f in enumerate(pipe_features) if f in PURE_LOG_FEATURES]

    def warm_up(self, historical_df: pd.DataFrame) -> int:
        """Pre-warms the internal causal context buffer."""
        return self.context.warm_up(historical_df)

    def reset(self) -> None:
        """Resets the context buffer."""
        self.context.reset()

    def export_state(self) -> Dict[str, Any]:
        """Exports underlying context state for checkpointing."""
        return self.context.export_state()

    def load_state(self, state: Dict[str, Any]) -> None:
        """Loads underlying context state from a validated checkpoint."""
        self.context.load_state(state)

    def extract_features_df(
        self,
        df: pd.DataFrame,
        causal_mode: str = "simultaneous",
        flush: bool = True,
        include_metadata: bool = False,
    ) -> pd.DataFrame:
        """
        Generates and extracts the 26 PURE_ONE_WAY_FEATURES as a DataFrame.
        Guarantees exact column ordering and naming.
        If include_metadata is True, also includes '_is_valid', '_context_state',
        and '_overflow_events' columns.
        """
        augmented = self.context.process_batch(df, causal_mode=causal_mode, flush=flush)
        if len(augmented) == 0:
            cols = self.feature_names + (["_is_valid", "_context_state", "_overflow_events"] if include_metadata else [])
            return pd.DataFrame(columns=cols)

        if include_metadata:
            cols = self.feature_names + ["_is_valid", "_context_state", "_overflow_events"]
            return augmented[cols].copy()
        return augmented[self.feature_names].copy()

    def transform_flow_fast(
        self,
        flow: Dict[str, Any],
        return_status: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, bool, str]]:
        """
        Fast path: Preprocesses a single flow in microsecond pure NumPy.
        Bypasses Pandas DataFrame creation, achieving >60x speedup over standard pipeline.
        Returns standardized array of shape (1, 26).
        If return_status is True, returns (scaled_array, is_valid, context_state).
        """
        if self._medians is None or self._means is None or self._scales is None:
            raise RuntimeError(
                "Fast transforms require a fitted PureOneWayLogPreprocessingPipeline. "
                "Initialize engine with a pre-fitted pipeline."
            )

        # 1. Validation & Quarantine
        is_val, reason, sanitized = validate_and_sanitize_flow_record(
            flow,
            clock_mode=self.context.clock_mode,
            max_future_drift_seconds=self.context.max_future_drift_seconds,
            max_timestamp_horizon=self.context.max_timestamp_horizon,
            current_time_fn=self.context.current_time_fn,
        )
        if not is_val:
            self.context.quarantine_manager.quarantine(flow, reason)
            if return_status:
                return np.full((1, 26), np.nan, dtype=np.float64), False, "QUARANTINED"
            raise ValueError(f"Flow record quarantined: {reason}")

        # 2. Update temporal context with sanitized values
        ts = sanitized["Timestamp"]
        iat = sanitized.get("Fwd IAT Total", 0.0)
        bytes_ = sanitized["Total Length of Fwd Packets"]
        pkts = sanitized["Total Fwd Packets"]

        b_rate, p_rate = compute_pure_forward_rates_scalar(iat, bytes_, pkts)
        rf, rb, rs = self.context.update(ts, pkts, bytes_)

        # 3. Populate raw feature vector (26 dimensions)
        raw_vals = np.empty(26, dtype=np.float64)
        for idx, feat in enumerate(self.feature_names):
            if feat == "Fwd_Bytes_Per_Sec":
                val = b_rate
            elif feat == "Fwd_Packets_Per_Sec":
                val = p_rate
            elif feat == "Rolling_Flow_Velocity_10s":
                val = rf
            elif feat == "Rolling_Bytes_Velocity_10s":
                val = rb
            elif feat == "Rolling_Single_Pkt_Ratio_10s":
                val = rs
            else:
                raw_val = sanitized.get(feat, np.nan)
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    val = np.nan

            # Non-negative clamping for physical features
            if feat in NON_NEGATIVE_PHYSICAL_FEATURES and val < 0.0:
                val = np.nan
            if feat in TCP_WINDOW_FEATURES and val < 0.0:
                val = np.nan
            if math.isinf(val):
                val = np.nan

            raw_vals[idx] = val

        # 3. Log transform skewed features
        for log_idx in self._log_indices:
            v = raw_vals[log_idx]
            if not np.isnan(v) and v >= 0.0:
                raw_vals[log_idx] = np.log1p(v)

        # 4. Impute missing values
        nan_mask = np.isnan(raw_vals)
        raw_vals[nan_mask] = self._medians[nan_mask]

        # 5. Standard scale
        scaled = ((raw_vals - self._means) / self._scales).reshape(1, 26)

        if return_status:
            return scaled, self.is_valid, self.context_state
        return scaled

    def transform_batch(
        self,
        df: pd.DataFrame,
        causal_mode: str = "simultaneous",
        return_status: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, bool, str]]:
        """
        Batch path: Transforms a DataFrame using the underlying pipeline.
        Returns standardized array of shape (N, 26).
        If return_status is True, returns (scaled_array, is_valid, context_state).
        """
        if self.pipeline is None:
            raise RuntimeError("Engine requires a fitted pipeline to perform transform_batch().")

        features_df = self.extract_features_df(df, causal_mode=causal_mode, flush=True)
        scaled = self.pipeline.transform(features_df)
        if return_status:
            return scaled, self.is_valid, self.context_state
        return scaled
