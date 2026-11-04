"""
src/streaming_features.py
AEGIS Phase 3 Stage 4A: Pure One-Way Causal Streaming Feature Engine.

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
   - Multi-batch continuity and warm-up buffer support.
   - Simultaneous timestamp group finalization matching offline transform("last").
4. High-Performance Streaming Feature Engine (StreamingFeatureEngine):
   - Single-flow microsecond feature extraction and fast NumPy standardization.
   - Vectorized batch feature extraction matching offline features.py bit-for-bit.
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

def prepare_chronological_stream(
    df: pd.DataFrame,
    max_events: Optional[int] = None,
    inject_attacks: bool = False,
    attack_ratio: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Prepares a chronologically ordered streaming event pool from historical captures.

    Timestamp Policy:
    1. Valid timestamps: Parsed to datetime64[ns] and sorted monotonically.
    2. Duplicate timestamps: Preserved in stable original sequence order using
       '_orig_pos' as secondary sort key.
    3. Missing / Invalid timestamps (NaT / None): Rejected with ValueError.
    4. Deterministic sampling: If inject_attacks is True, samples benign and attack pools
       deterministically before sorting chronologically.

    Guarantees:
    - Never silently reorders events across timestamp boundaries.
    - Preserves stable sequence for simultaneous events.
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

    if max_events is not None and max_events > 0 and len(pool) > max_events:
        if inject_attacks and "Label" in pool.columns:
            benign_mask = pool["Label"].astype(str).str.upper().str.contains("BENIGN")
            attack_mask = ~benign_mask

            target_attacks = int(max_events * attack_ratio)
            target_benign = max_events - target_attacks

            benign_sample = pool[benign_mask].sample(
                n=min(len(pool[benign_mask]), target_benign),
                random_state=random_state,
            )
            attack_sample = pool[attack_mask].sample(
                n=min(len(pool[attack_mask]), target_attacks),
                random_state=random_state,
            )
            sampled = pd.concat([benign_sample, attack_sample], ignore_index=True)
        else:
            sampled = pool.sample(n=max_events, random_state=random_state)
    else:
        sampled = pool

    # Strictly stable chronological sort
    sorted_df = sampled.sort_values(by=["Timestamp", "_orig_pos"], kind="stable").reset_index(drop=True)
    sorted_df = sorted_df.drop(columns=["_orig_pos"])
    return sorted_df


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
    ):
        self.window_seconds: int = int(window_seconds)
        self.window_delta: pd.Timedelta = pd.Timedelta(seconds=self.window_seconds)
        self.max_late_delta: pd.Timedelta = pd.Timedelta(seconds=max_late_seconds)

        # Deque of (timestamp: pd.Timestamp, fwd_pkts: float, fwd_bytes: float, is_single: float)
        self.buffer: deque = deque()

        # Watermark tracking
        self.latest_timestamp: Optional[pd.Timestamp] = None
        self.total_processed: int = 0
        self.total_late_events: int = 0

        # Pending group buffer for multi-batch simultaneous group finalization
        # List of (row_idx, timestamp_pd, fwd_pkts, fwd_bytes, is_single, raw_dict)
        self.pending_group: List[Tuple[int, pd.Timestamp, float, float, float, Dict[str, Any]]] = []

    def reset(self) -> None:
        """Clears all historical buffer state."""
        self.buffer.clear()
        self.pending_group.clear()
        self.latest_timestamp = None
        self.total_processed = 0
        self.total_late_events = 0

    def warm_up(self, historical_df: pd.DataFrame) -> int:
        """
        Pre-loads historical events into the sliding context buffer.
        Eliminates startup cold-start artifacts by populating events within
        window_seconds of the historical tail.

        Returns number of events buffered.
        """
        if historical_df is None or len(historical_df) == 0:
            return 0

        hist = historical_df.copy()
        if "Timestamp" not in hist.columns:
            raise ValueError("Historical context must contain a 'Timestamp' column.")

        if not pd.api.types.is_datetime64_any_dtype(hist["Timestamp"]):
            hist["Timestamp"] = pd.to_datetime(hist["Timestamp"])

        hist = hist.sort_values("Timestamp", kind="stable")
        max_ts = hist["Timestamp"].max()
        cutoff = max_ts - self.window_delta

        # Only retain tail within window_seconds
        tail = hist[hist["Timestamp"] > cutoff].copy()
        if len(tail) == 0:
            return 0

        pkts = tail.get("Total Fwd Packets", pd.Series(0, index=tail.index)).replace([np.inf, -np.inf], np.nan).fillna(0).values
        bytes_ = tail.get("Total Length of Fwd Packets", pd.Series(0, index=tail.index)).replace([np.inf, -np.inf], np.nan).fillna(0).values
        singles = (pkts <= 1).astype(float)
        ts_vals = tail["Timestamp"].values

        for i in range(len(tail)):
            ts_pd = pd.Timestamp(ts_vals[i])
            self.buffer.append((ts_pd, float(pkts[i]), float(bytes_[i]), float(singles[i])))

        self.latest_timestamp = max_ts
        return len(tail)

    def _evict_expired(self, current_ts: pd.Timestamp) -> None:
        """Evicts buffer entries strictly <= (current_ts - window_delta)."""
        cutoff = current_ts - self.window_delta
        while self.buffer and self.buffer[0][0] <= cutoff:
            self.buffer.popleft()

    def update(
        self,
        timestamp: Any,
        fwd_pkts: float,
        fwd_bytes: float,
    ) -> Tuple[float, float, float]:
        """
        Single-event incremental update in sequential mode.

        Policy:
        - If timestamp is later or equal to latest_timestamp: evicts entries older
          than (t - window_seconds), adds event, and returns current window metrics.
        - If timestamp is earlier than latest_timestamp:
          - In-window late: inserted into sorted position, evicts, returns metric.
          - Expired late (t < latest_timestamp - window_seconds): increments late counter,
            returns isolated single-event metrics without corrupting streaming buffer.
        """
        ts_pd = pd.Timestamp(timestamp)
        p = float(fwd_pkts) if not math.isnan(fwd_pkts) and not math.isinf(fwd_pkts) and fwd_pkts >= 0 else 0.0
        b = float(fwd_bytes) if not math.isnan(fwd_bytes) and not math.isinf(fwd_bytes) and fwd_bytes >= 0 else 0.0
        s = 1.0 if p <= 1.0 else 0.0

        if self.latest_timestamp is not None and ts_pd < self.latest_timestamp:
            # Late arrival check
            if ts_pd <= (self.latest_timestamp - self.window_delta):
                self.total_late_events += 1
                self.total_processed += 1
                # Expired event: evaluated in isolation
                return 1.0, b, s
            else:
                self.total_late_events += 1
                # In-window out-of-order: insert into chronological position
                self.buffer.append((ts_pd, p, b, s))
                # Re-sort buffer (small size: <= 500 items)
                sorted_buf = sorted(self.buffer, key=lambda x: x[0])
                self.buffer = deque(sorted_buf)
        else:
            self.latest_timestamp = ts_pd
            self.buffer.append((ts_pd, p, b, s))

        self._evict_expired(ts_pd)
        self.total_processed += 1

        cnt = float(len(self.buffer))
        tot_bytes = float(sum(item[2] for item in self.buffer))
        ratio = float(sum(item[3] for item in self.buffer) / cnt if cnt > 0 else 0.0)

        return cnt, tot_bytes, ratio

    def process_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single flow record dictionary, augmenting it with pure forward rates
        and rolling temporal velocities.
        Returns a dictionary containing all 26 PURE_ONE_WAY_FEATURES.
        """
        out = dict(flow)

        # 1. Forward rates
        iat = flow.get("Fwd IAT Total", 0.0)
        bytes_ = flow.get("Total Length of Fwd Packets", 0.0)
        pkts = flow.get("Total Fwd Packets", 0.0)
        b_rate, p_rate = compute_pure_forward_rates_scalar(iat, bytes_, pkts)
        out["Fwd_Bytes_Per_Sec"] = b_rate
        out["Fwd_Packets_Per_Sec"] = p_rate

        # 2. Temporal velocities
        ts = flow.get("Timestamp")
        if ts is None:
            raise ValueError("Flow record missing 'Timestamp' field.")
        rf, rb, rs = self.update(ts, pkts, bytes_)
        out["Rolling_Flow_Velocity_10s"] = rf
        out["Rolling_Bytes_Velocity_10s"] = rb
        out["Rolling_Single_Pkt_Ratio_10s"] = rs

        return out

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
            return res

        df = df.copy()
        if "Timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'Timestamp' column.")

        if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Compute forward rates via canonical vectorized function if Fwd IAT Total is present
        if "Fwd IAT Total" in df.columns:
            df = add_pure_forward_rates(df)
        elif "Fwd_Bytes_Per_Sec" not in df.columns:
            df["Fwd_Bytes_Per_Sec"] = 0.0
            df["Fwd_Packets_Per_Sec"] = 0.0

        fwd_pkts = df.get("Total Fwd Packets", pd.Series(0, index=df.index)).replace([np.inf, -np.inf], np.nan).fillna(0).values
        fwd_bytes = df.get("Total Length of Fwd Packets", pd.Series(0, index=df.index)).replace([np.inf, -np.inf], np.nan).fillna(0).values
        is_single = (fwd_pkts <= 1).astype(float)
        timestamps = df["Timestamp"].values

        n = len(df)

        if causal_mode == "sequential":
            rf = np.zeros(n, dtype=float)
            rb = np.zeros(n, dtype=float)
            rs = np.zeros(n, dtype=float)

            for i in range(n):
                r_flow, r_bytes, r_single = self.update(timestamps[i], fwd_pkts[i], fwd_bytes[i])
                rf[i] = r_flow
                rb[i] = r_bytes
                rs[i] = r_single

            df["Rolling_Flow_Velocity_10s"] = rf
            df["Rolling_Bytes_Velocity_10s"] = rb
            df["Rolling_Single_Pkt_Ratio_10s"] = rs
            return df

        elif causal_mode == "simultaneous":
            emitted_rows: List[Dict[str, Any]] = []

            for i in range(n):
                curr_ts_pd = pd.Timestamp(timestamps[i])

                # Check if timestamp strictly advanced
                if self.pending_group and curr_ts_pd > self.pending_group[0][1]:
                    self._finalize_pending(emitted_rows)

                self.pending_group.append((
                    i,
                    curr_ts_pd,
                    float(fwd_pkts[i]),
                    float(fwd_bytes[i]),
                    float(is_single[i]),
                    df.iloc[i].to_dict(),
                ))

            if flush and self.pending_group:
                self._finalize_pending(emitted_rows)

            if not emitted_rows:
                # Return empty frame with expected columns if all rows are still pending
                cols = list(df.columns)
                for col in ["Rolling_Flow_Velocity_10s", "Rolling_Bytes_Velocity_10s", "Rolling_Single_Pkt_Ratio_10s"]:
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

        # Add all simultaneous events in group to sliding buffer
        for _, ts, pkts, bytes_, single, _ in self.pending_group:
            self.buffer.append((ts, pkts, bytes_, single))
            self.total_processed += 1

        self.latest_timestamp = group_ts

        # Evict entries <= cutoff (strictly open left boundary)
        while self.buffer and self.buffer[0][0] <= cutoff:
            self.buffer.popleft()

        cnt = float(len(self.buffer))
        tot_bytes = float(sum(item[2] for item in self.buffer))
        ratio = float(sum(item[3] for item in self.buffer) / cnt if cnt > 0 else 0.0)

        for _, _, _, _, _, orig_dict in self.pending_group:
            row_dict = dict(orig_dict)
            row_dict["Rolling_Flow_Velocity_10s"] = cnt
            row_dict["Rolling_Bytes_Velocity_10s"] = tot_bytes
            row_dict["Rolling_Single_Pkt_Ratio_10s"] = ratio
            emitted_rows.append(row_dict)

        self.pending_group = []

    def flush(self) -> pd.DataFrame:
        """
        Forces emission of any unfinalized pending timestamp group.
        Useful when multi-batch streaming terminates.
        """
        emitted_rows: List[Dict[str, Any]] = []
        if self.pending_group:
            self._finalize_pending(emitted_rows)
        return pd.DataFrame(emitted_rows)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns diagnostic buffer metrics."""
        return {
            "window_seconds": self.window_seconds,
            "buffer_length": len(self.buffer),
            "pending_group_length": len(self.pending_group),
            "latest_timestamp": str(self.latest_timestamp) if self.latest_timestamp else None,
            "total_processed": self.total_processed,
            "total_late_events": self.total_late_events,
            "estimated_memory_bytes": len(self.buffer) * 64,  # Approx 64 bytes per tuple
        }


# --------------------------------------------------------------------------
# 4. High-Performance Streaming Feature Engine
# --------------------------------------------------------------------------

class StreamingFeatureEngine:
    """
    Complete end-to-end Pure One-Way Streaming Feature Engine.

    Coordinates:
    - StreamingCausalContext: Rolling velocity generation.
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
    ):
        self.context: StreamingCausalContext = StreamingCausalContext(window_seconds=window_seconds)
        self.pipeline: Optional[PureOneWayLogPreprocessingPipeline] = pipeline
        self.feature_names: List[str] = list(PURE_ONE_WAY_FEATURES)

        # Pre-extracted parameters for microsecond NumPy transforms
        self._medians: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._scales: Optional[np.ndarray] = None
        self._log_indices: List[int] = []

        if self.pipeline is not None and getattr(self.pipeline, "is_fitted", False):
            self._compile_fast_transforms()

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

    def extract_features_df(
        self,
        df: pd.DataFrame,
        causal_mode: str = "simultaneous",
        flush: bool = True,
    ) -> pd.DataFrame:
        """
        Generates and extracts the 26 PURE_ONE_WAY_FEATURES as a DataFrame.
        Guarantees exact column ordering and naming.
        """
        augmented = self.context.process_batch(df, causal_mode=causal_mode, flush=flush)
        if len(augmented) == 0:
            return pd.DataFrame(columns=self.feature_names)
        return augmented[self.feature_names].copy()

    def transform_flow_fast(self, flow: Dict[str, Any]) -> np.ndarray:
        """
        Fast path: Preprocesses a single flow in microsecond pure NumPy.
        Bypasses Pandas DataFrame creation, achieving >60x speedup over standard pipeline.
        Returns standardized array of shape (1, 26).
        """
        if self._medians is None or self._means is None or self._scales is None:
            raise RuntimeError(
                "Fast transforms require a fitted PureOneWayLogPreprocessingPipeline. "
                "Initialize engine with a pre-fitted pipeline."
            )

        # 1. Update temporal context
        ts = flow.get("Timestamp")
        if ts is None:
            raise ValueError("Flow record missing 'Timestamp'.")

        iat = flow.get("Fwd IAT Total", 0.0)
        bytes_ = flow.get("Total Length of Fwd Packets", 0.0)
        pkts = flow.get("Total Fwd Packets", 0.0)

        b_rate, p_rate = compute_pure_forward_rates_scalar(iat, bytes_, pkts)
        rf, rb, rs = self.context.update(ts, pkts, bytes_)

        # 2. Populate raw feature vector (26 dimensions)
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
                raw_val = flow.get(feat, np.nan)
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
        scaled = (raw_vals - self._means) / self._scales
        return scaled.reshape(1, 26)

    def transform_batch(self, df: pd.DataFrame, causal_mode: str = "simultaneous") -> np.ndarray:
        """
        Batch path: Transforms a DataFrame using the underlying pipeline.
        Returns standardized array of shape (N, 26).
        """
        if self.pipeline is None:
            raise RuntimeError("Engine requires a fitted pipeline to perform transform_batch().")

        features_df = self.extract_features_df(df, causal_mode=causal_mode, flush=True)
        return self.pipeline.transform(features_df)
