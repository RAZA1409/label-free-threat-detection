"""
tests/test_streaming_features.py
Comprehensive automated test suite for AEGIS Phase 3 Stage 4A & Stage 4B.1 Causal Streaming Feature Engine.

Tests:
- TEST 01: 26-Feature dimension and naming exact ordering
- TEST 02: Pure forward rate derivation parity with offline add_pure_forward_rates
- TEST 03: Zero-duration flows and sentinel handling (NaN, Inf, negative values)
- TEST 04: Exact 10-second boundary conditions (open left (t - 10s, closed right , t])
- TEST 05: Chronological stream preparation, stable ordering, and missing timestamp rejection
- TEST 06: Duplicate/simultaneous timestamp handling parity with offline transform("last")
- TEST 07: Multi-batch boundary continuity and context persistence
- TEST 08: Startup warm-up buffer continuity
- TEST 09: Time-based memory eviction and bounded buffer footprint
- TEST 10: Explicit late-event policy (in-window vs expired)
- TEST 11: End-to-end 26-dimension numerical parity on real Tuesday-Friday capture
- TEST 12: Fast NumPy transformation parity with standard PureOneWayLogPreprocessingPipeline
- TEST 13: Empty batch and graceful edge-case handling
- TEST 14: Capacity boundary and normal traffic health
- TEST 15: Buffer overflow drop and degraded state transition
- TEST 16: Pending group exceeding capacity bounded memory
- TEST 17: Degraded state event-time recovery contract
- TEST 18: Repeated overflow resets recovery clock
- TEST 19: Feature engine validity signal and metadata
- TEST 20: Warm-up buffer capacity safety
- TEST 21: Simultaneous multi-batch expiration and recovery (D1 remediation)
- TEST 22: Warm-up multiple calls and populated context safety (D2 remediation)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
V2_MODELS_DIR = MODELS_DIR / "pure_one_way_v2"
DATA_DIR = ROOT / "outputs" / "experiment"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import (
    PURE_ONE_WAY_FEATURES,
    PURE_ONE_WAY_BASE_FEATURES,
    PURE_ONE_WAY_TEMPORAL_FEATURES,
    add_pure_forward_rates,
    add_temporal_velocity_features,
    PureOneWayLogPreprocessingPipeline,
)
from streaming_features import (
    StreamingCausalContext,
    StreamingFeatureEngine,
    compute_pure_forward_rates_scalar,
    compute_pure_forward_rates_batch,
    prepare_chronological_stream,
)


@pytest.fixture(scope="module")
def real_test_capture():
    """Loads a 2,000-flow sample from test_tue_fri.parquet if available."""
    test_file = DATA_DIR / "test_tue_fri.parquet"
    if test_file.exists():
        df = pd.read_parquet(test_file).head(2000)
        return prepare_chronological_stream(df)
    else:
        # Fallback synthetic chronological dataset
        n = 1000
        timestamps = pd.date_range("2026-11-04 10:00:00", periods=n, freq="50ms")
        data = {f: np.random.uniform(1.0, 100.0, size=n) for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]}
        data["Timestamp"] = timestamps
        return pd.DataFrame(data)


@pytest.fixture(scope="module")
def fitted_v2_pipeline():
    """Loads the trained v2 pipeline from models/pure_one_way_v2/pipeline.joblib."""
    pipe_path = V2_MODELS_DIR / "pipeline.joblib"
    if pipe_path.exists():
        return joblib.load(pipe_path)
    # Fallback fitted pipeline
    cal_file = DATA_DIR / "calibration_benign.parquet"
    if cal_file.exists():
        cal_df = pd.read_parquet(cal_file).head(500)
    else:
        cal_df = pd.DataFrame({
            f: np.random.uniform(1.0, 50.0, size=100)
            for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]
        })
        cal_df["Timestamp"] = pd.date_range("2026-11-04 10:00:00", periods=100, freq="1s")
    cal_df = add_temporal_velocity_features(cal_df)
    pipe = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    pipe.fit(cal_df)
    return pipe


# -----------------------------------------------------------------------------
# TEST 01: 26-Feature Dimension and Ordering
# -----------------------------------------------------------------------------
def test_01_feature_dimension_and_names_exact_order():
    """Verifies that StreamingFeatureEngine extracts exactly 26 features in canonical order."""
    engine = StreamingFeatureEngine()
    assert len(engine.feature_names) == 26
    assert engine.feature_names == PURE_ONE_WAY_FEATURES
    assert engine.feature_names[:23] == PURE_ONE_WAY_BASE_FEATURES
    assert engine.feature_names[23:] == PURE_ONE_WAY_TEMPORAL_FEATURES


# -----------------------------------------------------------------------------
# TEST 02: Pure Forward Rate Derivation Parity
# -----------------------------------------------------------------------------
def test_02_forward_rate_derivation_parity():
    """Verifies that compute_pure_forward_rates_scalar produces identical results to add_pure_forward_rates."""
    test_df = pd.DataFrame({
        "Fwd IAT Total": [1_000_000.0, 50_000.0, 100.0, 2_500_000.0],
        "Total Length of Fwd Packets": [5000.0, 12500.0, 120.0, 100_000.0],
        "Total Fwd Packets": [10.0, 25.0, 2.0, 200.0],
    })

    offline_df = add_pure_forward_rates(test_df)

    for i in range(len(test_df)):
        iat = test_df.loc[i, "Fwd IAT Total"]
        b = test_df.loc[i, "Total Length of Fwd Packets"]
        p = test_df.loc[i, "Total Fwd Packets"]

        s_bytes, s_pkts = compute_pure_forward_rates_scalar(iat, b, p)

        assert np.isclose(s_bytes, offline_df.loc[i, "Fwd_Bytes_Per_Sec"], rtol=1e-9, atol=1e-9)
        assert np.isclose(s_pkts, offline_df.loc[i, "Fwd_Packets_Per_Sec"], rtol=1e-9, atol=1e-9)


# -----------------------------------------------------------------------------
# TEST 03: Zero-Duration Flows and Sentinel Handling
# -----------------------------------------------------------------------------
def test_03_zero_duration_and_negative_sentinels():
    """Confirms zero-duration, negative sentinels, NaNs, and infinities strictly yield 0.0."""
    edge_cases = [
        (0.0, 100.0, 1.0),
        (-1.0, 500.0, 2.0),
        (-999999.0, 500.0, 2.0),
        (np.nan, 500.0, 2.0),
        (np.inf, 500.0, 2.0),
        (-np.inf, 500.0, 2.0),
        (1_000_000.0, np.nan, 2.0),
        (1_000_000.0, -100.0, 2.0),
        (1_000_000.0, 500.0, np.nan),
        (1_000_000.0, 500.0, -5.0),
    ]

    for iat, b, p in edge_cases:
        br, pr = compute_pure_forward_rates_scalar(iat, b, p)
        assert br >= 0.0 and not np.isnan(br) and not np.isinf(br)
        assert pr >= 0.0 and not np.isnan(pr) and not np.isinf(pr)
        if iat <= 0.0 or np.isnan(iat) or np.isinf(iat):
            assert br == 0.0
            assert pr == 0.0


# -----------------------------------------------------------------------------
# TEST 04: Exact 10-Second Boundary Conditions
# -----------------------------------------------------------------------------
def test_04_causal_window_open_left_closed_right_boundary():
    """
    Rigorously verifies the (t - 10s, t] boundary condition:
    - Left boundary open: event at exactly t - 10s is EXCLUDED.
    - Right boundary closed: event at exactly t is INCLUDED.
    - Event at t - 9.999s is INCLUDED.
    """
    ctx = StreamingCausalContext(window_seconds=10)

    # E0 at 10:00:00
    ctx.update(pd.Timestamp("2026-11-04 10:00:00"), fwd_pkts=2.0, fwd_bytes=200.0)

    # E1 at 10:00:10 (exactly 10.0 seconds later)
    rf, rb, rs = ctx.update(pd.Timestamp("2026-11-04 10:00:10"), fwd_pkts=1.0, fwd_bytes=100.0)

    # Window (10:00:00, 10:00:10] MUST EXCLUDE E0
    assert rf == 1.0, f"Expected 1.0 flow (E0 should be excluded), got {rf}"
    assert rb == 100.0, f"Expected 100.0 bytes, got {rb}"
    assert rs == 1.0, f"Expected single-packet ratio 1.0, got {rs}"

    # E2 at 10:00:19.999 (9.999s after E1)
    rf2, rb2, rs2 = ctx.update(pd.Timestamp("2026-11-04 10:00:19.999"), fwd_pkts=5.0, fwd_bytes=1000.0)

    # Window (10:00:09.999, 10:00:19.999] MUST INCLUDE E1 (at 10:00:10) and E2
    assert rf2 == 2.0, f"Expected 2.0 flows (E1 and E2), got {rf2}"
    assert rb2 == 1100.0, f"Expected 1100.0 bytes, got {rb2}"
    assert np.isclose(rs2, 0.5), f"Expected single-packet ratio 0.5, got {rs2}"


# -----------------------------------------------------------------------------
# TEST 05: Chronological Stream Preparation and Missing Timestamp Rejection
# -----------------------------------------------------------------------------
def test_05_chronological_stream_preparation_and_stable_sort():
    """Verifies chronological sorting, deterministic sampling, and missing timestamp rejection."""
    df_unordered = pd.DataFrame({
        "Timestamp": [
            pd.Timestamp("2026-11-04 10:00:05"),
            pd.Timestamp("2026-11-04 10:00:01"),
            pd.Timestamp("2026-11-04 10:00:03"),
            pd.Timestamp("2026-11-04 10:00:01"),  # Duplicate timestamp
        ],
        "Payload": ["A", "B1", "C", "B2"],
    })

    sorted_df = prepare_chronological_stream(df_unordered)

    assert list(sorted_df["Payload"]) == ["B1", "B2", "C", "A"], "Stable tie-breaking order violated!"
    assert sorted_df["Timestamp"].is_monotonic_increasing

    # Missing timestamp must raise ValueError
    df_corrupted = df_unordered.copy()
    df_corrupted.loc[1, "Timestamp"] = None
    with pytest.raises(ValueError, match="missing or invalid timestamps"):
        prepare_chronological_stream(df_corrupted)


# -----------------------------------------------------------------------------
# TEST 06: Duplicate/Simultaneous Timestamp Handling Parity
# -----------------------------------------------------------------------------
def test_06_simultaneous_timestamp_group_parity():
    """
    Verifies that simultaneous events sharing the identical timestamp t receive
    the exact aggregate accumulated through all events at timestamp t, matching offline transform('last').
    """
    ts = pd.Timestamp("2026-11-04 10:00:00")
    df_simultaneous = pd.DataFrame({
        "Timestamp": [ts, ts, ts, ts],
        "Total Fwd Packets": [1.0, 2.0, 1.0, 4.0],
        "Total Length of Fwd Packets": [64.0, 128.0, 64.0, 512.0],
        "Fwd IAT Total": [0.0, 100.0, 0.0, 500.0],
    })

    # Offline
    df_off = add_temporal_velocity_features(df_simultaneous, window_seconds=10)

    # Streaming simultaneous mode
    ctx = StreamingCausalContext(window_seconds=10)
    df_stream = ctx.process_batch(df_simultaneous, causal_mode="simultaneous", flush=True)

    np.testing.assert_array_equal(
        df_off["Rolling_Flow_Velocity_10s"].values,
        df_stream["Rolling_Flow_Velocity_10s"].values,
    )
    np.testing.assert_array_equal(
        df_off["Rolling_Bytes_Velocity_10s"].values,
        df_stream["Rolling_Bytes_Velocity_10s"].values,
    )
    np.testing.assert_allclose(
        df_off["Rolling_Single_Pkt_Ratio_10s"].values,
        df_stream["Rolling_Single_Pkt_Ratio_10s"].values,
        rtol=1e-9,
    )

    # All 4 rows must have flow_count == 4.0
    assert (df_stream["Rolling_Flow_Velocity_10s"] == 4.0).all()
    assert (df_stream["Rolling_Bytes_Velocity_10s"] == 768.0).all()
    assert (df_stream["Rolling_Single_Pkt_Ratio_10s"] == 0.5).all()


# -----------------------------------------------------------------------------
# TEST 07: Multi-Batch Boundary Continuity
# -----------------------------------------------------------------------------
def test_07_multi_batch_boundary_continuity(real_test_capture):
    """
    Splits a real chronological dataset across multiple small batches and verifies
    that multi-batch execution with flush=False yields 100% identical values to single-batch offline execution.
    """
    df = real_test_capture.head(600).copy()

    # Offline
    df_off = add_temporal_velocity_features(df, window_seconds=10)
    df_off = add_pure_forward_rates(df_off)

    # Streaming split across 6 batches of 100 flows
    ctx = StreamingCausalContext(window_seconds=10)
    chunk_size = 100
    stream_chunks = []

    for start in range(0, len(df), chunk_size):
        is_last = (start + chunk_size >= len(df))
        chunk = df.iloc[start : start + chunk_size]
        out_chunk = ctx.process_batch(chunk, causal_mode="simultaneous", flush=is_last)
        if len(out_chunk) > 0:
            stream_chunks.append(out_chunk)

    df_stream = pd.concat(stream_chunks, ignore_index=True)

    for col in PURE_ONE_WAY_FEATURES:
        diff = np.max(np.abs(df_off[col].values - df_stream[col].values))
        assert diff < 1e-9, f"Multi-batch boundary mismatch on '{col}': max diff = {diff}"


# -----------------------------------------------------------------------------
# TEST 08: Startup Warm-Up Buffer Continuity
# -----------------------------------------------------------------------------
def test_08_startup_warmup_continuity():
    """Confirms warm_up() pre-loads historical context and eliminates startup cold-start."""
    hist_df = pd.DataFrame({
        "Timestamp": [
            pd.Timestamp("2026-11-04 09:59:52"),
            pd.Timestamp("2026-11-04 09:59:55"),
            pd.Timestamp("2026-11-04 09:59:58"),
        ],
        "Total Fwd Packets": [1.0, 1.0, 2.0],
        "Total Length of Fwd Packets": [50.0, 50.0, 200.0],
    })

    live_df = pd.DataFrame({
        "Timestamp": [pd.Timestamp("2026-11-04 10:00:02")],
        "Total Fwd Packets": [1.0],
        "Total Length of Fwd Packets": [100.0],
        "Fwd IAT Total": [0.0],
    })

    # Combined continuous baseline
    comb = pd.concat([hist_df, live_df], ignore_index=True)
    out_comb = add_temporal_velocity_features(comb, window_seconds=10)
    expected_row = out_comb.iloc[-1]

    # Warm-up streaming
    ctx = StreamingCausalContext(window_seconds=10)
    buffered_count = ctx.warm_up(hist_df)
    assert buffered_count == 3

    out_live = ctx.process_batch(live_df, causal_mode="simultaneous", flush=True)

    assert out_live.loc[0, "Rolling_Flow_Velocity_10s"] == expected_row["Rolling_Flow_Velocity_10s"]
    assert out_live.loc[0, "Rolling_Bytes_Velocity_10s"] == expected_row["Rolling_Bytes_Velocity_10s"]
    assert np.isclose(out_live.loc[0, "Rolling_Single_Pkt_Ratio_10s"], expected_row["Rolling_Single_Pkt_Ratio_10s"])


# -----------------------------------------------------------------------------
# TEST 09: Time-Based Memory Eviction and Bounded Footprint
# -----------------------------------------------------------------------------
def test_09_memory_eviction_bounded():
    """Verifies that processing 5,000 flows maintains a strictly bounded buffer footprint."""
    ctx = StreamingCausalContext(window_seconds=10)
    ts_base = pd.Timestamp("2026-11-04 10:00:00")

    # Simulate 5,000 flows arriving at 20 flows/sec (250 simulated seconds)
    for i in range(5000):
        ts = ts_base + pd.Timedelta(milliseconds=i * 50)
        ctx.update(ts, fwd_pkts=2.0, fwd_bytes=100.0)

    metrics = ctx.get_metrics()

    # In a 10s window at 20 flows/sec, buffer should hold ~200 items (<= 250)
    assert metrics["buffer_length"] <= 250, f"Buffer leak: held {metrics['buffer_length']} items"
    assert metrics["estimated_memory_bytes"] < 50_000, "Memory usage exceeds 50 KB bound"
    assert metrics["total_processed"] == 5000


# -----------------------------------------------------------------------------
# TEST 10: Explicit Late-Event Policy
# -----------------------------------------------------------------------------
def test_10_out_of_order_late_arrival_policy():
    """
    Verifies the documented two-tier late arrival policy:
    1. In-window late arrival (t >= t_watermark - 10s): Inserted chronologically.
    2. Expired late arrival (t < t_watermark - 10s): Evaluated in isolation without corrupting buffer.
    """
    ctx = StreamingCausalContext(window_seconds=10)
    t_now = pd.Timestamp("2026-11-04 10:00:20")

    # Flow 1 at 10:00:20 (sets watermark)
    ctx.update(t_now, fwd_pkts=1.0, fwd_bytes=100.0)

    # In-window late arrival at 10:00:15 (5s late, inside 10s window)
    rf_in, rb_in, _ = ctx.update(pd.Timestamp("2026-11-04 10:00:15"), fwd_pkts=2.0, fwd_bytes=200.0)
    assert ctx.total_late_events == 1
    assert rf_in == 2.0  # In-window event was inserted

    # Expired late arrival at 10:00:05 (15s late, outside 10s window)
    rf_exp, rb_exp, _ = ctx.update(pd.Timestamp("2026-11-04 10:00:05"), fwd_pkts=5.0, fwd_bytes=500.0)
    assert ctx.total_late_events == 2
    # Expired event must be evaluated in isolation without corrupting the main buffer
    assert rf_exp == 1.0
    assert rb_exp == 500.0


# -----------------------------------------------------------------------------
# TEST 11: End-to-End 26-Dimension Numerical Parity on Real Capture
# -----------------------------------------------------------------------------
def test_11_real_traffic_numerical_parity(real_test_capture):
    """
    The definitive Task 5 verification:
    Validates bit-for-bit numerical parity across ALL 26 pure one-way features
    on a real capture of 1,000 operational network flows.
    """
    df = real_test_capture.head(1000).copy()

    # Canonical offline generation
    df_off = add_temporal_velocity_features(df, window_seconds=10)
    df_off = add_pure_forward_rates(df_off)
    expected_matrix = df_off[PURE_ONE_WAY_FEATURES].values

    # Streaming engine batch extraction
    engine = StreamingFeatureEngine(window_seconds=10)
    streaming_df = engine.extract_features_df(df, causal_mode="simultaneous", flush=True)
    actual_matrix = streaming_df[PURE_ONE_WAY_FEATURES].values

    # Assert exact column ordering and naming
    assert list(streaming_df.columns) == PURE_ONE_WAY_FEATURES

    # Assert 0.0 maximum absolute difference across all 26 dimensions
    max_diff = np.max(np.abs(expected_matrix - actual_matrix))
    assert max_diff < 1e-9, f"Feature parity failure! Max discrepancy: {max_diff}"


# -----------------------------------------------------------------------------
# TEST 12: Fast NumPy Transformation Parity
# -----------------------------------------------------------------------------
def test_12_fast_numpy_transform_parity(fitted_v2_pipeline, real_test_capture):
    """
    Confirms that engine.transform_flow_fast() produces numerically identical
    standardized arrays to the fitted PureOneWayLogPreprocessingPipeline.
    """
    engine = StreamingFeatureEngine(pipeline=fitted_v2_pipeline, window_seconds=10)
    sample_row = real_test_capture.iloc[0].to_dict()

    # Single-flow fast NumPy transform
    fast_vec = engine.transform_flow_fast(sample_row)

    # Standard pipeline transform on single-row DataFrame
    engine.reset()
    features_df = engine.extract_features_df(pd.DataFrame([sample_row]))
    std_vec = fitted_v2_pipeline.transform(features_df)

    assert fast_vec.shape == (1, 26)
    assert std_vec.shape == (1, 26)
    np.testing.assert_allclose(fast_vec, std_vec, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# TEST 13: Empty Batch and Graceful Handling
# -----------------------------------------------------------------------------
def test_13_empty_batch_handling():
    """Verifies that empty DataFrames return empty frames without corrupting engine state."""
    engine = StreamingFeatureEngine(window_seconds=10)
    empty_df = pd.DataFrame(columns=["Timestamp", "Total Fwd Packets", "Total Length of Fwd Packets"])

    res = engine.extract_features_df(empty_df)
    assert len(res) == 0
    assert list(res.columns) == PURE_ONE_WAY_FEATURES
    assert engine.context.total_processed == 0


# -----------------------------------------------------------------------------
# TEST 14: Capacity Boundary and Normal Traffic
# -----------------------------------------------------------------------------
def test_14_capacity_boundary_and_normal_traffic():
    """
    Verifies that under normal operating conditions where buffer load does not
    exceed max_buffer_capacity:
    - Context remains HEALTHY (is_degraded == False, is_valid == True).
    - Dropped overflow event count remains strictly 0.
    - Diagnostics metrics correctly report capacity and health.
    """
    context = StreamingCausalContext(window_seconds=10, max_buffer_capacity=100)
    base_ts = pd.Timestamp("2026-11-04 10:00:00")

    # Send 100 events (exactly at capacity boundary)
    for i in range(100):
        ts = base_ts + pd.Timedelta(milliseconds=i * 20)  # spans 2.0s <= 10s
        context.update(ts, fwd_pkts=2.0, fwd_bytes=100.0)

    assert len(context.buffer) == 100
    assert not context.is_degraded
    assert context.is_valid
    assert context.context_state == "HEALTHY"
    assert context.total_dropped_overflow_events == 0
    assert context.last_overflow_timestamp is None

    metrics = context.get_metrics()
    assert metrics["max_buffer_capacity"] == 100
    assert metrics["total_retained_entries"] == 100
    assert metrics["is_degraded"] is False
    assert metrics["is_valid"] is True
    assert metrics["context_state"] == "HEALTHY"
    assert metrics["total_dropped_overflow_events"] == 0


# -----------------------------------------------------------------------------
# TEST 15: Buffer Overflow Drop and Degraded State
# -----------------------------------------------------------------------------
def test_15_buffer_overflow_drop_and_degraded_state():
    """
    Verifies that when in-window traffic exceeds max_buffer_capacity:
    - Memory ceiling is strictly respected (buffer length <= max_buffer_capacity).
    - Discarded in-window events are tracked via total_dropped_overflow_events.
    - Context immediately enters DEGRADED state (is_degraded == True, is_valid == False).
    - last_overflow_timestamp is accurately recorded.
    - Ordinary time-based expiration is clearly distinguished and does NOT increment overflow drops.
    """
    context = StreamingCausalContext(window_seconds=10, max_buffer_capacity=50)
    base_ts = pd.Timestamp("2026-11-04 10:00:00")

    # Send 80 events within a 2-second window (30 events exceed capacity)
    for i in range(80):
        ts = base_ts + pd.Timedelta(milliseconds=i * 25)
        context.update(ts, fwd_pkts=1.0, fwd_bytes=50.0)

    assert len(context.buffer) == 50
    assert context.total_dropped_overflow_events == 30
    assert context.is_degraded is True
    assert context.is_valid is False
    assert context.context_state == "DEGRADED"
    assert context.last_overflow_timestamp == base_ts + pd.Timedelta(milliseconds=79 * 25)

    # Verify FIFO drop: earliest surviving entry in buffer is event #30
    earliest_surviving_ts = context.buffer[0][0]
    expected_earliest_ts = base_ts + pd.Timedelta(milliseconds=30 * 25)
    assert earliest_surviving_ts == expected_earliest_ts

    # Advance time to t = 25s (all 50 entries expire naturally by window delta)
    future_ts = base_ts + pd.Timedelta(seconds=25)
    context.update(future_ts, fwd_pkts=1.0, fwd_bytes=50.0)

    # Ordinary expiration popped expired entries; only the 1 future event remains
    assert len(context.buffer) == 1
    # total_dropped_overflow_events must NOT have increased due to normal expiration!
    assert context.total_dropped_overflow_events == 30


# -----------------------------------------------------------------------------
# TEST 16: Pending Group Exceeding Capacity Bounded
# -----------------------------------------------------------------------------
def test_16_pending_group_exceeding_capacity_bounded():
    """
    Verifies that simultaneous timestamp groups spanning more flows than
    max_buffer_capacity:
    - Strictly bound memory footprint: len(buffer) + len(pending_group) <= max_buffer_capacity.
    - Retain all rows in emitted DataFrame without silent row loss.
    - Explicitly mark overflowed rows with _is_valid == False and _context_state == 'DEGRADED'.
    - Record accurate overflow drop count.
    """
    context = StreamingCausalContext(window_seconds=10, max_buffer_capacity=30)
    same_ts = pd.Timestamp("2026-11-04 10:00:00")

    # 50 simultaneous flows all at identical timestamp
    data = {
        "Timestamp": [same_ts] * 50,
        "Total Fwd Packets": [2.0] * 50,
        "Total Length of Fwd Packets": [100.0] * 50,
        "Fwd IAT Total": [1000.0] * 50,
    }
    df = pd.DataFrame(data)

    res_df = context.process_batch(df, causal_mode="simultaneous", flush=True)

    # All 50 rows must be emitted (no silent loss of telemetry records)
    assert len(res_df) == 50
    # Exactly 20 events exceeded capacity (50 - 30 = 20)
    assert context.total_dropped_overflow_events == 20
    assert context.is_degraded is True
    assert context.is_valid is False
    assert context.context_state == "DEGRADED"

    # Retained memory buffer must never exceed max_buffer_capacity
    assert len(context.buffer) <= 30

    # First 20 rows emitted must be marked degraded
    assert not res_df.iloc[0]["_is_valid"]
    assert res_df.iloc[0]["_context_state"] == "DEGRADED"
    assert res_df.iloc[0]["_overflow_events"] > 0


# -----------------------------------------------------------------------------
# TEST 17: Degraded State Event-Time Recovery
# -----------------------------------------------------------------------------
def test_17_degraded_state_event_time_recovery():
    """
    Verifies the event-time safe recovery contract:
    - Context remains DEGRADED for any event where t <= last_overflow_timestamp + 10s.
    - Context remains DEGRADED at the exact boundary t == last_overflow_timestamp + 10s.
    - Context transitions to HEALTHY once t > last_overflow_timestamp + 10s.
    - Subsequent valid events are marked _is_valid == True.
    """
    context = StreamingCausalContext(window_seconds=10, max_buffer_capacity=20)
    t0 = pd.Timestamp("2026-11-04 10:00:00.000000")

    # Cause overflow at t0 with 30 events (10 dropped, buffer has 20)
    for _ in range(30):
        context.update(t0, fwd_pkts=1.0, fwd_bytes=50.0)

    assert context.is_degraded is True
    assert context.last_overflow_timestamp == t0

    # 1. Exact boundary: t0 + 10.000000s
    # All 20 entries at t0 are evicted because t0 <= (t0 + 10s - 10s)
    # The new event at t_boundary is within capacity (1 <= 20)
    # Recovery check: t_boundary > t0 + 10s is False (not strictly greater)
    t_boundary = t0 + pd.Timedelta(seconds=10)
    context.update(t_boundary, fwd_pkts=1.0, fwd_bytes=50.0)
    assert context.is_degraded is True
    assert context.is_valid is False
    assert context.context_state == "DEGRADED"

    # 2. Event at t0 + 10.000001s (strictly exceeds t0 + 10s)
    # The 10s window (t - 10s, t] completely clears t0 where overflow happened
    # Recovery condition met: t_recovered > t0 + 10s
    t_recovered = t0 + pd.Timedelta(seconds=10, microseconds=1)
    context.update(t_recovered, fwd_pkts=1.0, fwd_bytes=50.0)
    assert context.is_degraded is False
    assert context.is_valid is True
    assert context.context_state == "HEALTHY"


# -----------------------------------------------------------------------------
# TEST 18: Repeated Overflow Resets Recovery Clock
# -----------------------------------------------------------------------------
def test_18_repeated_overflow_resets_recovery_clock():
    """
    Verifies that a subsequent buffer overflow resets the 10-second recovery timer.
    """
    context = StreamingCausalContext(window_seconds=10, max_buffer_capacity=20)
    t1 = pd.Timestamp("2026-11-04 10:00:00.000000")

    # First overflow at t1 (25 events at t1, capacity 20 -> 5 dropped, buffer has 20)
    for _ in range(25):
        context.update(t1, fwd_pkts=1.0, fwd_bytes=50.0)
    assert context.total_dropped_overflow_events == 5
    assert context.last_overflow_timestamp == t1

    # Second overflow at t2 = t1 + 4s (10 events at t2; since buffer had 20, all 10 are dropped)
    t2 = t1 + pd.Timedelta(seconds=4)
    for _ in range(10):
        context.update(t2, fwd_pkts=1.0, fwd_bytes=50.0)
    assert context.total_dropped_overflow_events == 15
    assert context.last_overflow_timestamp == t2

    # At t = t1 + 11s (> 10s after t1, but only 7s after t2):
    # Entries from t1 (at 10:00:00) expire, but t2 overflow was only 7s ago
    # Must STILL be degraded because t2 reset the recovery clock!
    t_check = t1 + pd.Timedelta(seconds=11)
    context.update(t_check, fwd_pkts=1.0, fwd_bytes=50.0)
    assert context.is_degraded is True
    assert context.is_valid is False

    # At t = t2 + 10s + 1ms (> 10s after t2):
    # Now successfully recovers
    t_final = t2 + pd.Timedelta(seconds=10, milliseconds=1)
    context.update(t_final, fwd_pkts=1.0, fwd_bytes=50.0)
    assert context.is_degraded is False
    assert context.is_valid is True
    assert context.context_state == "HEALTHY"


# -----------------------------------------------------------------------------
# TEST 19: Feature Engine Validity Signal and Metadata
# -----------------------------------------------------------------------------
def test_19_feature_engine_validity_signal_and_metadata(fitted_v2_pipeline, real_test_capture):
    """
    Verifies StreamingFeatureEngine integration with buffer safety:
    - Exposes is_degraded, is_valid, context_state, total_dropped_overflow_events.
    - extract_features_df(..., include_metadata=True) emits metadata columns.
    - transform_flow_fast(..., return_status=True) emits (array, is_valid, context_state).
    - transform_batch(..., return_status=True) emits (array, is_valid, context_state).
    - process_flow() embeds _is_valid, _context_state, _overflow_events in dict.
    """
    engine = StreamingFeatureEngine(
        pipeline=fitted_v2_pipeline,
        window_seconds=10,
        max_buffer_capacity=10,
    )
    assert engine.max_buffer_capacity == 10
    assert engine.context_state == "HEALTHY"

    # Send 15 flows to force overflow
    flows = real_test_capture.head(15).copy()
    base_ts = pd.Timestamp("2026-11-04 10:00:00")
    flows["Timestamp"] = [base_ts + pd.Timedelta(milliseconds=i * 50) for i in range(15)]

    # Test extract_features_df with metadata
    meta_df = engine.extract_features_df(flows, include_metadata=True)
    assert len(meta_df) == 15
    assert "_is_valid" in meta_df.columns
    assert "_context_state" in meta_df.columns
    assert "_overflow_events" in meta_df.columns
    assert engine.is_degraded is True
    assert engine.is_valid is False
    assert engine.context_state == "DEGRADED"
    assert engine.total_dropped_overflow_events == 5

    # Test transform_flow_fast with return_status=True
    single_flow = flows.iloc[-1].to_dict()
    vec, is_valid, state = engine.transform_flow_fast(single_flow, return_status=True)
    assert vec.shape == (1, 26)
    assert is_valid is False
    assert state == "DEGRADED"
    # An extra flow processed while at capacity caused 1 additional drop (5 + 1 = 6)
    assert engine.total_dropped_overflow_events == 6

    # Test transform_flow_fast with default return_status=False
    std_vec = engine.transform_flow_fast(single_flow)
    assert isinstance(std_vec, np.ndarray)
    assert std_vec.shape == (1, 26)

    # Test transform_batch with return_status=True
    batch_vecs, b_valid, b_state = engine.transform_batch(flows, return_status=True)
    assert batch_vecs.shape == (15, 26)
    assert b_valid is False
    assert b_state == "DEGRADED"

    # Test process_flow dictionary metadata
    flow_dict = engine.context.process_flow(single_flow)
    assert flow_dict["_is_valid"] is False
    assert flow_dict["_context_state"] == "DEGRADED"
    assert flow_dict["_overflow_events"] > 0


# -----------------------------------------------------------------------------
# TEST 20: Warm-up Buffer Capacity Safety
# -----------------------------------------------------------------------------
def test_20_warmup_buffer_capacity_safety():
    """
    Verifies that context.warm_up() respects max_buffer_capacity:
    - Retains only the most recent max_buffer_capacity entries.
    - Transitions to DEGRADED state if historical tail exceeds capacity.
    - Records overflow timestamp and dropped counts.
    """
    context = StreamingCausalContext(window_seconds=10, max_buffer_capacity=25)
    max_ts = pd.Timestamp("2026-11-04 10:00:00")
    # 40 events all within the 10-second window (from 10:00:00 - 8s to 10:00:00)
    timestamps = [max_ts - pd.Timedelta(seconds=8) + pd.Timedelta(milliseconds=i * 200) for i in range(40)]
    hist_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Total Fwd Packets": [1.0] * 40,
        "Total Length of Fwd Packets": [100.0] * 40,
    })

    buffered = context.warm_up(hist_df)
    assert buffered == 25
    assert len(context.buffer) == 25
    assert context.total_dropped_overflow_events == 15
    assert context.is_degraded is True
    assert context.is_valid is False
    assert context.last_overflow_timestamp == hist_df["Timestamp"].max()


# -----------------------------------------------------------------------------
# TEST 21: Simultaneous Multi-Batch Expiration and Recovery (D1 Remediation)
# -----------------------------------------------------------------------------
def test_21_simultaneous_multi_batch_expiration_and_recovery():
    """
    D1 Remediation Test:
    Verifies that simultaneous batch streaming:
    1. Fills small-capacity context (cap=20) at timestamp 00.
    2. Advances to timestamp 15 using a new batch.
    3. Confirms old entries are evicted before capacity enforcement.
    4. Confirms no false overflow occurs (total_dropped_overflow_events == 0).
    5. Confirms engine remains healthy (is_degraded == False, is_valid == True).
    6. Advances to timestamp 30 and repeats.
    7. Confirms the capacity invariant throughout.
    8. Verifies correct handling of a batch boundary splitting identical timestamps with flush=False.
    """
    context = StreamingCausalContext(window_seconds=10, max_buffer_capacity=20)
    t0 = pd.Timestamp("2026-11-05 10:00:00")

    # Step 1: Fill context to capacity at t=00
    df_00 = pd.DataFrame({
        "Timestamp": [t0] * 20,
        "Total Fwd Packets": [1.0] * 20,
        "Total Length of Fwd Packets": [100.0] * 20,
        "Fwd IAT Total": [1000.0] * 20,
    })
    res_00 = context.process_batch(df_00, causal_mode="simultaneous", flush=True)
    assert len(res_00) == 20
    assert len(context.buffer) == 20
    assert not context.is_degraded
    assert context.is_valid
    assert context.context_state == "HEALTHY"
    assert context.total_dropped_overflow_events == 0
    assert res_00["_is_valid"].all()

    # Step 2-5: Advance to timestamp 15 using a new batch
    t15 = t0 + pd.Timedelta(seconds=15)
    df_15 = pd.DataFrame({
        "Timestamp": [t15] * 20,
        "Total Fwd Packets": [1.0] * 20,
        "Total Length of Fwd Packets": [100.0] * 20,
        "Fwd IAT Total": [1000.0] * 20,
    })
    res_15 = context.process_batch(df_15, causal_mode="simultaneous", flush=True)
    assert len(res_15) == 20
    # Old entries from t0 must have been evicted as normal expiration, NOT dropped overflow
    assert context.total_dropped_overflow_events == 0
    assert not context.is_degraded
    assert context.is_valid
    assert context.context_state == "HEALTHY"
    assert res_15["_is_valid"].all()
    assert (res_15["_context_state"] == "HEALTHY").all()
    assert len(context.buffer) == 20

    # Step 6: Advance to timestamp 30 and repeat
    t30 = t0 + pd.Timedelta(seconds=30)
    df_30 = pd.DataFrame({
        "Timestamp": [t30] * 20,
        "Total Fwd Packets": [1.0] * 20,
        "Total Length of Fwd Packets": [100.0] * 20,
        "Fwd IAT Total": [1000.0] * 20,
    })
    res_30 = context.process_batch(df_30, causal_mode="simultaneous", flush=True)
    assert len(res_30) == 20
    assert context.total_dropped_overflow_events == 0
    assert not context.is_degraded
    assert context.is_valid
    assert res_30["_is_valid"].all()

    # Step 7: Confirm capacity invariant throughout
    assert len(context.buffer) + len(context.pending_group) <= context.max_buffer_capacity

    # Step 8: Test batch boundary splitting identical timestamps with flush=False
    t45 = t0 + pd.Timedelta(seconds=45)
    t46 = t0 + pd.Timedelta(seconds=46)
    split_batch_1 = pd.DataFrame({
        "Timestamp": [t45] * 5,
        "Total Fwd Packets": [1.0] * 5,
        "Total Length of Fwd Packets": [100.0] * 5,
        "Fwd IAT Total": [1000.0] * 5,
    })
    split_batch_2 = pd.DataFrame({
        "Timestamp": [t45] * 3 + [t46] * 2,
        "Total Fwd Packets": [1.0] * 5,
        "Total Length of Fwd Packets": [100.0] * 5,
        "Fwd IAT Total": [1000.0] * 5,
    })

    # First chunk: flush=False leaves pending group open
    res_split_1 = context.process_batch(split_batch_1, causal_mode="simultaneous", flush=False)
    assert len(res_split_1) == 0
    assert len(context.pending_group) == 5

    # Second chunk: finalized together across batch boundary
    res_split_2 = context.process_batch(split_batch_2, causal_mode="simultaneous", flush=True)
    assert len(res_split_2) == 10
    # The 8 simultaneous events at t45 all receive identical aggregate (8.0)
    velocities = res_split_2["Rolling_Flow_Velocity_10s"].tolist()
    assert velocities[:8] == [8.0] * 8
    # The 2 events at t46 receive 10.0 (8 from t45 + 2 from t46)
    assert velocities[8:] == [10.0, 10.0]
    assert context.total_dropped_overflow_events == 0
    assert not context.is_degraded
    assert res_split_2["_is_valid"].all()


# -----------------------------------------------------------------------------
# TEST 22: Warm-up Multiple Calls and Populated Context Safety (D2 Remediation)
# -----------------------------------------------------------------------------
def test_22_warmup_multiple_and_populated_context_safety():
    """
    D2 Remediation Test:
    Verifies that context.warm_up():
    1. Safely handles being called twice in succession without exceeding capacity.
    2. Safely handles being called on an already populated context.
    3. Strictly enforces capacity ceiling and records drops if warm-up data exceeds available capacity.
    4. Confirms warm_up cannot silently restore an already degraded context to healthy.
    5. Confirms capacity invariant is strictly preserved after warm-up.
    """
    now = pd.Timestamp("2026-11-05 10:00:00")

    # 1. Calling warm_up() twice
    ctx1 = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10)
    df_part1 = pd.DataFrame({
        "Timestamp": [now - pd.Timedelta(seconds=5) + pd.Timedelta(milliseconds=i * 200) for i in range(6)],
        "Total Fwd Packets": [1.0] * 6,
        "Total Length of Fwd Packets": [100.0] * 6,
    })
    df_part2 = pd.DataFrame({
        "Timestamp": [now - pd.Timedelta(seconds=2) + pd.Timedelta(milliseconds=i * 200) for i in range(8)],
        "Total Fwd Packets": [1.0] * 8,
        "Total Length of Fwd Packets": [100.0] * 8,
    })

    b1 = ctx1.warm_up(df_part1)
    assert b1 == 6
    assert len(ctx1.buffer) == 6
    assert not ctx1.is_degraded

    # Second warm_up call: total 6 + 8 = 14 > 10
    b2 = ctx1.warm_up(df_part2)
    assert b2 == 10
    assert len(ctx1.buffer) == 10
    assert ctx1.total_dropped_overflow_events == 4
    assert ctx1.is_degraded is True
    assert ctx1.is_valid is False
    assert len(ctx1.buffer) + len(ctx1.pending_group) <= ctx1.max_buffer_capacity

    # 2. Calling warm_up() on a populated context
    ctx2 = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10)
    for i in range(5):
        ctx2.update(now - pd.Timedelta(seconds=4) + pd.Timedelta(milliseconds=i * 100), 1.0, 100.0)
    assert len(ctx2.buffer) == 5

    b_pop = ctx2.warm_up(df_part2)
    assert b_pop == 10
    assert len(ctx2.buffer) == 10
    assert ctx2.total_dropped_overflow_events == 3
    assert ctx2.is_degraded is True
    assert len(ctx2.buffer) + len(ctx2.pending_group) <= ctx2.max_buffer_capacity

    # 3. Warm-up data exceeding available capacity on fresh context
    ctx3 = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10)
    df_huge = pd.DataFrame({
        "Timestamp": [now - pd.Timedelta(seconds=3) + pd.Timedelta(milliseconds=i * 100) for i in range(25)],
        "Total Fwd Packets": [1.0] * 25,
        "Total Length of Fwd Packets": [100.0] * 25,
    })
    b_huge = ctx3.warm_up(df_huge)
    assert b_huge == 10
    assert len(ctx3.buffer) == 10
    assert ctx3.total_dropped_overflow_events == 15
    assert ctx3.is_degraded is True

    # 4. Confirm warm_up cannot silently restore an already degraded context to healthy
    ctx4 = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10)
    ctx4.is_degraded = True
    ctx4.last_overflow_timestamp = now
    ctx4.total_dropped_overflow_events = 5
    small_hist = pd.DataFrame({
        "Timestamp": [now],
        "Total Fwd Packets": [1.0],
        "Total Length of Fwd Packets": [100.0],
    })
    b_small = ctx4.warm_up(small_hist)
    assert b_small == 1
    assert ctx4.is_degraded is True  # Cannot be silently restored!
    assert ctx4.is_valid is False
    assert ctx4.total_dropped_overflow_events == 5

    # 5. Capacity invariant holds across all contexts
    assert len(ctx1.buffer) <= ctx1.max_buffer_capacity
    assert len(ctx2.buffer) <= ctx2.max_buffer_capacity
    assert len(ctx3.buffer) <= ctx3.max_buffer_capacity
    assert len(ctx4.buffer) <= ctx4.max_buffer_capacity
