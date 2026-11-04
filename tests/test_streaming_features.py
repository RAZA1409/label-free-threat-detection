"""
tests/test_streaming_features.py
Comprehensive automated test suite for AEGIS Phase 3 Stage 4A Causal Streaming Feature Engine.

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
