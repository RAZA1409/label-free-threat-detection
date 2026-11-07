"""
tests/test_pure_one_way_features.py
Comprehensive automated test suite for SIH26145 Phase 1 Pure One-Way Feature Specification.
Tests:
- TEST 01: Feature specification validation
- TEST 02: Reverse-direction exclusion
- TEST 03: Forward-only rate calculation
- TEST 04: Directional invariance (CRITICAL)
- TEST 05: Feature order consistency
- TEST 06: Preprocessing robustness (NaNs, infinities, negative sentinels, extreme values)
- TEST 07: Rolling feature correctness & causal invariance
- TEST 08: Pipeline output shape and validity
- TEST 09: Feature reproducibility
- TEST 10: Model compatibility guard
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from features import (
    PURE_ONE_WAY_BASE_FEATURES,
    PURE_ONE_WAY_TEMPORAL_FEATURES,
    PURE_ONE_WAY_FEATURES,
    add_pure_forward_rates,
    add_temporal_velocity_features,
    PureOneWayPreprocessingPipeline,
    PureOneWayLogPreprocessingPipeline,
    check_model_compatibility,
    AUGMENTED_ONE_WAY_FEATURES,
)


# -----------------------------------------------------------------------------
# TEST 01: Feature specification validation
# -----------------------------------------------------------------------------
def test_01_feature_specification_validation():
    """Verify that all selected pure features exist, are non-empty, and contain no duplicates."""
    assert len(PURE_ONE_WAY_FEATURES) == 26, f"Expected 26 features, got {len(PURE_ONE_WAY_FEATURES)}"
    assert len(set(PURE_ONE_WAY_FEATURES)) == len(PURE_ONE_WAY_FEATURES), "Duplicate features detected in specification!"

    # Confirm exact sub-counts
    assert len(PURE_ONE_WAY_BASE_FEATURES) == 23, "Expected 23 base features"
    assert len(PURE_ONE_WAY_TEMPORAL_FEATURES) == 3, "Expected 3 temporal features"
    assert PURE_ONE_WAY_FEATURES == PURE_ONE_WAY_BASE_FEATURES + PURE_ONE_WAY_TEMPORAL_FEATURES


# -----------------------------------------------------------------------------
# TEST 02: Reverse-direction exclusion
# -----------------------------------------------------------------------------
def test_02_reverse_direction_exclusion():
    """Confirm that no confirmed backward-only or bidirectional feature remains in pure spec."""
    prohibited_features = [
        "Flow Bytes/s",          # Leaked Bwd Bytes in numerator
        "Flow Duration",         # Leaked reverse packet end timestamp in 77.4% of flows
        "Flow Packets/s",        # Bidirectional packet rate
        "Fwd Packets/s",         # Divided by bidirectional Flow Duration
        "Total Backward Packets",
        "Total Length of Bwd Packets",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Bwd PSH Flags",
        "Bwd URG Flags",
        "Bwd Header Length",
        "Bwd Packets/s",
        "Avg Bwd Segment Size",
        "Init_Win_bytes_backward",
        "Down/Up Ratio",
        "Average Packet Size",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "ACK Flag Count",        # Leaked reverse server ACKs
        "PSH Flag Count",        # Flow-wide PSH
        "URG Flag Count",        # Flow-wide URG
        "FIN Flag Count",        # Flow-wide FIN
        "RST Flag Count",        # Flow-wide RST
    ]

    for prohibited in prohibited_features:
        assert prohibited not in PURE_ONE_WAY_FEATURES, f"Violation: Prohibited feature '{prohibited}' found in PURE_ONE_WAY_FEATURES!"


# -----------------------------------------------------------------------------
# TEST 03: Forward-only rate calculation
# -----------------------------------------------------------------------------
def test_03_forward_only_rate_calculation():
    """Independently calculate forward byte & packet rate, verify units and edge cases."""
    test_cases = pd.DataFrame([
        # Case A: Standard flow (10 packets, 5000 bytes, 1.0 second duration = 1,000,000 us)
        {"Total Length of Fwd Packets": 5000.0, "Total Fwd Packets": 10.0, "Fwd IAT Total": 1_000_000.0},
        # Case B: Fast flow (2 packets, 120 bytes, 100 us duration = 0.0001s)
        {"Total Length of Fwd Packets": 120.0, "Total Fwd Packets": 2.0, "Fwd IAT Total": 100.0},
        # Case C: Zero-duration single packet flow (1 packet, 64 bytes, 0 us duration)
        {"Total Length of Fwd Packets": 64.0, "Total Fwd Packets": 1.0, "Fwd IAT Total": 0.0},
        # Case D: Zero-bytes probe flow (1 packet, 0 bytes, 0 us duration)
        {"Total Length of Fwd Packets": 0.0, "Total Fwd Packets": 1.0, "Fwd IAT Total": 0.0},
    ])

    res = add_pure_forward_rates(test_cases)

    # Expected calculations:
    # Case A: 5000 bytes / 1.0s = 5000.0 B/s; 10 pkts / 1.0s = 10.0 pkts/s
    assert np.isclose(res.loc[0, "Fwd_Bytes_Per_Sec"], 5000.0), f"Expected 5000.0, got {res.loc[0, 'Fwd_Bytes_Per_Sec']}"
    assert np.isclose(res.loc[0, "Fwd_Packets_Per_Sec"], 10.0), f"Expected 10.0, got {res.loc[0, 'Fwd_Packets_Per_Sec']}"

    # Case B: 120 bytes / 0.0001s = 1,200,000.0 B/s; 2 pkts / 0.0001s = 20,000.0 pkts/s
    assert np.isclose(res.loc[1, "Fwd_Bytes_Per_Sec"], 1_200_000.0)
    assert np.isclose(res.loc[1, "Fwd_Packets_Per_Sec"], 20_000.0)

    # Case C (Zero duration): Should safely be 0.0, NOT inf or NaN!
    assert res.loc[2, "Fwd_Bytes_Per_Sec"] == 0.0
    assert res.loc[2, "Fwd_Packets_Per_Sec"] == 0.0
    assert not np.isnan(res.loc[2, "Fwd_Bytes_Per_Sec"])
    assert not np.isinf(res.loc[2, "Fwd_Bytes_Per_Sec"])

    # Case D: Zero bytes zero duration
    assert res.loc[3, "Fwd_Bytes_Per_Sec"] == 0.0
    assert res.loc[3, "Fwd_Packets_Per_Sec"] == 0.0


# -----------------------------------------------------------------------------
# TEST 04: Directional invariance (CRITICAL)
# -----------------------------------------------------------------------------
def test_04_directional_invariance():
    """
    CRITICAL TEST:
    Construct two synthetic flows with IDENTICAL forward characteristics.
    Flow 1: Benign web response received (100 backward packets, 150,000 backward bytes, server ACKs, server FIN).
    Flow 2: Unreachable server or dropped return path (0 backward packets, 0 backward bytes, 0 return flags).

    The pure one-way pipeline MUST produce 100% IDENTICAL feature vectors for both flows.
    """
    base_fwd = {
        "Timestamp": pd.Timestamp("2026-11-04 10:00:00"),
        "Total Fwd Packets": 5.0,
        "Total Length of Fwd Packets": 1500.0,
        "act_data_pkt_fwd": 3.0,
        "Subflow Fwd Packets": 5.0,
        "Subflow Fwd Bytes": 1500.0,
        "Fwd Header Length": 160.0,
        "min_seg_size_forward": 32.0,
        "Avg Fwd Segment Size": 300.0,
        "Init_Win_bytes_forward": 29200.0,
        "Fwd Packet Length Max": 500.0,
        "Fwd Packet Length Min": 40.0,
        "Fwd Packet Length Mean": 300.0,
        "Fwd Packet Length Std": 120.0,
        "Fwd IAT Total": 500_000.0,
        "Fwd IAT Mean": 125_000.0,
        "Fwd IAT Std": 50_000.0,
        "Fwd IAT Max": 200_000.0,
        "Fwd IAT Min": 10_000.0,
        "Fwd PSH Flags": 1.0,
        "Fwd URG Flags": 0.0,
        "SYN Flag Count": 1.0,
    }

    # Flow A: Huge reverse traffic
    flow_a = dict(base_fwd)
    flow_a.update({
        "Total Backward Packets": 100.0,
        "Total Length of Bwd Packets": 150_000.0,
        "Bwd Packet Length Mean": 1500.0,
        "Bwd IAT Total": 480_000.0,
        "Flow Duration": 950_000.0,  # Leaked reverse duration!
        "Flow Bytes/s": 159_473.68,   # Leaked reverse bytes!
        "ACK Flag Count": 1.0,       # Leaked reverse ACK!
        "PSH Flag Count": 15.0,      # Leaked reverse PSH!
        "FIN Flag Count": 1.0,       # Leaked reverse FIN!
    })

    # Flow B: Zero reverse traffic
    flow_b = dict(base_fwd)
    flow_b.update({
        "Total Backward Packets": 0.0,
        "Total Length of Bwd Packets": 0.0,
        "Bwd Packet Length Mean": 0.0,
        "Bwd IAT Total": 0.0,
        "Flow Duration": 500_000.0,
        "Flow Bytes/s": 3_000.0,
        "ACK Flag Count": 0.0,
        "PSH Flag Count": 1.0,
        "FIN Flag Count": 0.0,
    })

    df = pd.DataFrame([flow_a, flow_b])
    df = add_temporal_velocity_features(df)

    pipeline = PureOneWayPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    pipeline.fit(df)
    vecs = pipeline.transform(df)

    # Verify output vectors are exactly identical
    np.testing.assert_allclose(
        vecs[0], vecs[1], rtol=1e-5, atol=1e-5,
        err_msg="Directional invariance FAILED: Pure one-way pipeline produced different features based on reverse traffic!"
    )


# -----------------------------------------------------------------------------
# TEST 05: Feature order consistency
# -----------------------------------------------------------------------------
def test_05_feature_order_consistency():
    """Verify that training and inference use identical feature names and order."""
    pipe = PureOneWayPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    assert pipe.features == PURE_ONE_WAY_FEATURES

    # Check that Log pipeline also preserves identical order
    log_pipe = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    assert log_pipe.features == PURE_ONE_WAY_FEATURES


# -----------------------------------------------------------------------------
# TEST 06: Preprocessing robustness (NaNs, infinities, negative sentinels, extremes)
# -----------------------------------------------------------------------------
def test_06_preprocessing_robustness():
    """Test pipeline handling of NaNs, infinities, negative sentinels, and extreme values."""
    data = {f: [10.0, np.nan, np.inf, -np.inf, -1.0, 1e9] for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]}
    data["Timestamp"] = pd.date_range("2026-11-04 10:00:00", periods=6, freq="1s")

    df = pd.DataFrame(data)
    df = add_temporal_velocity_features(df)

    pipe = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    pipe.fit(df)
    transformed = pipe.transform(df)

    assert transformed.shape == (6, 26)
    assert not np.isnan(transformed).any(), "NaN values found in transformed output!"
    assert not np.isinf(transformed).any(), "Infinite values found in transformed output!"


# -----------------------------------------------------------------------------
# TEST 07: Rolling feature correctness & causal invariance
# -----------------------------------------------------------------------------
def test_07_rolling_feature_correctness():
    """Verify rolling-window calculations use strictly causal backward-looking windows (t - window, t]."""
    timestamps = [
        pd.Timestamp("2026-11-04 10:00:00"),
        pd.Timestamp("2026-11-04 10:00:02"),
        pd.Timestamp("2026-11-04 10:00:05"),
        pd.Timestamp("2026-11-04 10:00:15"), # Outside previous (t-10s, t] window
    ]
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Total Fwd Packets": [1.0, 1.0, 5.0, 2.0],
        "Total Length of Fwd Packets": [64.0, 64.0, 2500.0, 120.0],
    })

    augmented = add_temporal_velocity_features(df, window_seconds=10)

    # Event 0 (10:00:00): Strictly isolated at start of stream -> 1 flow, 64 bytes, 1.0 single ratio
    assert augmented.loc[0, "Rolling_Flow_Velocity_10s"] == 1.0
    assert augmented.loc[0, "Rolling_Bytes_Velocity_10s"] == 64.0
    assert augmented.loc[0, "Rolling_Single_Pkt_Ratio_10s"] == 1.0

    # Event 1 (10:00:02): 2 flows in (09:59:52, 10:00:02] -> 2 flows, 128 bytes, 1.0 single ratio
    assert augmented.loc[1, "Rolling_Flow_Velocity_10s"] == 2.0
    assert augmented.loc[1, "Rolling_Bytes_Velocity_10s"] == 128.0
    assert augmented.loc[1, "Rolling_Single_Pkt_Ratio_10s"] == 1.0

    # Event 2 (10:00:05): 3 flows in (09:59:55, 10:00:05] -> 3 flows, sum bytes = 2628, 2 single packet flows (ratio = 2/3 = 0.6667)
    assert augmented.loc[2, "Rolling_Flow_Velocity_10s"] == 3.0
    assert augmented.loc[2, "Rolling_Bytes_Velocity_10s"] == 2628.0
    assert np.isclose(augmented.loc[2, "Rolling_Single_Pkt_Ratio_10s"], 2.0 / 3.0)

    # Event 3 (10:00:15): Window is (10:00:05, 10:00:15] -> 1 flow, 120 bytes, 0 single packet flows
    assert augmented.loc[3, "Rolling_Flow_Velocity_10s"] == 1.0
    assert augmented.loc[3, "Rolling_Bytes_Velocity_10s"] == 120.0
    assert augmented.loc[3, "Rolling_Single_Pkt_Ratio_10s"] == 0.0


# -----------------------------------------------------------------------------
# TEST 08: Pipeline output shape and validity
# -----------------------------------------------------------------------------
def test_08_pipeline_output_shape_and_validity():
    """Confirm transformed output has exactly 26 dimensions and finite float values."""
    n_samples = 25
    data = {f: np.random.uniform(1.0, 500.0, size=n_samples) for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]}
    data["Timestamp"] = pd.date_range("2026-11-04 10:00:00", periods=n_samples, freq="500ms")

    df = pd.DataFrame(data)
    df = add_temporal_velocity_features(df)

    pipeline = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    pipeline.fit(df)
    out = pipeline.transform(df)

    assert out.shape == (n_samples, 26)
    assert out.dtype == np.float64


# -----------------------------------------------------------------------------
# TEST 09: Feature reproducibility
# -----------------------------------------------------------------------------
def test_09_feature_reproducibility():
    """Confirm identical inputs produce identical feature outputs across multiple runs."""
    data = {f: np.random.uniform(5.0, 200.0, size=10) for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]}
    data["Timestamp"] = pd.date_range("2026-11-04 10:00:00", periods=10, freq="1s")
    df = pd.DataFrame(data)
    df = add_temporal_velocity_features(df)

    pipe = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    pipe.fit(df)

    run_1 = pipe.transform(df)
    run_2 = pipe.transform(df)

    np.testing.assert_array_equal(run_1, run_2)


# -----------------------------------------------------------------------------
# TEST 10: Model compatibility guard
# -----------------------------------------------------------------------------
def test_10_model_compatibility_guard():
    """Verify that the new 26-dim feature pipeline cannot be accidentally passed into incompatible 32-dim legacy models."""
    class DummyLegacyModel:
        n_features_in_ = 32

    legacy_model = DummyLegacyModel()
    pipe_pure = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)

    is_compat, reason = check_model_compatibility(legacy_model, pipe_pure, PURE_ONE_WAY_FEATURES)
    assert not is_compat, "Guard failed: Pure 26-dim pipeline was incorrectly declared compatible with 32-dim model!"
    assert "32 features, but specification has 26" in reason


# -----------------------------------------------------------------------------
# TEST 11: Temporal causality and non-leakage of future events
# -----------------------------------------------------------------------------
def test_11_temporal_causality_no_future_leakage():
    """
    Rigorously verifies that an event occurring at timestamp t is strictly causally isolated
    from any future events occurring at t' > t, even if those future events fall within
    the same 10-second calendar epoch.
    """
    # Event E0 at t = 10:00:01
    df_isolated = pd.DataFrame({
        "Timestamp": [pd.Timestamp("2026-11-04 10:00:01")],
        "Total Fwd Packets": [2.0],
        "Total Length of Fwd Packets": [200.0],
    })
    res_isolated = add_temporal_velocity_features(df_isolated, window_seconds=10)
    iso_flow = res_isolated.loc[0, "Rolling_Flow_Velocity_10s"]
    iso_bytes = res_isolated.loc[0, "Rolling_Bytes_Velocity_10s"]
    iso_single = res_isolated.loc[0, "Rolling_Single_Pkt_Ratio_10s"]

    # Now append multiple future events occurring at t=10:00:05, 10:00:08 (same calendar epoch [10:00:00, 10:00:10))
    df_augmented = pd.DataFrame({
        "Timestamp": [
            pd.Timestamp("2026-11-04 10:00:01"),
            pd.Timestamp("2026-11-04 10:00:05"),
            pd.Timestamp("2026-11-04 10:00:08"),
        ],
        "Total Fwd Packets": [2.0, 1.0, 1.0],
        "Total Length of Fwd Packets": [200.0, 50.0, 100.0],
    })
    res_augmented = add_temporal_velocity_features(df_augmented, window_seconds=10)

    # Event 0's metrics MUST remain 100% identical and unchanged despite future arrivals
    assert res_augmented.loc[0, "Rolling_Flow_Velocity_10s"] == iso_flow, (
        f"Temporal leakage detected! Flow velocity changed from {iso_flow} to {res_augmented.loc[0, 'Rolling_Flow_Velocity_10s']}"
    )
    assert res_augmented.loc[0, "Rolling_Bytes_Velocity_10s"] == iso_bytes, (
        f"Temporal leakage detected! Byte velocity changed from {iso_bytes} to {res_augmented.loc[0, 'Rolling_Bytes_Velocity_10s']}"
    )
    assert res_augmented.loc[0, "Rolling_Single_Pkt_Ratio_10s"] == iso_single, (
        f"Temporal leakage detected! Single packet ratio changed from {iso_single} to {res_augmented.loc[0, 'Rolling_Single_Pkt_Ratio_10s']}"
    )

    # Verify out-of-order input handling: event E0 provided at the end of the dataframe
    df_unordered = pd.DataFrame({
        "Timestamp": [
            pd.Timestamp("2026-11-04 10:00:05"),
            pd.Timestamp("2026-11-04 10:00:08"),
            pd.Timestamp("2026-11-04 10:00:01"), # E0 is at index 2
        ],
        "Total Fwd Packets": [1.0, 1.0, 2.0],
        "Total Length of Fwd Packets": [50.0, 100.0, 200.0],
    })
    res_unordered = add_temporal_velocity_features(df_unordered, window_seconds=10)

    # The event at index 2 (10:00:01) must still strictly match iso values
    assert res_unordered.loc[2, "Rolling_Flow_Velocity_10s"] == iso_flow
    assert res_unordered.loc[2, "Rolling_Bytes_Velocity_10s"] == iso_bytes
    assert res_unordered.loc[2, "Rolling_Single_Pkt_Ratio_10s"] == iso_single


# -----------------------------------------------------------------------------
# TEST 12: Cross-partition boundary policy & historical continuity
# -----------------------------------------------------------------------------
def test_12_cross_partition_boundary_policy():
    """
    Verifies that:
    1. Independent partition calculation has a controlled cold-start transient at boundary.
    2. Events occurring >= 10s after boundary are 100% identical between isolated and continuous stream.
    3. When optional historical context is supplied, boundary events seamlessly match full continuous stream.
    """
    times_p1 = [
        pd.Timestamp("2026-11-04 10:00:00"),
        pd.Timestamp("2026-11-04 10:00:05"),
        pd.Timestamp("2026-11-04 10:00:08"),  # 2s before boundary
    ]
    df_p1 = pd.DataFrame({
        "Timestamp": times_p1,
        "Total Fwd Packets": [1.0, 2.0, 3.0],
        "Total Length of Fwd Packets": [64.0, 128.0, 256.0],
    })

    times_p2 = [
        pd.Timestamp("2026-11-04 10:00:10"),  # 2s after boundary (within 10s of p1[2])
        pd.Timestamp("2026-11-04 10:00:12"),  # 4s after boundary
        pd.Timestamp("2026-11-04 10:00:25"),  # 15s after boundary (> 10s after p1, independent of p1)
    ]
    df_p2 = pd.DataFrame({
        "Timestamp": times_p2,
        "Total Fwd Packets": [4.0, 5.0, 6.0],
        "Total Length of Fwd Packets": [512.0, 1024.0, 2048.0],
    })

    # Continuous stream
    df_continuous = pd.concat([df_p1, df_p2], ignore_index=True)
    out_continuous = add_temporal_velocity_features(df_continuous, window_seconds=10)

    # Isolated partition 2 (cold start at boundary)
    out_isolated_p2 = add_temporal_velocity_features(df_p2, window_seconds=10)

    # Event >= 10s after boundary (index 2 of p2) MUST be identical in isolated vs continuous
    assert out_isolated_p2.loc[2, "Rolling_Flow_Velocity_10s"] == out_continuous.loc[5, "Rolling_Flow_Velocity_10s"]
    assert out_isolated_p2.loc[2, "Rolling_Bytes_Velocity_10s"] == out_continuous.loc[5, "Rolling_Bytes_Velocity_10s"]

    # Partition 2 with historical context from partition 1
    out_hist_p2 = add_temporal_velocity_features(df_p2, window_seconds=10, historical_context=df_p1)

    # All rows in out_hist_p2 must 100% match out_continuous rows 3, 4, 5
    np.testing.assert_array_equal(
        out_hist_p2["Rolling_Flow_Velocity_10s"].values,
        out_continuous.iloc[3:]["Rolling_Flow_Velocity_10s"].values,
    )
    np.testing.assert_array_equal(
        out_hist_p2["Rolling_Bytes_Velocity_10s"].values,
        out_continuous.iloc[3:]["Rolling_Bytes_Velocity_10s"].values,
    )
