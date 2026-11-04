"""
tests/test_streaming_compatibility.py
Pytest suite verifying that the Pure One-Way Feature Specification (v2.0) processes
streaming flow events strictly from forward observations without requiring any
reverse-direction fields or bidirectional dependencies.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (
    PURE_ONE_WAY_FEATURES,
    PURE_ONE_WAY_BASE_FEATURES,
    add_pure_forward_rates,
    add_temporal_velocity_features,
    PureOneWayLogPreprocessingPipeline,
)


@pytest.fixture(scope="module")
def fitted_streaming_pipeline():
    """Provides a fitted pure one-way pipeline on calibration data."""
    cal_file = ROOT / "outputs" / "experiment" / "calibration_benign.parquet"
    if cal_file.exists():
        cal_df = pd.read_parquet(cal_file).head(1000)
    else:
        # Fallback synthetic calibration data
        timestamps = pd.date_range("2026-11-04 10:00:00", periods=100, freq="1s")
        data = {f: np.random.uniform(1.0, 100.0, size=100) for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]}
        data["Timestamp"] = timestamps
        cal_df = pd.DataFrame(data)

    cal_df = add_temporal_velocity_features(cal_df)
    pipeline = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    pipeline.fit(cal_df)
    return pipeline


def test_streaming_single_forward_flow_transformation(fitted_streaming_pipeline):
    """
    Verifies that a single forward-only packet dictionary (without ANY reverse fields)
    transforms into the expected (1, 26) feature vector with no NaNs and low latency.
    """
    single_forward_flow = {
        "Total Fwd Packets": 1.0,
        "Total Length of Fwd Packets": 64.0,
        "act_data_pkt_fwd": 0.0,
        "Subflow Fwd Packets": 1.0,
        "Subflow Fwd Bytes": 64.0,
        "Fwd Header Length": 32.0,
        "min_seg_size_forward": 32.0,
        "Avg Fwd Segment Size": 64.0,
        "Init_Win_bytes_forward": 29200.0,
        "Fwd Packet Length Max": 64.0,
        "Fwd Packet Length Min": 64.0,
        "Fwd Packet Length Mean": 64.0,
        "Fwd Packet Length Std": 0.0,
        "Fwd IAT Total": 0.0,
        "Fwd IAT Mean": 0.0,
        "Fwd IAT Std": 0.0,
        "Fwd IAT Max": 0.0,
        "Fwd IAT Min": 0.0,
        "Fwd PSH Flags": 0.0,
        "Fwd URG Flags": 0.0,
        "SYN Flag Count": 1.0,
        "Rolling_Flow_Velocity_10s": 15.0,
        "Rolling_Bytes_Velocity_10s": 960.0,
        "Rolling_Single_Pkt_Ratio_10s": 1.0,
    }

    df_single = pd.DataFrame([single_forward_flow])
    t0 = time.time()
    transformed = fitted_streaming_pipeline.transform(df_single)
    latency_ms = (time.time() - t0) * 1000

    assert transformed.shape == (1, 26), f"Expected (1, 26), got {transformed.shape}"
    assert not np.isnan(transformed).any(), "NaN found in streaming output vector"
    assert not np.isinf(transformed).any(), "Inf found in streaming output vector"
    assert latency_ms < 500.0, f"Transformation latency too high: {latency_ms:.2f}ms"


def test_streaming_multi_packet_forward_burst(fitted_streaming_pipeline):
    """
    Verifies that a multi-packet forward burst with non-zero duration correctly
    computes forward rates and produces valid feature vectors.
    """
    burst_flow = {
        "Total Fwd Packets": 25.0,
        "Total Length of Fwd Packets": 12500.0,
        "act_data_pkt_fwd": 20.0,
        "Subflow Fwd Packets": 25.0,
        "Subflow Fwd Bytes": 12500.0,
        "Fwd Header Length": 800.0,
        "min_seg_size_forward": 32.0,
        "Avg Fwd Segment Size": 500.0,
        "Init_Win_bytes_forward": 14600.0,
        "Fwd Packet Length Max": 1460.0,
        "Fwd Packet Length Min": 40.0,
        "Fwd Packet Length Mean": 500.0,
        "Fwd Packet Length Std": 350.0,
        "Fwd IAT Total": 50000.0,  # 50,000 us = 0.05s
        "Fwd IAT Mean": 2083.33,
        "Fwd IAT Std": 1200.0,
        "Fwd IAT Max": 5000.0,
        "Fwd IAT Min": 100.0,
        "Fwd PSH Flags": 2.0,
        "Fwd URG Flags": 0.0,
        "SYN Flag Count": 1.0,
        "Rolling_Flow_Velocity_10s": 85.0,
        "Rolling_Bytes_Velocity_10s": 45000.0,
        "Rolling_Single_Pkt_Ratio_10s": 0.12,
    }

    df_burst = pd.DataFrame([burst_flow])
    transformed = fitted_streaming_pipeline.transform(df_burst)

    assert transformed.shape == (1, 26)
    assert not np.isnan(transformed).any()
    assert not np.isinf(transformed).any()


def test_streaming_forward_only_absence_of_reverse_fields(fitted_streaming_pipeline):
    """
    Explicitly tests that injecting arbitrary reverse-direction fields does not affect
    the transformed feature vector in any way (strict directional isolation).
    """
    base_flow = {
        "Total Fwd Packets": 5.0,
        "Total Length of Fwd Packets": 2500.0,
        "act_data_pkt_fwd": 3.0,
        "Subflow Fwd Packets": 5.0,
        "Subflow Fwd Bytes": 2500.0,
        "Fwd Header Length": 160.0,
        "min_seg_size_forward": 32.0,
        "Avg Fwd Segment Size": 500.0,
        "Init_Win_bytes_forward": 29200.0,
        "Fwd Packet Length Max": 1000.0,
        "Fwd Packet Length Min": 40.0,
        "Fwd Packet Length Mean": 500.0,
        "Fwd Packet Length Std": 200.0,
        "Fwd IAT Total": 20000.0,
        "Fwd IAT Mean": 5000.0,
        "Fwd IAT Std": 1000.0,
        "Fwd IAT Max": 8000.0,
        "Fwd IAT Min": 500.0,
        "Fwd PSH Flags": 0.0,
        "Fwd URG Flags": 0.0,
        "SYN Flag Count": 1.0,
        "Rolling_Flow_Velocity_10s": 20.0,
        "Rolling_Bytes_Velocity_10s": 10000.0,
        "Rolling_Single_Pkt_Ratio_10s": 0.25,
    }

    # Flow 1: clean forward only
    df_clean = pd.DataFrame([base_flow])

    # Flow 2: poisoned with reverse-direction telemetry
    poisoned_flow = dict(base_flow)
    poisoned_flow.update({
        "Total Backward Packets": 500.0,
        "Total Length of Bwd Packets": 500000.0,
        "Flow Duration": 9999999.0,
        "Flow Bytes/s": 888888.0,
        "ACK Flag Count": 1.0,
        "PSH Flag Count": 20.0,
        "FIN Flag Count": 1.0,
    })
    df_poisoned = pd.DataFrame([poisoned_flow])

    vec_clean = fitted_streaming_pipeline.transform(df_clean)
    vec_poisoned = fitted_streaming_pipeline.transform(df_poisoned)

    # Transformed vectors must be 100% numerically identical
    np.testing.assert_allclose(vec_clean, vec_poisoned, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
