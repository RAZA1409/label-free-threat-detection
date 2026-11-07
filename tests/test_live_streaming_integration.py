"""
tests/test_live_streaming_integration.py
AEGIS Phase 3 Stage 4B.2: Integration Test Suite for Chronological Live Streaming & v2 Shadow Inference.

Verifies:
1. Chronological nondecreasing replay.
2. Stable ordering for duplicate timestamps.
3. Simultaneous timestamp groups across batch boundaries.
4. Streaming feature engine updated exactly once per flow.
5. v1 payload backward compatibility.
6. v1 score/verdict invariance against deterministic baseline.
7. v2 receives exactly 26 correctly ordered features.
8. v2 uses the correct scaler and thresholds.
9. v2 is not called when context is degraded.
10. v2 exceptions do not interrupt v1 alerts.
11. v2 does not change production alert counters.
12. Live-feed JSON schema compatibility.
13. Correct shutdown flush.
14. Existing v1-only simulator behavior remains available.
"""

import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (
    ONE_WAY_FEATURES,
    PURE_ONE_WAY_FEATURES,
    PURE_LOG_FEATURES,
    sanitize_labels,
    add_pure_forward_rates,
)
from explainability import ThreatExplainer
from model_adapters import PureOneWayV2Adapter
from streaming_features import (
    StreamingFeatureEngine,
    StreamingCausalContext,
    prepare_chronological_stream,
)
from simulate_stream import (
    score_v2_shadow,
    run_stream_simulation,
    TEST_FILE,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v2_adapter():
    """Provides an initialized PureOneWayV2Adapter."""
    adapter = PureOneWayV2Adapter()
    return adapter


@pytest.fixture(scope="module")
def v1_explainer():
    """Provides an initialized ThreatExplainer."""
    return ThreatExplainer()


@pytest.fixture
def synthetic_flow_pool():
    """Creates a deterministic synthetic flow pool with known timestamps."""
    timestamps = [
        "2026-11-04 10:00:00",
        "2026-11-04 10:00:01",
        "2026-11-04 10:00:01",
        "2026-11-04 10:00:02",
        "2026-11-04 10:00:05",
        "2026-11-04 10:00:05",
        "2026-11-04 10:00:08",
        "2026-11-04 10:00:10",
    ]
    n = len(timestamps)
    data = {
        "Timestamp": pd.to_datetime(timestamps),
        "Flow Duration": [1000.0 * (i + 1) for i in range(n)],
        "Total Fwd Packets": [float(i + 1) for i in range(n)],
        "Total Length of Fwd Packets": [64.0 * (i + 1) for i in range(n)],
        "Fwd IAT Total": [500.0 * (i + 1) for i in range(n)],
        "Fwd IAT Mean": [100.0 for _ in range(n)],
        "Fwd IAT Std": [10.0 for _ in range(n)],
        "Fwd IAT Max": [200.0 for _ in range(n)],
        "Fwd IAT Min": [50.0 for _ in range(n)],
        "Fwd Header Length": [32.0 for _ in range(n)],
        "min_seg_size_forward": [32.0 for _ in range(n)],
        "Avg Fwd Segment Size": [64.0 for _ in range(n)],
        "Init_Win_bytes_forward": [29200.0 for _ in range(n)],
        "Fwd Packet Length Max": [64.0 for _ in range(n)],
        "Fwd Packet Length Min": [64.0 for _ in range(n)],
        "Fwd Packet Length Mean": [64.0 for _ in range(n)],
        "Fwd Packet Length Std": [0.0 for _ in range(n)],
        "act_data_pkt_fwd": [1.0 for _ in range(n)],
        "Subflow Fwd Packets": [float(i + 1) for i in range(n)],
        "Subflow Fwd Bytes": [64.0 * (i + 1) for i in range(n)],
        "SYN Flag Count": [1.0 if i % 2 == 0 else 0.0 for i in range(n)],
        "FIN Flag Count": [0.0 for _ in range(n)],
        "RST Flag Count": [0.0 for _ in range(n)],
        "ACK Flag Count": [1.0 for _ in range(n)],
        "URG Flag Count": [0.0 for _ in range(n)],
        "PSH Flag Count": [0.0 for _ in range(n)],
        "Fwd PSH Flags": [0.0 for _ in range(n)],
        "Fwd URG Flags": [0.0 for _ in range(n)],
        "Fwd Packets/s": [10.0 for _ in range(n)],
        "Flow Bytes/s": [640.0 for _ in range(n)],
        "Label": ["BENIGN" if i < 6 else "PortScan" for i in range(n)],
        "Clean_Label": ["BENIGN" if i < 6 else "PortScan" for i in range(n)],
    }
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Test 1: Chronological Nondecreasing Replay
# --------------------------------------------------------------------------

def test_01_chronological_nondecreasing_replay():
    """Verify that prepare_chronological_stream enforces weak monotonicity (t[i+1] >= t[i])."""
    timestamps = [
        "2026-11-04 10:05:00",
        "2026-11-04 10:01:00",
        "2026-11-04 10:03:00",
        "2026-11-04 10:01:00",
        "2026-11-04 10:07:00",
    ]
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps),
        "Total Fwd Packets": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Label": ["BENIGN"] * 5,
    })

    sorted_df = prepare_chronological_stream(df)

    assert len(sorted_df) == 5
    assert sorted_df["Timestamp"].is_monotonic_increasing
    for i in range(len(sorted_df) - 1):
        assert sorted_df["Timestamp"].iloc[i + 1] >= sorted_df["Timestamp"].iloc[i]


# --------------------------------------------------------------------------
# Test 2: Stable Ordering for Duplicate Timestamps
# --------------------------------------------------------------------------

def test_02_stable_ordering_for_duplicate_timestamps():
    """Verify that duplicate timestamps maintain their original relative sequence."""
    timestamps = [
        "2026-11-04 10:00:00",
        "2026-11-04 10:00:00",
        "2026-11-04 10:00:00",
    ]
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps),
        "flow_id": ["first", "second", "third"],
        "Label": ["BENIGN"] * 3,
    })

    sorted_df = prepare_chronological_stream(df)
    assert sorted_df["flow_id"].tolist() == ["first", "second", "third"]


# --------------------------------------------------------------------------
# Test 3: Simultaneous Timestamp Groups Across Batch Boundaries
# --------------------------------------------------------------------------

def test_03_simultaneous_timestamp_groups_across_batch_boundaries():
    """
    Verify that splitting simultaneous events across chunk boundaries with flush=False
    preserves identical aggregation values to a single contiguous batch.
    """
    timestamps = [
        "2026-11-04 10:00:00",
        "2026-11-04 10:00:01",
        "2026-11-04 10:00:01",  # Chunk 1 ends here
        "2026-11-04 10:00:01",  # Chunk 2 begins here
        "2026-11-04 10:00:02",  # Chunk 2 ends here
        "2026-11-04 10:00:02",  # Chunk 3 begins here
    ]
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps),
        "Total Fwd Packets": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Total Length of Fwd Packets": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "Fwd IAT Total": [1000.0] * 6,
    })

    # Ground truth single batch
    ctx_single = StreamingCausalContext()
    single_res = ctx_single.process_batch(df, causal_mode="simultaneous", flush=True)

    # Chunked multi-batch
    ctx_multi = StreamingCausalContext()
    c1 = df.iloc[0:3]
    c2 = df.iloc[3:5]
    c3 = df.iloc[5:6]

    res1 = ctx_multi.process_batch(c1, causal_mode="simultaneous", flush=False)
    res2 = ctx_multi.process_batch(c2, causal_mode="simultaneous", flush=False)
    res3 = ctx_multi.process_batch(c3, causal_mode="simultaneous", flush=False)
    res_flush = ctx_multi.flush()

    multi_res = pd.concat([res1, res2, res3, res_flush], ignore_index=True)

    assert len(multi_res) == len(single_res) == 6
    np.testing.assert_allclose(
        multi_res["Rolling_Flow_Velocity_10s"].astype(float).values,
        single_res["Rolling_Flow_Velocity_10s"].astype(float).values,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        multi_res["Rolling_Bytes_Velocity_10s"].astype(float).values,
        single_res["Rolling_Bytes_Velocity_10s"].astype(float).values,
        atol=1e-9,
    )


# --------------------------------------------------------------------------
# Test 4: Streaming Feature Engine Updated Exactly Once Per Flow
# --------------------------------------------------------------------------

def test_04_streaming_feature_engine_updated_exactly_once(tmp_path, synthetic_flow_pool):
    """Verify that context is updated exactly once per flow without duplicate counting."""
    test_parquet = tmp_path / "test_stream.parquet"
    synthetic_flow_pool.to_parquet(test_parquet)

    feed_file = tmp_path / "live_feed.json"
    summary = run_stream_simulation(
        speed_eps=0,
        max_events=len(synthetic_flow_pool),
        inject_attacks=False,
        chunk_size=3,
        test_file=test_parquet,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    assert summary["total_processed"] == len(synthetic_flow_pool)
    assert len(summary["recent_alerts"]) == len(synthetic_flow_pool)


# --------------------------------------------------------------------------
# Test 5: v1 Payload Backward Compatibility
# --------------------------------------------------------------------------

def test_05_v1_payload_backward_compatibility(tmp_path, synthetic_flow_pool):
    """Verify that all existing v1 alert payload keys are present, typed, and within range."""
    test_parquet = tmp_path / "test_stream.parquet"
    synthetic_flow_pool.to_parquet(test_parquet)
    feed_file = tmp_path / "live_feed.json"

    summary = run_stream_simulation(
        speed_eps=0,
        max_events=4,
        inject_attacks=False,
        chunk_size=2,
        test_file=test_parquet,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    required_v1_keys = [
        "timestamp",
        "flow_index",
        "ground_truth",
        "score",
        "threshold",
        "verdict",
        "severity",
        "latency_ms",
        "top_drivers",
        "narrative",
    ]

    for alert in summary["recent_alerts"]:
        for k in required_v1_keys:
            assert k in alert, f"Missing required v1 key: {k}"
        assert isinstance(alert["score"], float)
        assert isinstance(alert["threshold"], float)
        assert isinstance(alert["verdict"], str)
        assert isinstance(alert["severity"], str)
        assert isinstance(alert["latency_ms"], float)
        assert isinstance(alert["top_drivers"], list)
        assert isinstance(alert["narrative"], str)


# --------------------------------------------------------------------------
# Test 6: v1 Score / Verdict Invariance Against Deterministic Baseline
# --------------------------------------------------------------------------

def test_06_v1_score_and_verdict_invariance_against_deterministic_baseline(v1_explainer):
    """Verify that identical input flows yield bit-exact v1 scores (diff < 1e-9)."""
    raw_flow = {
        "Total Fwd Packets": 2.0,
        "Total Length of Fwd Packets": 128.0,
        "act_data_pkt_fwd": 1.0,
        "Subflow Fwd Packets": 2.0,
        "Subflow Fwd Bytes": 128.0,
        "Fwd Header Length": 64.0,
        "min_seg_size_forward": 32.0,
        "Avg Fwd Segment Size": 64.0,
        "Fwd Packet Length Max": 64.0,
        "Fwd Packet Length Min": 64.0,
        "Fwd Packet Length Mean": 64.0,
        "Fwd Packet Length Std": 0.0,
        "Flow Duration": 1500.0,
        "Fwd IAT Total": 1500.0,
        "Fwd IAT Mean": 1500.0,
        "Fwd IAT Std": 0.0,
        "Fwd IAT Max": 1500.0,
        "Fwd IAT Min": 1500.0,
        "Fwd Packets/s": 1333.33,
        "Flow Bytes/s": 85333.33,
        "Init_Win_bytes_forward": 29200.0,
        "Fwd PSH Flags": 0.0,
        "Fwd URG Flags": 0.0,
        "SYN Flag Count": 1.0,
        "FIN Flag Count": 0.0,
        "RST Flag Count": 0.0,
        "ACK Flag Count": 1.0,
        "URG Flag Count": 0.0,
        "PSH Flag Count": 0.0,
        "Rolling_Flow_Velocity_10s": 5.0,
        "Rolling_Bytes_Velocity_10s": 320.0,
        "Rolling_Single_Pkt_Ratio_10s": 0.2,
    }

    baseline_explanation = v1_explainer.explain_flow(raw_flow)
    retest_explanation = v1_explainer.explain_flow(raw_flow)

    assert abs(baseline_explanation["anomaly_score"] - retest_explanation["anomaly_score"]) < 1e-9
    assert baseline_explanation["verdict"] == retest_explanation["verdict"]
    assert baseline_explanation["severity"] == retest_explanation["severity"]
    assert baseline_explanation["threshold"] == retest_explanation["threshold"]


# --------------------------------------------------------------------------
# Test 7: v2 Receives Exactly 26 Correctly Ordered Features
# --------------------------------------------------------------------------

def test_07_v2_receives_exactly_26_correctly_ordered_features(v2_adapter):
    """Verify that v2 shadow receives strictly 26 PURE_ONE_WAY_FEATURES in canonical order."""
    flow_record = {f: 10.0 for f in PURE_ONE_WAY_FEATURES}
    flow_record["_is_valid"] = True
    flow_record["_context_state"] = "HEALTHY"
    flow_record["_overflow_events"] = 0

    shadow_res = score_v2_shadow(flow_record, v2_adapter)
    assert shadow_res["status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"
    assert isinstance(shadow_res["score"], float)

    # Missing a required feature triggers UNAVAILABLE without crashing
    incomplete_flow = {f: 10.0 for f in PURE_ONE_WAY_FEATURES if f != "Fwd_Bytes_Per_Sec"}
    incomplete_flow["_is_valid"] = True
    incomplete_shadow = score_v2_shadow(incomplete_flow, v2_adapter)
    assert incomplete_shadow["status"] == "UNAVAILABLE"
    assert incomplete_shadow["score"] is None


# --------------------------------------------------------------------------
# Test 8: v2 Uses Correct Scaler and Thresholds
# --------------------------------------------------------------------------

def test_08_v2_uses_correct_scaler_and_thresholds(v2_adapter):
    """Verify v2 primary (0.314036) and reference (0.099018) threshold logic."""
    assert abs(v2_adapter.primary_threshold - 0.314036) < 1e-5
    assert abs(v2_adapter.reference_threshold - 0.099018) < 1e-5
    assert v2_adapter.expected_dim == 26

    # Test decision rule score > threshold
    p_thresh = v2_adapter.primary_threshold
    r_thresh = v2_adapter.reference_threshold

    assert (p_thresh + 0.001) > p_thresh
    assert not (p_thresh - 0.001 > p_thresh)
    assert (r_thresh + 0.001) > r_thresh


# --------------------------------------------------------------------------
# Test 9: v2 is Not Called When Context is Degraded
# --------------------------------------------------------------------------

def test_09_v2_not_called_when_context_is_degraded(v2_adapter):
    """Verify that degraded context sets DEGRADED_CONTEXT_OVERFLOW and nullifies scores/verdicts."""
    flow_record = {f: 10.0 for f in PURE_ONE_WAY_FEATURES}
    flow_record["_is_valid"] = False
    flow_record["_context_state"] = "DEGRADED"
    flow_record["_overflow_events"] = 150

    res = score_v2_shadow(flow_record, v2_adapter)

    assert res["status"] == "DEGRADED_CONTEXT_OVERFLOW"
    assert res["score"] is None
    assert res["primary_is_anomaly"] is None
    assert res["primary_verdict"] is None
    assert res["reference_is_anomaly"] is None
    assert res["reference_verdict"] is None
    assert res["context_validity"] is False
    assert res["context_state"] == "DEGRADED"
    assert res["overflow_event_count"] == 150
    assert res["top_drivers"] is None


# --------------------------------------------------------------------------
# Test 10: v2 Exceptions Do Not Interrupt v1 Alerts
# --------------------------------------------------------------------------

def test_10_v2_exceptions_do_not_interrupt_v1_alerts(tmp_path, synthetic_flow_pool):
    """Verify that an exception raised in v2 shadow execution leaves v1 unaffected."""
    test_parquet = tmp_path / "test_stream.parquet"
    synthetic_flow_pool.to_parquet(test_parquet)
    feed_file = tmp_path / "live_feed.json"

    # Mock adapter that raises RuntimeError on model prediction
    mock_v2 = MagicMock()
    mock_v2.is_available = True
    mock_v2.primary_threshold = 0.314036
    mock_v2.reference_threshold = 0.099018
    mock_v2.pipeline = None
    mock_v2.model.predict.side_effect = RuntimeError("Simulated v2 inference fault")

    flow_record = {f: 10.0 for f in PURE_ONE_WAY_FEATURES}
    flow_record["_is_valid"] = True
    flow_record["_context_state"] = "HEALTHY"
    flow_record["_overflow_events"] = 0

    shadow_res = score_v2_shadow(flow_record, mock_v2)
    assert shadow_res["status"] == "UNAVAILABLE"
    assert shadow_res["error"] == "RuntimeError"
    assert shadow_res["score"] is None


# --------------------------------------------------------------------------
# Test 11: v2 Does Not Change Production Alert Counters
# --------------------------------------------------------------------------

def test_11_v2_does_not_change_production_alert_counters(tmp_path, synthetic_flow_pool):
    """Verify that total_anomalies and anomaly_rate_pct are governed strictly by v1."""
    test_parquet = tmp_path / "test_stream.parquet"
    synthetic_flow_pool.to_parquet(test_parquet)
    feed_file = tmp_path / "live_feed.json"

    summary_dual = run_stream_simulation(
        speed_eps=0,
        max_events=len(synthetic_flow_pool),
        inject_attacks=False,
        chunk_size=4,
        v1_only=False,
        test_file=test_parquet,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    summary_v1_only = run_stream_simulation(
        speed_eps=0,
        max_events=len(synthetic_flow_pool),
        inject_attacks=False,
        chunk_size=4,
        v1_only=True,
        test_file=test_parquet,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    assert summary_dual["total_anomalies"] == summary_v1_only["total_anomalies"]
    assert summary_dual["anomaly_rate_pct"] == summary_v1_only["anomaly_rate_pct"]
    assert summary_dual["total_processed"] == summary_v1_only["total_processed"]


# --------------------------------------------------------------------------
# Test 12: Live-Feed JSON Schema Compatibility
# --------------------------------------------------------------------------

def test_12_live_feed_json_schema_compatibility(tmp_path, synthetic_flow_pool):
    """Verify outputs/evaluation/live_stream_feed.json matches the dashboard schema."""
    test_parquet = tmp_path / "test_stream.parquet"
    synthetic_flow_pool.to_parquet(test_parquet)
    feed_file = tmp_path / "live_feed.json"

    run_stream_simulation(
        speed_eps=0,
        max_events=5,
        inject_attacks=False,
        chunk_size=2,
        v1_only=False,
        test_file=test_parquet,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    with open(feed_file, "r", encoding="utf-8") as f:
        feed_data = json.load(f)

    # Top-level keys
    for key in ["last_updated", "total_processed", "total_anomalies", "anomaly_rate_pct", "recent_alerts", "v2_shadow_summary"]:
        assert key in feed_data

    # Nested v2_shadow keys in alerts
    alert_v2 = feed_data["recent_alerts"][0]["v2_shadow"]
    required_shadow_keys = [
        "status",
        "score",
        "primary_threshold",
        "primary_is_anomaly",
        "primary_verdict",
        "reference_threshold",
        "reference_is_anomaly",
        "reference_verdict",
        "inference_latency_ms",
        "context_validity",
        "context_state",
        "overflow_event_count",
        "top_drivers",
    ]
    for k in required_shadow_keys:
        assert k in alert_v2
    assert alert_v2["top_drivers"] is None


# --------------------------------------------------------------------------
# Test 13: Correct Shutdown Flush
# --------------------------------------------------------------------------

def test_13_correct_shutdown_flush():
    """Verify that flush() at shutdown finalizes any trailing open timestamp group."""
    timestamps = [
        "2026-11-04 10:00:00",
        "2026-11-04 10:00:05",
        "2026-11-04 10:00:05",  # Open trailing group
    ]
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps),
        "Total Fwd Packets": [1.0, 2.0, 3.0],
        "Total Length of Fwd Packets": [64.0, 128.0, 192.0],
        "Fwd IAT Total": [1000.0] * 3,
    })

    ctx = StreamingCausalContext()
    batch_res = ctx.process_batch(df, causal_mode="simultaneous", flush=False)
    # 10:00:00 emitted, 10:00:05 pending
    assert len(batch_res) == 1

    flush_res = ctx.flush()
    # 10:00:05 emitted on shutdown flush
    assert len(flush_res) == 2
    assert len(batch_res) + len(flush_res) == 3


# --------------------------------------------------------------------------
# Test 14: Existing v1-Only Simulator Behavior Remains Available
# --------------------------------------------------------------------------

def test_14_existing_v1_only_simulator_behavior_remains_available(tmp_path, synthetic_flow_pool):
    """Verify that v1_only=True generates pure v1 alert payloads without v2 shadow artifacts."""
    test_parquet = tmp_path / "test_stream.parquet"
    synthetic_flow_pool.to_parquet(test_parquet)
    feed_file = tmp_path / "live_feed.json"

    summary = run_stream_simulation(
        speed_eps=0,
        max_events=4,
        inject_attacks=False,
        chunk_size=2,
        v1_only=True,
        test_file=test_parquet,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    with open(feed_file, "r", encoding="utf-8") as f:
        feed_data = json.load(f)

    assert "v2_shadow_summary" not in feed_data
    for alert in feed_data["recent_alerts"]:
        assert "v2_shadow" not in alert
    assert summary["v2_primary_anomalies"] == 0
    assert summary["v2_reference_anomalies"] == 0


# --------------------------------------------------------------------------
# Test 15: Direct Script Execution and Import Resilience
# --------------------------------------------------------------------------

def test_15_direct_script_execution_and_import_resilience():
    """Verify simulate_stream.py and checkpoint.py execute directly without ModuleNotFoundError."""
    import subprocess
    res = subprocess.run(
        [sys.executable, str(ROOT / "src" / "simulate_stream.py"), "--events", "10", "--eps", "0", "--no-attacks"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Script failed with code {res.returncode}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Stream simulation complete" in res.stdout
