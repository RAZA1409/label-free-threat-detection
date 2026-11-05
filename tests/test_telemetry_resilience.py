"""
tests/test_telemetry_resilience.py
AEGIS Phase 3 Stage 4B.3C: Comprehensive Telemetry Resilience Test Suite.

Verifies:
1. Strict RFC 8259 JSON serialization with safe normalization of NumPy types,
   NaN, and non-finite values to None (null).
2. Atomic JSON file persistence using temporary file swap and os.replace().
3. Safe JSON reading handling missing, 0-byte, whitespace, and malformed files.
4. Bounded telemetry batching and throttling for benign traffic.
5. Immediate zero-latency persistence for critical alerts and quarantined flows.
6. Fault tolerance under disk I/O, permission, and serialization errors.
7. In-memory health tracking and automatic recovery to HEALTHY state.
8. Alert preservation guarantee: zero alert loss across storage outages.
9. Stream pipeline and inference invariance: model scores, thresholds, and
   verdicts are completely decoupled from persistence failures.
10. Schema backward compatibility for dashboard and live feed consumers.
"""

import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from src.telemetry import (
    sanitize_for_json,
    atomic_write_json,
    safe_read_json,
    TelemetryManager,
)
from src.simulate_stream import run_stream_simulation
from src.features import ONE_WAY_FEATURES, AUGMENTED_ONE_WAY_FEATURES


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_stream_dataset(tmp_path):
    """Generates a small valid test dataset for streaming simulation."""
    np.random.seed(42)
    n_rows = 30
    timestamps = [f"2026-11-04 10:00:{i:02d}" for i in range(n_rows)]
    data = {
        "Timestamp": timestamps,
        "Source_IP": [f"192.168.1.{10 + (i % 5)}" for i in range(n_rows)],
        "Destination_IP": ["10.0.0.1"] * n_rows,
        "Protocol": [6] * n_rows,
        "Label": ["BENIGN"] * 25 + ["DDoS"] * 5,
        "Clean_Label": ["BENIGN"] * 25 + ["DDoS"] * 5,
    }
    for col in ONE_WAY_FEATURES:
        if col not in data:
            data[col] = np.random.uniform(10.0, 100.0, size=n_rows)

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "synthetic_stream.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


# -----------------------------------------------------------------------------
# Test 1: Strict JSON Normalization (NumPy types, NaN, Inf)
# -----------------------------------------------------------------------------

def test_sanitize_for_json_numpy_and_non_finite():
    """Verify that NumPy types, NaN, and Inf are converted to strict JSON-compliant types."""
    raw_payload = {
        "np_int": np.int64(42),
        "np_int32": np.int32(10),
        "np_float": np.float64(3.14159),
        "np_bool_true": np.bool_(True),
        "np_bool_false": np.bool_(False),
        "py_nan": float("nan"),
        "np_nan": np.nan,
        "py_pos_inf": float("inf"),
        "py_neg_inf": float("-inf"),
        "np_pos_inf": np.inf,
        "np_neg_inf": -np.inf,
        "array_1d": np.array([1.0, 2.0, np.nan, np.inf]),
        "series": pd.Series([10, 20, 30]),
        "timestamp": pd.Timestamp("2026-11-04 10:00:00"),
        "nested_dict": {
            123: np.float32(99.5),
            "inner_nan": float("nan"),
        },
    }

    sanitized = sanitize_for_json(raw_payload)

    # Verify native Python types
    assert isinstance(sanitized["np_int"], int)
    assert isinstance(sanitized["np_int32"], int)
    assert isinstance(sanitized["np_float"], float)
    assert isinstance(sanitized["np_bool_true"], bool)
    assert sanitized["np_bool_true"] is True
    assert isinstance(sanitized["np_bool_false"], bool)
    assert sanitized["np_bool_false"] is False

    # Verify non-finite values are normalized to None (JSON null)
    assert sanitized["py_nan"] is None
    assert sanitized["np_nan"] is None
    assert sanitized["py_pos_inf"] is None
    assert sanitized["py_neg_inf"] is None
    assert sanitized["np_pos_inf"] is None
    assert sanitized["np_neg_inf"] is None

    # Verify lists and timestamps
    assert sanitized["array_1d"] == [1.0, 2.0, None, None]
    assert sanitized["series"] == [10, 20, 30]
    assert sanitized["timestamp"] == "2026-11-04T10:00:00"
    assert "123" in sanitized["nested_dict"]
    assert sanitized["nested_dict"]["123"] == 99.5
    assert sanitized["nested_dict"]["inner_nan"] is None

    # Strict JSON serialization with allow_nan=False MUST succeed
    serialized = json.dumps(sanitized, allow_nan=False)
    assert "null" in serialized
    assert "NaN" not in serialized
    assert "Infinity" not in serialized


# -----------------------------------------------------------------------------
# Test 2: Atomic JSON Writer Atomicity and Replacement
# -----------------------------------------------------------------------------

def test_atomic_write_json_atomicity_and_swap(tmp_path):
    """Verify atomic write creates valid files and leaves no leftover temporary files."""
    target_file = tmp_path / "feed.json"
    data = {"status": "ACTIVE", "processed": 100, "scores": [0.1, 0.2]}

    success = atomic_write_json(target_file, data)
    assert success is True
    assert target_file.exists()

    with open(target_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["status"] == "ACTIVE"
    assert loaded["processed"] == 100

    # Ensure no temporary files (.tmp) remain
    tmp_files = list(tmp_path.glob(".*.tmp"))
    assert len(tmp_files) == 0

    # Atomic update
    data_updated = {"status": "ACTIVE", "processed": 200, "scores": [0.3]}
    success2 = atomic_write_json(target_file, data_updated)
    assert success2 is True

    with open(target_file, "r", encoding="utf-8") as f:
        loaded2 = json.load(f)
    assert loaded2["processed"] == 200


# -----------------------------------------------------------------------------
# Test 3: Atomic Writer Cleanup on Failure
# -----------------------------------------------------------------------------

def test_atomic_write_json_cleanup_on_failure(tmp_path):
    """Verify temporary files are unlinked and target file is unharmed on failure."""
    target_file = tmp_path / "critical.json"
    initial_data = {"version": 1, "state": "good"}
    atomic_write_json(target_file, initial_data)

    # Simulate an error during os.replace
    with patch("os.replace", side_effect=OSError("Simulated replacement lock")):
        success = atomic_write_json(target_file, {"version": 2}, raise_on_error=False)
        assert success is False

        # When raise_on_error=True, it raises
        with pytest.raises(OSError, match="Simulated replacement lock"):
            atomic_write_json(target_file, {"version": 2}, raise_on_error=True)

    # Initial file remains intact and uncorrupted
    with open(target_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["version"] == 1

    # No leftover .tmp files
    assert len(list(tmp_path.glob(".*.tmp"))) == 0


# -----------------------------------------------------------------------------
# Test 4: Fault-Tolerant JSON Reader Edge Cases
# -----------------------------------------------------------------------------

def test_safe_read_json_edge_cases(tmp_path):
    """Verify safe_read_json handles missing, empty, whitespace, and corrupt files."""
    default_payload = {"status": "idle", "total_processed": 0}

    # Case A: Missing file
    missing_file = tmp_path / "non_existent.json"
    res = safe_read_json(missing_file, default=default_payload)
    assert res == default_payload

    # Case B: 0-byte file (empty)
    empty_file = tmp_path / "empty.json"
    empty_file.touch()
    assert empty_file.stat().st_size == 0
    res = safe_read_json(empty_file, default=default_payload)
    assert res == default_payload

    # Case C: Whitespace-only file
    ws_file = tmp_path / "whitespace.json"
    ws_file.write_text("   \n\t  \n  ", encoding="utf-8")
    res = safe_read_json(ws_file, default=default_payload)
    assert res == default_payload

    # Case D: Malformed / Truncated JSON
    malformed_file = tmp_path / "corrupt.json"
    malformed_file.write_text('{"total_processed": 50, "alerts": [', encoding="utf-8")
    res = safe_read_json(malformed_file, default=default_payload)
    assert res == default_payload

    # Case E: Valid file
    valid_file = tmp_path / "valid.json"
    valid_file.write_text('{"total_processed": 42, "status": "ok"}', encoding="utf-8")
    res = safe_read_json(valid_file, default=default_payload)
    assert res["total_processed"] == 42
    assert res["status"] == "ok"


# -----------------------------------------------------------------------------
# Test 5: Bounded Telemetry Batching and Throttling
# -----------------------------------------------------------------------------

def test_telemetry_manager_throttling(tmp_path):
    """Verify non-critical writes are throttled while critical alerts bypass throttling."""
    feed_file = tmp_path / "live_feed.json"
    mgr = TelemetryManager(
        feed_path=feed_file,
        enabled=True,
        throttle_interval_sec=10.0,  # Large interval
        throttle_event_count=5,     # Batch of 5
    )

    feed_state = {"total_processed": 0, "total_anomalies": 0}

    # First write at startup is allowed (last_write is 0.0)
    mgr.record_event(is_critical=False)
    written = mgr.update_and_write(feed_state, force=False)
    assert written is True
    assert mgr.total_writes_succeeded == 1

    # Events 2, 3, 4, 5: throttled and suppressed (total 4 suppressed)
    for i in range(2, 6):
        mgr.record_event(is_critical=False)
        feed_state["total_processed"] = i
        mgr.update_and_write(feed_state, force=False)

    assert mgr.total_writes_suppressed == 4
    assert mgr.total_writes_succeeded == 1

    # Event 6: this is the 5th event since last write -> triggers write
    mgr.record_event(is_critical=False)
    feed_state["total_processed"] = 6
    mgr.update_and_write(feed_state, force=False)
    assert mgr.total_writes_succeeded == 2
    assert mgr.events_since_last_write == 0

    # Critical event: forces immediate write even when count is 1 and interval not elapsed
    mgr.record_event(is_critical=True)
    feed_state["total_processed"] = 7
    feed_state["total_anomalies"] = 1
    mgr.update_and_write(feed_state, force=True)
    assert mgr.total_writes_succeeded == 3

    # Shutdown flush: forces immediate write
    mgr.flush(feed_state)
    assert mgr.total_writes_succeeded == 4


# -----------------------------------------------------------------------------
# Test 6: In-Memory Health Tracking and Degradation
# -----------------------------------------------------------------------------

def test_telemetry_manager_degraded_tracking_under_io_error(tmp_path):
    """Verify I/O errors update health to DEGRADED without throwing exceptions."""
    feed_file = tmp_path / "feed.json"
    mgr = TelemetryManager(feed_path=feed_file, enabled=True, throttle_interval_sec=0, throttle_event_count=1)

    feed_state = {"total_processed": 1}

    # Simulate disk write failure
    with patch("src.telemetry.atomic_write_json", side_effect=OSError("No space left on device")):
        success = mgr.update_and_write(feed_state, force=True)
        assert success is False
        assert mgr.is_degraded is True
        assert mgr.consecutive_failures == 1
        assert mgr.total_failures == 1
        assert "No space left on device" in mgr.last_error

        health = mgr.get_health()
        assert health["status"] == "DEGRADED"
        assert health["is_degraded"] is True
        assert health["consecutive_failures"] == 1
        assert health["total_failures"] == 1

        # Second failure increments failure counters
        mgr.update_and_write(feed_state, force=True)
        assert mgr.consecutive_failures == 2
        assert mgr.total_failures == 2


# -----------------------------------------------------------------------------
# Test 7: Automatic Recovery After Persistence Restores
# -----------------------------------------------------------------------------

def test_telemetry_manager_automatic_recovery(tmp_path):
    """Verify telemetry health automatically restores to HEALTHY when disk becomes writable."""
    feed_file = tmp_path / "feed.json"
    mgr = TelemetryManager(feed_path=feed_file, enabled=True, throttle_interval_sec=0, throttle_event_count=1)
    feed_state = {"total_processed": 1}

    # Induce failures
    with patch("src.telemetry.atomic_write_json", side_effect=PermissionError("Permission denied")):
        mgr.update_and_write(feed_state, force=True)
        mgr.update_and_write(feed_state, force=True)

    assert mgr.is_degraded is True
    assert mgr.consecutive_failures == 2
    assert mgr.total_failures == 2

    # Now disk recovers
    recovered = mgr.update_and_write(feed_state, force=True)
    assert recovered is True
    assert mgr.is_degraded is False
    assert mgr.consecutive_failures == 0
    assert mgr.total_failures == 2  # Total cumulative failures retained for audit
    assert mgr.last_error is None

    health = mgr.get_health()
    assert health["status"] == "HEALTHY"
    assert health["is_degraded"] is False
    assert health["consecutive_failures"] == 0
    assert health["total_writes_succeeded"] == 1


# -----------------------------------------------------------------------------
# Test 8: Critical Alert Backlog Preservation Guarantee
# -----------------------------------------------------------------------------

def test_alert_preservation_guarantee_during_storage_outage(tmp_path):
    """Verify alerts generated during an outage are retained in memory and written on recovery."""
    feed_file = tmp_path / "live_feed.json"
    mgr = TelemetryManager(feed_path=feed_file, enabled=True, throttle_interval_sec=0, throttle_event_count=1)

    alerts = [
        {"timestamp": "10:00:01", "verdict": "ALERT", "score": 0.85},
        {"timestamp": "10:00:02", "verdict": "ALERT", "score": 0.92},
        {"timestamp": "10:00:03", "verdict": "QUARANTINED", "score": None},
    ]

    feed_state = {
        "last_updated": "10:00:00",
        "total_processed": 3,
        "total_anomalies": 3,
        "recent_alerts": [],
    }

    # During outage: 3 alerts are generated
    with patch("src.telemetry.atomic_write_json", side_effect=OSError("Disk full")):
        for a in alerts:
            feed_state["recent_alerts"].append(a)
            mgr.update_and_write(feed_state, force=True, recent_alert=a)

    assert mgr.is_degraded is True
    assert mgr.get_health()["backlog_alerts_retained"] == 3

    # Outage resolves: flush writes state to disk
    mgr.flush(feed_state)
    assert mgr.is_degraded is False
    assert feed_file.exists()

    with open(feed_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)

    assert persisted["telemetry_health"]["status"] == "HEALTHY"
    assert len(persisted["recent_alerts"]) == 3
    assert persisted["recent_alerts"][0]["score"] == 0.85
    assert persisted["recent_alerts"][1]["score"] == 0.92
    assert persisted["recent_alerts"][2]["verdict"] == "QUARANTINED"


# -----------------------------------------------------------------------------
# Test 9: Model Inference Invariance Under Telemetry Failure
# -----------------------------------------------------------------------------

def test_streaming_inference_unaffected_by_telemetry_crash(synthetic_stream_dataset, tmp_path):
    """Verify that complete telemetry write failures do not crash inference or alter scores."""
    normal_feed = tmp_path / "normal_feed.json"
    res_normal = run_stream_simulation(
        speed_eps=0,
        max_events=20,
        inject_attacks=True,
        chunk_size=5,
        v1_only=False,
        test_file=synthetic_stream_dataset,
        live_feed_file=normal_feed,
        write_feed=True,
        verbose=False,
    )

    failing_feed = tmp_path / "failing_feed.json"
    # Monkeypatch atomic_write_json in telemetry module to fail on every single call
    with patch("src.telemetry.atomic_write_json", side_effect=OSError("Disk hardware error")):
        res_failing = run_stream_simulation(
            speed_eps=0,
            max_events=20,
            inject_attacks=True,
            chunk_size=5,
            v1_only=False,
            test_file=synthetic_stream_dataset,
            live_feed_file=failing_feed,
            write_feed=True,
            verbose=False,
        )

    # Simulation did not crash!
    # Primary inference metrics MUST match exactly
    assert res_failing["total_processed"] == res_normal["total_processed"]
    assert res_failing["total_anomalies"] == res_normal["total_anomalies"]
    assert res_failing["anomaly_rate_pct"] == res_normal["anomaly_rate_pct"]
    assert res_failing["v2_primary_anomalies"] == res_normal["v2_primary_anomalies"]
    assert res_failing["v2_reference_anomalies"] == res_normal["v2_reference_anomalies"]
    assert res_failing["v2_degraded_count"] == res_normal["v2_degraded_count"]
    assert res_failing["v2_unavailable_count"] == res_normal["v2_unavailable_count"]
    assert res_failing["total_dropped_overflow_events"] == res_normal["total_dropped_overflow_events"]

    # All alerts are intact in memory
    assert len(res_failing["recent_alerts"]) == len(res_normal["recent_alerts"])
    for a_norm, a_fail in zip(res_normal["recent_alerts"], res_failing["recent_alerts"]):
        assert a_norm["verdict"] == a_fail["verdict"]
        assert a_norm["score"] == a_fail["score"]
        assert a_norm["threshold"] == a_fail["threshold"]

    # Health status reflects degradation
    assert res_failing["telemetry_health"]["status"] == "DEGRADED"
    assert res_failing["telemetry_health"]["total_failures"] > 0


# -----------------------------------------------------------------------------
# Test 10: Live Feed Schema Backward Compatibility
# -----------------------------------------------------------------------------

def test_live_feed_schema_backward_compatibility(synthetic_stream_dataset, tmp_path):
    """Verify outputs/evaluation/live_stream_feed.json preserves 100% of schema keys."""
    feed_file = tmp_path / "live_stream_feed.json"
    run_stream_simulation(
        speed_eps=0,
        max_events=10,
        inject_attacks=False,
        chunk_size=5,
        v1_only=False,
        test_file=synthetic_stream_dataset,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    with open(feed_file, "r", encoding="utf-8") as f:
        feed_data = json.load(f)

    # Standard top-level keys
    required_keys = [
        "last_updated",
        "total_processed",
        "total_anomalies",
        "anomaly_rate_pct",
        "quarantine_summary",
        "recent_alerts",
        "v2_shadow_summary",
        "telemetry_health",
    ]
    for k in required_keys:
        assert k in feed_data, f"Missing required key: {k}"

    # Telemetry health schema
    th = feed_data["telemetry_health"]
    for k in ["status", "is_degraded", "consecutive_failures", "total_failures", "last_successful_write"]:
        assert k in th

    assert th["status"] == "HEALTHY"
    assert th["is_degraded"] is False


# -----------------------------------------------------------------------------
# Test 11: Dashboard Reader Safe Handling of Corrupt Live Feed
# -----------------------------------------------------------------------------

def test_dashboard_server_safe_reading_of_corrupt_feed(tmp_path):
    """Verify that safe_read_json provides stable fallbacks for dashboard endpoints."""
    corrupt_feed = tmp_path / "corrupt_feed.json"
    corrupt_feed.write_text("{ incomplete json: ...", encoding="utf-8")

    default_feed = {
        "status": "idle",
        "total_processed": 0,
        "total_anomalies": 0,
        "anomaly_rate_pct": 0.0,
        "recent_alerts": [],
        "telemetry_health": {
            "status": "HEALTHY",
            "consecutive_failures": 0,
            "total_failures": 0,
            "last_error": None,
        },
    }

    # Reader retrieves default fallback cleanly without raising
    result = safe_read_json(corrupt_feed, default=default_feed)
    assert result == default_feed
    assert result["status"] == "idle"
    assert result["telemetry_health"]["status"] == "HEALTHY"
