"""
tests/test_input_sanitization.py
AEGIS Phase 3 Stage 4B.3A: Input Sanitization & Malformed-Flow Quarantine Test Suite.

Verifies:
1. Valid flow record passes validation and sanitization.
2. Missing timestamp is quarantined (MISSING_TIMESTAMP).
3. Invalid or NaT timestamp is quarantined (INVALID_TIMESTAMP).
4. Missing packet or byte counters quarantined (MISSING_REQUIRED_FIELD).
5. Negative packet counter strictly quarantined, never clamped (NEGATIVE_COUNTER).
6. Negative byte counter strictly quarantined, never clamped (NEGATIVE_COUNTER).
7. Infinite and NaN counters quarantined (INFINITE_COUNTER / INVALID_COUNTER).
8. Corrupted non-numeric feature string quarantined (NON_NUMERIC_FEATURE).
9. Unexpected nested data types quarantined (UNEXPECTED_DATA_TYPE / INVALID_RECORD_TYPE).
10. Quarantined records never enter causal sliding context or pending group.
11. Buffer state and latest timestamp strictly unchanged after malformed flow arrival.
12. Quarantine manager metrics and reason categorization increment correctly.
13. Diagnostic sample storage strictly bounded by max_diagnostic_samples FIFO ceiling.
14. Defensive v1 fault boundary catches explain_flow exception without crashing simulator.
15. v1 fault boundary never fabricates anomaly or benign verdict (UNAVAILABLE / PROCESSING_ERROR).
16. Subsequent flow executes normally after preceding flow v1 fault.
17. v2 shadow adapter attribute access safety (Finding D1 hardening).
18. Full system regression preservation and clean quarantine telemetry.
"""

import sys
import json
import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import deque

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (
    ONE_WAY_FEATURES,
    PURE_ONE_WAY_FEATURES,
    sanitize_labels,
)
from explainability import ThreatExplainer
from model_adapters import PureOneWayV2Adapter
from streaming_features import (
    validate_and_sanitize_flow_record,
    FlowQuarantineManager,
    StreamingCausalContext,
    StreamingFeatureEngine,
)
from simulate_stream import (
    score_v2_shadow,
    run_stream_simulation,
    TEST_FILE,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def valid_flow_record():
    """Provides a valid baseline network flow dictionary."""
    return {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 5.0,
        "Total Length of Fwd Packets": 500.0,
        "Fwd IAT Total": 1000.0,
        "Fwd Packet Length Max": 100.0,
        "Fwd Packet Length Min": 20.0,
        "Fwd Packet Length Mean": 50.0,
        "Fwd Packet Length Std": 10.0,
        "Fwd Header Length": 32.0,
        "min_seg_size_forward": 32.0,
        "Avg Fwd Segment Size": 50.0,
        "Init_Win_bytes_forward": 29200.0,
        "Label": "BENIGN",
        "Clean_Label": "BENIGN",
    }


@pytest.fixture
def synthetic_parquet_file(tmp_path):
    """Creates a temporary Parquet file with a mix of valid flows."""
    timestamps = [
        "2026-11-04 10:00:00",
        "2026-11-04 10:00:01",
        "2026-11-04 10:00:02",
        "2026-11-04 10:00:03",
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
        "Fwd PSH Flags": [0.0 for _ in range(n)],
        "Fwd URG Flags": [0.0 for _ in range(n)],
        "FIN Flag Count": [0.0 for _ in range(n)],
        "SYN Flag Count": [0.0 for _ in range(n)],
        "RST Flag Count": [0.0 for _ in range(n)],
        "PSH Flag Count": [0.0 for _ in range(n)],
        "ACK Flag Count": [1.0 for _ in range(n)],
        "URG Flag Count": [0.0 for _ in range(n)],
        "CWE Flag Count": [0.0 for _ in range(n)],
        "ECE Flag Count": [0.0 for _ in range(n)],
        "Down/Up Ratio": [0.0 for _ in range(n)],
        "Fwd Act Data Pkts": [float(i + 1) for i in range(n)],
        "Fwd Seg Size Min": [32.0 for _ in range(n)],
        "Active Mean": [0.0 for _ in range(n)],
        "Active Std": [0.0 for _ in range(n)],
        "Active Max": [0.0 for _ in range(n)],
        "Active Min": [0.0 for _ in range(n)],
        "Idle Mean": [0.0 for _ in range(n)],
        "Idle Std": [0.0 for _ in range(n)],
        "Idle Max": [0.0 for _ in range(n)],
        "Idle Min": [0.0 for _ in range(n)],
        "Label": ["BENIGN" for _ in range(n)],
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_synthetic.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


# --------------------------------------------------------------------------
# Test 01: Valid Flow Record Passes Validation
# --------------------------------------------------------------------------

def test_01_valid_flow_record_passes_validation(valid_flow_record):
    """Verifies that a well-formed flow record passes validation and retains correct types."""
    is_val, reason, sanitized = validate_and_sanitize_flow_record(valid_flow_record)
    assert is_val is True
    assert reason is None
    assert isinstance(sanitized, dict)
    assert isinstance(sanitized["Timestamp"], pd.Timestamp)
    assert sanitized["Total Fwd Packets"] == 5.0
    assert sanitized["Total Length of Fwd Packets"] == 500.0
    assert sanitized["Label"] == "BENIGN"


# --------------------------------------------------------------------------
# Test 02: Missing Timestamp Quarantined
# --------------------------------------------------------------------------

def test_02_missing_timestamp_quarantined():
    """Verifies that a record missing Timestamp or with Timestamp=None is quarantined."""
    # Missing key
    rec1 = {"Total Fwd Packets": 2.0, "Total Length of Fwd Packets": 128.0}
    is_val1, reason1, san1 = validate_and_sanitize_flow_record(rec1)
    assert is_val1 is False
    assert reason1 == "MISSING_TIMESTAMP"
    assert san1 is None

    # Timestamp is None
    rec2 = {"Timestamp": None, "Total Fwd Packets": 2.0, "Total Length of Fwd Packets": 128.0}
    is_val2, reason2, san2 = validate_and_sanitize_flow_record(rec2)
    assert is_val2 is False
    assert reason2 == "MISSING_TIMESTAMP"
    assert san2 is None


# --------------------------------------------------------------------------
# Test 03: Invalid Timestamp NaT or Unparseable Quarantined
# --------------------------------------------------------------------------

def test_03_invalid_timestamp_nat_or_unparseable_quarantined():
    """Verifies that NaT, empty string, or unparseable timestamps are quarantined."""
    # pd.NaT
    rec_nat = {"Timestamp": pd.NaT, "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0}
    is_val, reason, san = validate_and_sanitize_flow_record(rec_nat)
    assert is_val is False
    assert reason == "INVALID_TIMESTAMP"
    assert san is None

    # Unparseable string
    rec_unp = {"Timestamp": "not-a-valid-date", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0}
    is_val, reason, san = validate_and_sanitize_flow_record(rec_unp)
    assert is_val is False
    assert reason == "INVALID_TIMESTAMP"
    assert san is None

    # Empty string
    rec_empty = {"Timestamp": "", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0}
    is_val, reason, san = validate_and_sanitize_flow_record(rec_empty)
    assert is_val is False
    assert reason == "INVALID_TIMESTAMP"
    assert san is None


# --------------------------------------------------------------------------
# Test 04: Missing Packet or Byte Counters Quarantined
# --------------------------------------------------------------------------

def test_04_missing_packet_or_byte_counter_quarantined():
    """Verifies that records missing Total Fwd Packets or Total Length of Fwd Packets are quarantined."""
    # Missing Total Fwd Packets
    rec1 = {"Timestamp": "2026-11-04 10:00:00", "Total Length of Fwd Packets": 100.0}
    is_val1, reason1, san1 = validate_and_sanitize_flow_record(rec1)
    assert is_val1 is False
    assert reason1 == "MISSING_REQUIRED_FIELD"
    assert san1 is None

    # Total Length of Fwd Packets is None
    rec2 = {"Timestamp": "2026-11-04 10:00:00", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": None}
    is_val2, reason2, san2 = validate_and_sanitize_flow_record(rec2)
    assert is_val2 is False
    assert reason2 == "MISSING_REQUIRED_FIELD"
    assert san2 is None


# --------------------------------------------------------------------------
# Test 05: Negative Packet Counter Quarantined, Never Clamped
# --------------------------------------------------------------------------

def test_05_negative_packet_counter_quarantined_not_clamped():
    """
    CRITICAL: Verifies negative Total Fwd Packets is strictly quarantined with
    NEGATIVE_COUNTER and NEVER silently clamped to zero.
    """
    rec_neg = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": -1.0,
        "Total Length of Fwd Packets": 64.0,
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec_neg)
    assert is_val is False
    assert reason == "NEGATIVE_COUNTER"
    assert san is None


# --------------------------------------------------------------------------
# Test 06: Negative Byte Counter Quarantined, Never Clamped
# --------------------------------------------------------------------------

def test_06_negative_byte_counter_quarantined_not_clamped():
    """
    CRITICAL: Verifies negative Total Length of Fwd Packets is strictly quarantined
    with NEGATIVE_COUNTER and NEVER silently clamped to zero.
    """
    rec_neg_bytes = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 2.0,
        "Total Length of Fwd Packets": -50.0,
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec_neg_bytes)
    assert is_val is False
    assert reason == "NEGATIVE_COUNTER"
    assert san is None


# --------------------------------------------------------------------------
# Test 07: Infinite or NaN Counters Quarantined
# --------------------------------------------------------------------------

def test_07_infinite_or_nan_counters_quarantined():
    """Verifies that infinite (+inf, -inf) or NaN packet/byte counters are quarantined."""
    # Positive infinity
    rec_inf = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": float("inf"),
        "Total Length of Fwd Packets": 64.0,
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec_inf)
    assert is_val is False
    assert reason == "INFINITE_COUNTER"
    assert san is None

    # NaN counter
    rec_nan = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 1.0,
        "Total Length of Fwd Packets": float("nan"),
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec_nan)
    assert is_val is False
    assert reason == "INVALID_COUNTER"
    assert san is None


# --------------------------------------------------------------------------
# Test 08: Corrupted Non-Numeric Feature Quarantined
# --------------------------------------------------------------------------

def test_08_corrupted_non_numeric_feature_quarantined():
    """Verifies that a feature containing a corrupted non-numeric string is quarantined."""
    rec_corrupt = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 1.0,
        "Total Length of Fwd Packets": 64.0,
        "Fwd Packet Length Max": "corrupted_payload_string",
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec_corrupt)
    assert is_val is False
    assert reason == "NON_NUMERIC_FEATURE"
    assert san is None


# --------------------------------------------------------------------------
# Test 09: Unexpected Data Type Quarantined
# --------------------------------------------------------------------------

def test_09_unexpected_data_type_quarantined():
    """Verifies that unexpected data structures (lists, dicts, or non-record input) are quarantined."""
    # List in feature value
    rec_list = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 1.0,
        "Total Length of Fwd Packets": 64.0,
        "Fwd Header Length": [32, 64],
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec_list)
    assert is_val is False
    assert reason == "UNEXPECTED_DATA_TYPE"
    assert san is None

    # Non-record type passed as input
    is_val_str, reason_str, _ = validate_and_sanitize_flow_record("not a record")
    assert is_val_str is False
    assert reason_str == "INVALID_RECORD_TYPE"


# --------------------------------------------------------------------------
# Test 10: Invalid Record Never Enters Causal Context
# --------------------------------------------------------------------------

def test_10_invalid_record_never_enters_causal_context():
    """Verifies that malformed rows in a DataFrame are dropped and never enter context buffer."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)

    df = pd.DataFrame([
        {"Timestamp": "2026-11-04 10:00:00", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0},
        {"Timestamp": "2026-11-04 10:00:01", "Total Fwd Packets": -5.0, "Total Length of Fwd Packets": 128.0},  # Negative pkts
        {"Timestamp": "2026-11-04 10:00:02", "Total Fwd Packets": 2.0, "Total Length of Fwd Packets": 128.0},
        {"Timestamp": pd.NaT, "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0},  # NaT timestamp
        {"Timestamp": "2026-11-04 10:00:03", "Total Fwd Packets": 3.0, "Total Length of Fwd Packets": 192.0},
    ])

    emitted = ctx.process_batch(df, causal_mode="simultaneous", flush=True)

    # Only 3 valid rows should be processed and emitted
    assert len(emitted) == 3
    assert qm.total_quarantined_records == 2
    assert qm.quarantine_reason_counts.get("NEGATIVE_COUNTER") == 1
    assert qm.quarantine_reason_counts.get("INVALID_TIMESTAMP") == 1

    # Buffer should contain exactly 3 valid flows
    assert len(ctx.buffer) == 3


# --------------------------------------------------------------------------
# Test 11: Buffer State and Latest Timestamp Unchanged After Quarantine
# --------------------------------------------------------------------------

def test_11_buffer_state_and_latest_timestamp_unchanged_after_quarantine():
    """Verifies that quarantined flows do NOT mutate buffer length, items, or latest_timestamp."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)

    # Initial valid flow
    df_valid = pd.DataFrame([
        {"Timestamp": "2026-11-04 10:00:00", "Total Fwd Packets": 2.0, "Total Length of Fwd Packets": 200.0},
        {"Timestamp": "2026-11-04 10:00:05", "Total Fwd Packets": 4.0, "Total Length of Fwd Packets": 400.0},
    ])
    ctx.process_batch(df_valid, causal_mode="simultaneous", flush=True)

    buf_len_before = len(ctx.buffer)
    ts_before = ctx.latest_timestamp
    buf_snapshot = list(ctx.buffer)

    # Send malformed flows
    df_malformed = pd.DataFrame([
        {"Timestamp": "2026-11-04 10:00:06", "Total Fwd Packets": -10.0, "Total Length of Fwd Packets": 100.0},
        {"Timestamp": "invalid_date", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 50.0},
    ])
    ctx.process_batch(df_malformed, causal_mode="simultaneous", flush=True)

    # State must be strictly identical
    assert len(ctx.buffer) == buf_len_before
    assert ctx.latest_timestamp == ts_before
    assert list(ctx.buffer) == buf_snapshot
    assert qm.total_quarantined_records == 2


# --------------------------------------------------------------------------
# Test 12: Quarantine Counters Increment Correctly
# --------------------------------------------------------------------------

def test_12_quarantine_counters_increment_correctly():
    """Verifies that FlowQuarantineManager accurately categorizes and tallies rejection reasons."""
    qm = FlowQuarantineManager()

    qm.quarantine({"Timestamp": None}, "MISSING_TIMESTAMP")
    qm.quarantine({"Timestamp": pd.NaT}, "INVALID_TIMESTAMP")
    qm.quarantine({"Total Fwd Packets": -1}, "NEGATIVE_COUNTER")
    qm.quarantine({"Total Fwd Packets": float("inf")}, "INFINITE_COUNTER")
    qm.quarantine({"Fwd Packet Length Max": "bad"}, "NON_NUMERIC_FEATURE")
    qm.quarantine({"Fwd Header Length": [1, 2]}, "UNEXPECTED_DATA_TYPE")
    qm.quarantine({}, "MISSING_REQUIRED_FIELD")

    metrics = qm.get_metrics()
    assert metrics["total_quarantined_records"] == 7
    assert metrics["invalid_timestamp_count"] == 2
    assert metrics["invalid_packet_byte_counter_count"] == 2
    assert metrics["invalid_numeric_feature_count"] == 2
    assert metrics["missing_required_field_count"] == 1
    assert metrics["recent_quarantined_samples_count"] == 7

    qm.reset()
    assert qm.total_quarantined_records == 0
    assert len(qm.recent_quarantined_samples) == 0


# --------------------------------------------------------------------------
# Test 13: Bounded Diagnostic Sample Storage
# --------------------------------------------------------------------------

def test_13_bounded_diagnostic_sample_storage():
    """Verifies that FlowQuarantineManager enforces max_diagnostic_samples FIFO limit."""
    ceiling = 100
    qm = FlowQuarantineManager(max_diagnostic_samples=ceiling)

    # Ingest 150 malformed records
    for i in range(150):
        qm.quarantine({"index": i, "Timestamp": f"invalid_{i}"}, "INVALID_TIMESTAMP")

    assert qm.total_quarantined_records == 150
    # Must be bounded by ceiling
    assert len(qm.recent_quarantined_samples) == ceiling
    # Oldest samples (0-49) evicted, newest samples (50-149) retained
    newest_sample = qm.recent_quarantined_samples[-1]
    assert "invalid_149" in newest_sample["timestamp"]


# --------------------------------------------------------------------------
# Test 14: Defensive v1 Fault Boundary Catches Exception
# --------------------------------------------------------------------------

def test_14_v1_champion_fault_boundary_catches_exception(synthetic_parquet_file, tmp_path):
    """
    Verifies that an unhandled exception inside ThreatExplainer.explain_flow()
    is safely caught by the fault boundary without crashing the simulator.
    """
    feed_file = tmp_path / "live_feed_v1_err.json"

    # Patch explain_flow to simulate an internal pipeline failure
    with patch.object(ThreatExplainer, "explain_flow", side_effect=RuntimeError("Simulated scikit-learn transform fault")):
        summary = run_stream_simulation(
            speed_eps=0,
            max_events=2,
            inject_attacks=False,
            chunk_size=2,
            v1_only=False,
            test_file=synthetic_parquet_file,
            live_feed_file=feed_file,
            write_feed=True,
            verbose=False,
        )

    # Simulation must complete successfully without crashing
    assert summary["total_processed"] == 2
    assert summary["total_anomalies"] == 0


# --------------------------------------------------------------------------
# Test 15: v1 Fault Boundary Never Fabricates Anomaly or Benign Verdict
# --------------------------------------------------------------------------

def test_15_v1_fault_boundary_does_not_fabricate_anomaly_or_benign(synthetic_parquet_file, tmp_path):
    """
    Verifies that when explain_flow fails:
    - verdict is strictly 'UNAVAILABLE (PROCESSING_ERROR)' (NOT 'ALERT', NOT 'PASS')
    - severity is 'UNAVAILABLE' (NOT 'NORMAL', NOT 'HIGH')
    - score is None
    - total_anomalies is NOT incremented
    """
    feed_file = tmp_path / "live_feed_fault.json"

    with patch.object(ThreatExplainer, "explain_flow", side_effect=ValueError("Feature matrix corrupted")):
        summary = run_stream_simulation(
            speed_eps=0,
            max_events=1,
            inject_attacks=False,
            chunk_size=1,
            v1_only=True,
            test_file=synthetic_parquet_file,
            live_feed_file=feed_file,
            write_feed=True,
            verbose=False,
        )

    assert summary["total_anomalies"] == 0

    with open(feed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    alert = data["recent_alerts"][0]
    assert alert["verdict"] == "UNAVAILABLE (PROCESSING_ERROR)"
    assert alert["severity"] == "UNAVAILABLE"
    assert alert["score"] is None
    assert "v1_error" in alert
    assert "ValueError" in alert["v1_error"]


# --------------------------------------------------------------------------
# Test 16: Subsequent Flow Processed Normally After v1 Failure
# --------------------------------------------------------------------------

def test_16_subsequent_flow_processed_normally_after_v1_failure(synthetic_parquet_file, tmp_path):
    """
    Verifies that after a transient v1 processing error on flow 1,
    flow 2 is processed normally with standard scoring and verdict.
    """
    feed_file = tmp_path / "live_feed_transient.json"

    real_explainer = ThreatExplainer()
    real_explain = real_explainer.explain_flow

    call_count = 0

    def mock_explain(features):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Transient pipeline exception")
        return real_explain(features)

    with patch.object(ThreatExplainer, "explain_flow", side_effect=mock_explain):
        summary = run_stream_simulation(
            speed_eps=0,
            max_events=2,
            inject_attacks=False,
            chunk_size=2,
            v1_only=True,
            test_file=synthetic_parquet_file,
            live_feed_file=feed_file,
            write_feed=True,
            verbose=False,
        )

    assert summary["total_processed"] == 2

    with open(feed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    alerts = data["recent_alerts"]
    # First flow failed safely
    assert alerts[0]["verdict"] == "UNAVAILABLE (PROCESSING_ERROR)"
    assert alerts[0]["score"] is None

    # Second flow succeeded with valid verdict
    assert alerts[1]["verdict"] in ["BENIGN (NORMAL)", "ANOMALOUS (THREAT DETECTED)"]
    assert alerts[1]["score"] is not None
    assert isinstance(alerts[1]["score"], float)


# --------------------------------------------------------------------------
# Test 17: v2 Mock Adapter Attribute Access Safety (Finding D1 Hardening)
# --------------------------------------------------------------------------

def test_17_v2_mock_adapter_attribute_access_safety_finding_d1(valid_flow_record):
    """
    Verifies that score_v2_shadow() safely tolerates a bare mock adapter lacking
    primary_threshold and reference_threshold attributes without raising AttributeError.
    """
    class BareMockAdapter:
        is_available = False
        load_error = "Mock adapter uninitialized"

    mock = BareMockAdapter()
    res = score_v2_shadow(valid_flow_record, mock)

    assert res["status"] == "UNAVAILABLE"
    assert res["primary_threshold"] == 0.314036
    assert res["reference_threshold"] == 0.099018
    assert res["score"] is None


# --------------------------------------------------------------------------
# Test 18: Full Regression Preservation and Quarantine Summary Telemetry
# --------------------------------------------------------------------------

def test_18_full_regression_preservation(synthetic_parquet_file, tmp_path):
    """
    Verifies full integration of Stage 4B.3A:
    - Normal valid streams run with quarantine_summary in feed and summary.
    - total_quarantined_records is 0 for completely valid data.
    - 32-dim Champion v1 and 26-dim Shadow v2 operate seamlessly together.
    """
    feed_file = tmp_path / "live_feed_clean.json"

    summary = run_stream_simulation(
        speed_eps=0,
        max_events=4,
        inject_attacks=False,
        chunk_size=2,
        v1_only=False,
        test_file=synthetic_parquet_file,
        live_feed_file=feed_file,
        write_feed=True,
        verbose=False,
    )

    assert summary["total_processed"] == 4
    assert "quarantine_summary" in summary
    assert summary["quarantine_summary"]["total_quarantined_records"] == 0

    with open(feed_file, "r", encoding="utf-8") as f:
        feed = json.load(f)

    assert "quarantine_summary" in feed
    assert feed["quarantine_summary"]["total_quarantined_records"] == 0
    assert len(feed["recent_alerts"]) == 4
    assert "v2_shadow" in feed["recent_alerts"][0]


# --------------------------------------------------------------------------
# Stage 4B.3A Remediation Tests (Findings F01, F02, F03)
# --------------------------------------------------------------------------

# --- F01: Single-Flow Streaming Validation Bypass Tests ---

def test_19_f01_process_flow_negative_packet_count():
    """F01: Verifies process_flow() quarantines negative packet counts and does not mutate buffer."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)
    malformed = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": -1.0,
        "Total Length of Fwd Packets": 100.0,
    }
    res = ctx.process_flow(malformed)
    assert res["_is_valid"] is False
    assert res["_context_state"] == "QUARANTINED"
    assert res["_quarantine_reason"] == "NEGATIVE_COUNTER"
    assert len(ctx.buffer) == 0
    assert qm.total_quarantined_records == 1
    assert qm.quarantine_reason_counts.get("NEGATIVE_COUNTER") == 1


def test_20_f01_process_flow_negative_byte_count():
    """F01: Verifies process_flow() quarantines negative byte counts and does not mutate buffer."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)
    malformed = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 5.0,
        "Total Length of Fwd Packets": -50.0,
    }
    res = ctx.process_flow(malformed)
    assert res["_is_valid"] is False
    assert res["_context_state"] == "QUARANTINED"
    assert res["_quarantine_reason"] == "NEGATIVE_COUNTER"
    assert len(ctx.buffer) == 0
    assert qm.total_quarantined_records == 1


def test_21_f01_process_flow_invalid_timestamp():
    """F01: Verifies process_flow() quarantines unparseable, NaT, or missing timestamps."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)

    # Unparseable date string
    res_str = ctx.process_flow({"Timestamp": "unparseable_date_xyz", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0})
    assert res_str["_is_valid"] is False
    assert res_str["_context_state"] == "QUARANTINED"
    assert res_str["_quarantine_reason"] == "INVALID_TIMESTAMP"

    # Missing timestamp
    res_missing = ctx.process_flow({"Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0})
    assert res_missing["_is_valid"] is False
    assert res_missing["_context_state"] == "QUARANTINED"
    assert res_missing["_quarantine_reason"] == "MISSING_TIMESTAMP"

    # pd.NaT timestamp
    res_nat = ctx.process_flow({"Timestamp": pd.NaT, "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 64.0})
    assert res_nat["_is_valid"] is False
    assert res_nat["_context_state"] == "QUARANTINED"
    assert res_nat["_quarantine_reason"] == "INVALID_TIMESTAMP"

    assert len(ctx.buffer) == 0
    assert qm.total_quarantined_records == 3


def test_22_f01_transform_flow_fast_invalid_record():
    """F01: Verifies transform_flow_fast() validates inputs, increments quarantine, and handles return_status."""
    v2 = PureOneWayV2Adapter()
    pipe = v2.pipeline if v2.is_available else None
    qm = FlowQuarantineManager()
    engine = StreamingFeatureEngine(pipeline=pipe, quarantine_manager=qm)

    malformed = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": -10.0,
        "Total Length of Fwd Packets": 100.0,
    }

    # return_status = True
    vec, is_valid, state = engine.transform_flow_fast(malformed, return_status=True)
    assert is_valid is False
    assert state == "QUARANTINED"
    assert vec.shape == (1, 26)
    assert np.all(np.isnan(vec))
    assert qm.total_quarantined_records == 1

    # return_status = False should raise ValueError
    with pytest.raises(ValueError, match="Flow record quarantined: NEGATIVE_COUNTER"):
        engine.transform_flow_fast(malformed, return_status=False)

    assert qm.total_quarantined_records == 2
    assert len(engine.context.buffer) == 0


def test_23_f01_valid_record_after_invalid_record(valid_flow_record):
    """F01: Verifies valid record processed after quarantined record succeeds cleanly."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)

    # 1. Invalid flow
    bad_flow = {"Timestamp": "2026-11-04 10:00:00", "Total Fwd Packets": -5.0, "Total Length of Fwd Packets": 50.0}
    bad_res = ctx.process_flow(bad_flow)
    assert bad_res["_is_valid"] is False
    assert len(ctx.buffer) == 0
    assert qm.total_quarantined_records == 1

    # 2. Valid flow
    good_res = ctx.process_flow(valid_flow_record)
    assert good_res["_is_valid"] is True
    assert good_res["_context_state"] == "HEALTHY"
    assert len(ctx.buffer) == 1
    assert qm.total_quarantined_records == 1  # unchanged


def test_24_f01_buffer_length_and_watermark_unchanged(valid_flow_record):
    """F01: Verifies buffer length, buffer items, and latest_timestamp strictly unchanged after invalid arrivals."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)

    # Ingest 1 valid flow
    valid_rec = dict(valid_flow_record)
    valid_rec["Timestamp"] = "2026-11-04 10:00:00"
    ctx.process_flow(valid_rec)

    initial_buf_len = len(ctx.buffer)
    initial_watermark = ctx.latest_timestamp
    initial_buf_content = list(ctx.buffer)

    # Multiple invalid attempts
    ctx.process_flow({"Timestamp": "2026-11-04 10:00:05", "Total Fwd Packets": -1.0, "Total Length of Fwd Packets": 100.0})
    ctx.process_flow({"Timestamp": "not-a-date", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 100.0})
    ctx.process_flow({"Timestamp": "2026-11-04 10:00:10", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": -100.0})

    assert len(ctx.buffer) == initial_buf_len
    assert ctx.latest_timestamp == initial_watermark
    assert list(ctx.buffer) == initial_buf_content
    assert qm.total_quarantined_records == 3


def test_25_f01_exactly_one_quarantine_entry_per_invalid_record():
    """F01: Verifies each invalid record produces exactly one quarantine entry without duplication."""
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)

    for i in range(5):
        flow = {"Timestamp": f"2026-11-04 10:00:0{i}", "Total Fwd Packets": -1.0 * (i + 1), "Total Length of Fwd Packets": 10.0}
        ctx.process_flow(flow)
        assert qm.total_quarantined_records == i + 1

    assert qm.get_metrics()["total_quarantined_records"] == 5
    assert len(qm.recent_quarantined_samples) == 5


def test_26_f01_single_flow_and_batch_numerical_parity(valid_flow_record):
    """F01: Verifies exact numerical parity between sequential process_flow() and process_batch() on valid flows."""
    rows = []
    for i in range(5):
        row = dict(valid_flow_record)
        row["Timestamp"] = f"2026-11-04 10:00:0{i}"
        row["Total Fwd Packets"] = float(i + 1)
        row["Total Length of Fwd Packets"] = float(100 * (i + 1))
        row["Fwd IAT Total"] = float(500 * (i + 1))
        rows.append(row)

    # Batch execution
    ctx_batch = StreamingCausalContext()
    batch_df = ctx_batch.process_batch(pd.DataFrame(rows), causal_mode="simultaneous", flush=True)

    # Single-flow execution
    ctx_single = StreamingCausalContext()
    single_rows = [ctx_single.process_flow(r) for r in rows]
    single_df = pd.DataFrame(single_rows)

    cols_to_compare = [
        "Fwd_Bytes_Per_Sec",
        "Fwd_Packets_Per_Sec",
        "Rolling_Flow_Velocity_10s",
        "Rolling_Bytes_Velocity_10s",
        "Rolling_Single_Pkt_Ratio_10s",
    ]

    for col in cols_to_compare:
        np.testing.assert_allclose(
            batch_df[col].values,
            single_df[col].values,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Numerical mismatch on column {col}",
        )


# --- F02: score_v2_shadow() Threshold Failure Isolation Tests ---

def test_27_f02_score_v2_shadow_primary_threshold_none(valid_flow_record):
    """F02: Verifies score_v2_shadow() returns INVALID_THRESHOLD when primary_threshold is None."""
    mock = MagicMock()
    mock.is_available = True
    mock.primary_threshold = None
    mock.reference_threshold = 0.099018

    res = score_v2_shadow(valid_flow_record, mock)
    assert res["status"] == "UNAVAILABLE"
    assert res["error"] == "INVALID_THRESHOLD"
    assert "primary threshold is None" in res["error_message"]
    assert res["score"] is None
    assert res["primary_threshold"] is None


def test_28_f02_score_v2_shadow_reference_threshold_none(valid_flow_record):
    """F02: Verifies score_v2_shadow() returns INVALID_THRESHOLD when reference_threshold is None."""
    mock = MagicMock()
    mock.is_available = True
    mock.primary_threshold = 0.314036
    mock.reference_threshold = None

    res = score_v2_shadow(valid_flow_record, mock)
    assert res["status"] == "UNAVAILABLE"
    assert res["error"] == "INVALID_THRESHOLD"
    assert "reference threshold is None" in res["error_message"]
    assert res["score"] is None
    assert res["reference_threshold"] is None


def test_29_f02_score_v2_shadow_missing_threshold_attribute(valid_flow_record):
    """F02: Verifies score_v2_shadow() returns INVALID_THRESHOLD when threshold attribute is absent."""
    class NoPrimaryMock:
        is_available = True
        reference_threshold = 0.099018

    res = score_v2_shadow(valid_flow_record, NoPrimaryMock())
    assert res["status"] == "UNAVAILABLE"
    assert res["error"] == "INVALID_THRESHOLD"
    assert "missing 'primary_threshold'" in res["error_message"]


def test_30_f02_score_v2_shadow_nan_threshold(valid_flow_record):
    """F02: Verifies score_v2_shadow() rejects NaN thresholds."""
    mock = MagicMock()
    mock.is_available = True
    mock.primary_threshold = float("nan")
    mock.reference_threshold = 0.099018

    res = score_v2_shadow(valid_flow_record, mock)
    assert res["status"] == "UNAVAILABLE"
    assert res["error"] == "INVALID_THRESHOLD"
    assert "NaN" in res["error_message"]


def test_31_f02_score_v2_shadow_infinite_threshold(valid_flow_record):
    """F02: Verifies score_v2_shadow() rejects infinite thresholds (+inf, -inf)."""
    mock = MagicMock()
    mock.is_available = True
    mock.primary_threshold = float("inf")
    mock.reference_threshold = 0.099018

    res = score_v2_shadow(valid_flow_record, mock)
    assert res["status"] == "UNAVAILABLE"
    assert res["error"] == "INVALID_THRESHOLD"
    assert "infinite" in res["error_message"]


def test_32_f02_score_v2_shadow_invalid_string_threshold(valid_flow_record):
    """F02: Verifies score_v2_shadow() rejects unparseable string thresholds."""
    mock = MagicMock()
    mock.is_available = True
    mock.primary_threshold = "corrupted_threshold_val"
    mock.reference_threshold = 0.099018

    res = score_v2_shadow(valid_flow_record, mock)
    assert res["status"] == "UNAVAILABLE"
    assert res["error"] == "INVALID_THRESHOLD"
    assert "could not be converted to float" in res["error_message"]


def test_33_f02_v2_failure_followed_by_successful_v1_inference(synthetic_parquet_file, tmp_path):
    """
    F02: Verifies that when v2 shadow fails due to None threshold,
    Champion v1 continues running to completion, generates valid alerts/pass, and feed records v2 unavailable.
    """
    feed_file = tmp_path / "live_feed_v2_thresh_fail.json"

    class BadThreshAdapter:
        is_available = True
        primary_threshold = None
        reference_threshold = 0.099018
        pipeline = None
        feature_names = list(PURE_ONE_WAY_FEATURES)

    with patch("simulate_stream.PureOneWayV2Adapter", return_value=BadThreshAdapter()):
        summary = run_stream_simulation(
            speed_eps=0,
            max_events=3,
            inject_attacks=False,
            chunk_size=2,
            v1_only=False,
            test_file=synthetic_parquet_file,
            live_feed_file=feed_file,
            write_feed=True,
            verbose=False,
        )

    # v1 completed fully
    assert summary["total_processed"] == 3
    # v2 was completely isolated and recorded as unavailable
    assert summary["v2_unavailable_count"] == 3
    assert summary["v2_primary_anomalies"] == 0

    with open(feed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # All alerts processed by v1 successfully
    for alert in data["recent_alerts"]:
        assert alert["verdict"] in ["BENIGN (NORMAL TRAFFIC)", "ANOMALOUS (THREAT DETECTED)", "BENIGN (NORMAL)"]
        assert alert["v2_shadow"]["status"] == "UNAVAILABLE"
        assert alert["v2_shadow"]["error"] == "INVALID_THRESHOLD"


# --- F03: Explicit Boolean Rejection Tests ---

def test_34_f03_true_as_total_fwd_packets():
    """F03: Verifies that boolean True as Total Fwd Packets is rejected with UNEXPECTED_DATA_TYPE."""
    rec = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": True,
        "Total Length of Fwd Packets": 100.0,
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec)
    assert is_val is False
    assert reason == "UNEXPECTED_DATA_TYPE"
    assert san is None


def test_35_f03_false_as_total_length_of_fwd_packets():
    """F03: Verifies that boolean False as Total Length of Fwd Packets is rejected with UNEXPECTED_DATA_TYPE."""
    rec = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 1.0,
        "Total Length of Fwd Packets": False,
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec)
    assert is_val is False
    assert reason == "UNEXPECTED_DATA_TYPE"
    assert san is None


def test_36_f03_boolean_in_other_numeric_features():
    """F03: Verifies that booleans in other numeric features are rejected with UNEXPECTED_DATA_TYPE."""
    rec1 = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 1.0,
        "Total Length of Fwd Packets": 100.0,
        "Fwd Header Length": True,
    }
    is_val1, reason1, san1 = validate_and_sanitize_flow_record(rec1)
    assert is_val1 is False
    assert reason1 == "UNEXPECTED_DATA_TYPE"

    rec2 = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 1.0,
        "Total Length of Fwd Packets": 100.0,
        "Fwd Packet Length Max": False,
    }
    is_val2, reason2, san2 = validate_and_sanitize_flow_record(rec2)
    assert is_val2 is False
    assert reason2 == "UNEXPECTED_DATA_TYPE"


def test_37_f03_normal_integer_and_float_controls():
    """F03: Verifies that valid integer and float values pass validation without false positives."""
    rec = {
        "Timestamp": "2026-11-04 10:00:00",
        "Total Fwd Packets": 5,
        "Total Length of Fwd Packets": 500,
        "Fwd Header Length": 32,
        "Fwd Packet Length Mean": 50.5,
    }
    is_val, reason, san = validate_and_sanitize_flow_record(rec)
    assert is_val is True
    assert reason is None
    assert san["Total Fwd Packets"] == 5.0
    assert isinstance(san["Total Fwd Packets"], float)
    assert san["Total Length of Fwd Packets"] == 500.0
    assert isinstance(san["Total Length of Fwd Packets"], float)
    assert san["Fwd Header Length"] == 32.0
    assert san["Fwd Packet Length Mean"] == 50.5
