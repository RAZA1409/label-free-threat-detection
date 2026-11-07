"""
tests/test_temporal_replay_fidelity.py
AEGIS Phase 3: Temporal-Fidelity Chronological Replay & Window Selection Verification.

SIH Problem ID: SIH26145
Compliance: Pure one-way causal streaming; deterministic contiguous window replay;
temporal velocity preservation without synthetic fabrication or data leakage.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add repository paths for test execution
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from streaming_features import (
    prepare_chronological_stream,
    find_contiguous_replay_window,
    StreamingFeatureEngine,
    FlowQuarantineManager,
)


@pytest.fixture
def synthetic_attack_stream():
    """
    Creates a deterministic synthetic capture with known temporal structure:
    - 100 Benign flows (1 flow every second: 00:00:00 to 00:01:39)
    - 50 DDoS flows (high frequency, 25 flows/sec: 00:01:40 to 00:01:41)
    - 100 Benign flows (1 flow every second: 00:01:42 to 00:03:21)
    """
    rows = []
    uid = 0

    # 1. Baseline benign
    for i in range(100):
        ts = pd.Timestamp("2026-11-06 10:00:00") + pd.Timedelta(seconds=i)
        rows.append({
            "flow_id": f"flow_{uid}",
            "Timestamp": ts,
            "Total Fwd Packets": 2.0,
            "Total Length of Fwd Packets": 100.0,
            "Fwd IAT Total": 1_000_000.0,
            "Label": "BENIGN",
        })
        uid += 1

    # 2. Dense DDoS burst (50 flows across 2 seconds)
    for i in range(50):
        sec_offset = i // 25
        ms_offset = (i % 25) * 40
        ts = pd.Timestamp("2026-11-06 10:01:40") + pd.Timedelta(seconds=sec_offset, milliseconds=ms_offset)
        rows.append({
            "flow_id": f"flow_{uid}",
            "Timestamp": ts,
            "Total Fwd Packets": 1.0,
            "Total Length of Fwd Packets": 40.0,
            "Fwd IAT Total": 40_000.0,
            "Label": "DDoS",
        })
        uid += 1

    # 3. Post-attack benign
    for i in range(100):
        ts = pd.Timestamp("2026-11-06 10:01:42") + pd.Timedelta(seconds=i)
        rows.append({
            "flow_id": f"flow_{uid}",
            "Timestamp": ts,
            "Total Fwd Packets": 3.0,
            "Total Length of Fwd Packets": 150.0,
            "Fwd IAT Total": 1_000_000.0,
            "Label": "BENIGN",
        })
        uid += 1

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# TEST 01: Deterministic Contiguous-Window Selection
# -----------------------------------------------------------------------------
def test_01_deterministic_contiguous_window_selection(synthetic_attack_stream):
    """Verify that window selection is 100% deterministic and contiguous."""
    df = synthetic_attack_stream.copy()

    res1 = prepare_chronological_stream(df, max_events=80, inject_attacks=True, attack_ratio=0.25)
    res2 = prepare_chronological_stream(df, max_events=80, inject_attacks=True, attack_ratio=0.25)

    assert len(res1) == 80
    assert len(res2) == 80
    assert res1["flow_id"].tolist() == res2["flow_id"].tolist()
    assert res1.attrs["window_start_idx"] == res2.attrs["window_start_idx"]

    # Verify contiguity against original sorted capture
    sorted_orig = df.sort_values(by="Timestamp").reset_index(drop=True)
    w_start = res1.attrs["window_start_idx"]
    w_end = res1.attrs["window_end_idx"]
    expected_uids = sorted_orig.iloc[w_start:w_end]["flow_id"].tolist()
    assert res1["flow_id"].tolist() == expected_uids


# -----------------------------------------------------------------------------
# TEST 02: Chronological Ordering
# -----------------------------------------------------------------------------
def test_02_chronological_ordering(synthetic_attack_stream):
    """Verify monotonic non-decreasing event-time ordering: t_{i+1} >= t_i."""
    shuffled = synthetic_attack_stream.sample(frac=1.0, random_state=123).reset_index(drop=True)
    res = prepare_chronological_stream(shuffled, max_events=120, inject_attacks=True)

    assert res["Timestamp"].is_monotonic_increasing
    for i in range(len(res) - 1):
        assert res["Timestamp"].iloc[i + 1] >= res["Timestamp"].iloc[i]


# -----------------------------------------------------------------------------
# TEST 03: Duplicate Timestamp Stable Ordering
# -----------------------------------------------------------------------------
def test_03_duplicate_timestamp_stable_ordering():
    """Verify that flows with identical timestamps preserve original arrival sequence."""
    base_ts = pd.Timestamp("2026-11-06 12:00:00")
    df = pd.DataFrame({
        "flow_id": [f"f_{i}" for i in range(6)],
        "Timestamp": [base_ts, base_ts, base_ts + pd.Timedelta(seconds=1), base_ts, base_ts + pd.Timedelta(seconds=1), base_ts],
        "Label": ["BENIGN"] * 6,
    })

    res = prepare_chronological_stream(df)
    # The four flows at base_ts should appear in original order: f_0, f_1, f_3, f_5
    base_ts_flows = res[res["Timestamp"] == base_ts]["flow_id"].tolist()
    assert base_ts_flows == ["f_0", "f_1", "f_3", "f_5"]


# -----------------------------------------------------------------------------
# TEST 04: No Duplicate Flow Records
# -----------------------------------------------------------------------------
def test_04_no_duplicate_flow_records(synthetic_attack_stream):
    """Verify zero record duplication in the resulting chronological stream."""
    res = prepare_chronological_stream(synthetic_attack_stream, max_events=100, inject_attacks=True)
    uids = res["flow_id"].tolist()
    assert len(uids) == len(set(uids))


# -----------------------------------------------------------------------------
# TEST 05: Original Timestamps Preserved Exactly
# -----------------------------------------------------------------------------
def test_05_original_timestamps_preserved_exactly(synthetic_attack_stream):
    """Verify timestamps are preserved bit-for-bit without jitter, rounding, or shifting."""
    res = prepare_chronological_stream(synthetic_attack_stream, max_events=50, inject_attacks=True)
    orig_indexed = synthetic_attack_stream.set_index("flow_id")

    for _, row in res.iterrows():
        orig_ts = orig_indexed.loc[row["flow_id"], "Timestamp"]
        assert row["Timestamp"] == orig_ts


# -----------------------------------------------------------------------------
# TEST 06: Authentic 10-Second Rolling Velocity Preserved
# -----------------------------------------------------------------------------
def test_06_authentic_10s_rolling_velocity_preserved(synthetic_attack_stream):
    """
    Verify that contiguous window preserves the real physical flow velocity,
    which sparse multi-day subsampling destroys.
    """
    res = prepare_chronological_stream(synthetic_attack_stream, max_events=100, inject_attacks=True)
    qm = FlowQuarantineManager()
    engine = StreamingFeatureEngine(window_seconds=10, max_buffer_capacity=1000, clock_mode="event_time", quarantine_manager=qm)

    augmented = engine.context.process_batch(res, causal_mode="simultaneous", flush=True)

    # During the dense DDoS burst (50 flows in 2 seconds), velocity should reach at least 50
    ddos_augmented = augmented[augmented["Label"] == "DDoS"]
    assert len(ddos_augmented) > 0
    max_vel = ddos_augmented["Rolling_Flow_Velocity_10s"].max()
    assert max_vel >= 50.0, f"Expected velocity >= 50.0 during burst, got {max_vel}"


# -----------------------------------------------------------------------------
# TEST 07: Real Attack Burst and Surrounding Benign Traffic Retained
# -----------------------------------------------------------------------------
def test_07_real_attack_burst_and_surrounding_benign_traffic_retained(synthetic_attack_stream):
    """Verify selected window captures both pre-attack baseline and attack burst."""
    res = prepare_chronological_stream(synthetic_attack_stream, max_events=80, inject_attacks=True, attack_ratio=0.25)
    labels = res["Label"].tolist()

    assert "BENIGN" in labels
    assert "DDoS" in labels

    # Confirm pre-attack benign lead-in exists before first attack flow
    first_attack_pos = labels.index("DDoS")
    assert first_attack_pos > 0, "Window should contain benign lead-in traffic before attack onset"


# -----------------------------------------------------------------------------
# TEST 08: No Artificial Attack Injection
# -----------------------------------------------------------------------------
def test_08_no_artificial_attack_injection(synthetic_attack_stream):
    """Verify that all records returned originated from the input dataset."""
    res = prepare_chronological_stream(synthetic_attack_stream, max_events=70, inject_attacks=True)
    orig_uids = set(synthetic_attack_stream["flow_id"])

    for uid in res["flow_id"]:
        assert uid in orig_uids


# -----------------------------------------------------------------------------
# TEST 09: Correct Behavior When No Attack Interval Exists
# -----------------------------------------------------------------------------
def test_09_correct_behavior_when_no_attack_interval_exists():
    """Verify graceful contiguous fallback when dataset contains only benign traffic."""
    n = 100
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2026-11-06 10:00:00", periods=n, freq="1s"),
        "Total Fwd Packets": [1.0] * n,
        "Total Length of Fwd Packets": [50.0] * n,
        "Fwd IAT Total": [1000.0] * n,
        "Label": ["BENIGN"] * n,
    })

    res = prepare_chronological_stream(df, max_events=40, inject_attacks=True)
    assert len(res) == 40
    assert res.attrs["selection_mode"] == "contiguous_head"
    assert (res["Label"] == "BENIGN").all()


# -----------------------------------------------------------------------------
# TEST 10: Behavior When Event Count Is Smaller Than Natural Attack Burst
# -----------------------------------------------------------------------------
def test_10_behavior_when_requested_count_smaller_than_attack_interval():
    """Verify deterministic lead-in and truncation policy when burst exceeds max_events."""
    n = 300
    labels = ["BENIGN"] * 50 + ["DDoS"] * 200 + ["BENIGN"] * 50
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2026-11-06 10:00:00", periods=n, freq="100ms"),
        "Total Fwd Packets": [1.0] * n,
        "Total Length of Fwd Packets": [40.0] * n,
        "Fwd IAT Total": [100.0] * n,
        "Label": labels,
    })

    max_events = 60
    res = prepare_chronological_stream(df, max_events=max_events, inject_attacks=True)
    assert len(res) == max_events

    # 20% of 60 = 12 benign lead-in flows
    w_labels = res["Label"].tolist()
    first_ddos_idx = w_labels.index("DDoS")
    assert first_ddos_idx == 12
    assert "DDoS" in res.attrs["selection_note"]


# -----------------------------------------------------------------------------
# TEST 11: Behavior When Source Dataset Is Smaller Than max_events
# -----------------------------------------------------------------------------
def test_11_behavior_when_source_dataset_smaller_than_max_events():
    """Verify full dataset returned safely when length < max_events or max_events is None."""
    n = 25
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2026-11-06 10:00:00", periods=n, freq="500ms"),
        "Label": ["BENIGN"] * n,
    })

    # max_events > n
    res1 = prepare_chronological_stream(df, max_events=100)
    assert len(res1) == 25
    assert res1.attrs["selection_mode"] == "full_dataset"

    # max_events is None
    res2 = prepare_chronological_stream(df, max_events=None)
    assert len(res2) == 25
    assert res2.attrs["selection_mode"] == "full_dataset"


# -----------------------------------------------------------------------------
# TEST 12: Existing Timestamp Metadata Remains Correct
# -----------------------------------------------------------------------------
def test_12_existing_timestamp_metadata_remains_correct(synthetic_attack_stream):
    """Verify required metadata attributes are correctly set."""
    res = prepare_chronological_stream(synthetic_attack_stream, max_events=50, inject_attacks=True)

    assert "min_timestamp" in res.attrs
    assert "max_timestamp" in res.attrs
    assert "trusted_horizon" in res.attrs
    assert "selection_mode" in res.attrs
    assert "window_start_idx" in res.attrs
    assert "window_end_idx" in res.attrs
    assert "selection_note" in res.attrs

    expected_max = res["Timestamp"].max()
    expected_min = res["Timestamp"].min()
    assert res.attrs["max_timestamp"] == expected_max
    assert res.attrs["min_timestamp"] == expected_min
    assert res.attrs["trusted_horizon"] == expected_max + pd.Timedelta(seconds=10.0)


# -----------------------------------------------------------------------------
# TEST 13: Synthetic Temporal Burst Integration Test
# -----------------------------------------------------------------------------
def test_13_synthetic_temporal_burst_integration():
    """
    Focused integration test using synthetic deterministic dataset with known temporal burst:
    - 200 Benign background flows (sparse: 1 every 2s)
    - 150 Attack flows (dense: 50 flows/sec over 3s)
    - 100 Post-burst flows
    Verifies that streaming feature engine computes realistic temporal velocities without data leakage.
    """
    rows = []
    t_start = pd.Timestamp("2026-11-06 14:00:00")

    # Background
    for i in range(200):
        rows.append({
            "Timestamp": t_start + pd.Timedelta(seconds=i * 2),
            "Total Fwd Packets": 2.0,
            "Total Length of Fwd Packets": 120.0,
            "Fwd IAT Total": 2_000_000.0,
            "Label": "BENIGN",
        })

    # Dense burst
    t_burst = t_start + pd.Timedelta(seconds=400)
    for i in range(150):
        ms = (i % 50) * 20
        sec = i // 50
        rows.append({
            "Timestamp": t_burst + pd.Timedelta(seconds=sec, milliseconds=ms),
            "Total Fwd Packets": 1.0,
            "Total Length of Fwd Packets": 40.0,
            "Fwd IAT Total": 20_000.0,
            "Label": "DDoS",
        })

    # Post-burst
    t_post = t_burst + pd.Timedelta(seconds=5)
    for i in range(100):
        rows.append({
            "Timestamp": t_post + pd.Timedelta(seconds=i * 2),
            "Total Fwd Packets": 2.0,
            "Total Length of Fwd Packets": 120.0,
            "Fwd IAT Total": 2_000_000.0,
            "Label": "BENIGN",
        })

    df = pd.DataFrame(rows)

    # Replay 200 flows around the burst
    stream_pool = prepare_chronological_stream(df, max_events=200, inject_attacks=True, attack_ratio=0.25, preferred_attack="DDoS")

    assert len(stream_pool) == 200
    assert stream_pool["Timestamp"].is_monotonic_increasing

    # Stream through Causal Engine
    qm = FlowQuarantineManager()
    engine = StreamingFeatureEngine(window_seconds=10, max_buffer_capacity=1000, clock_mode="event_time", quarantine_manager=qm)
    augmented = engine.context.process_batch(stream_pool, causal_mode="simultaneous", flush=True)

    # Check velocity
    bg_flows = augmented[augmented["Label"] == "BENIGN"]
    burst_flows = augmented[augmented["Label"] == "DDoS"]

    assert len(burst_flows) > 0
    # Background velocity is ~5 flows per 10s (1 every 2s)
    # Burst velocity should reach > 100 flows per 10s
    assert bg_flows["Rolling_Flow_Velocity_10s"].iloc[0] <= 5.0
    assert burst_flows["Rolling_Flow_Velocity_10s"].max() >= 100.0
