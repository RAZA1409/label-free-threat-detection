"""
tests/test_timestamp_resilience.py
AEGIS Phase 3 Stage 4B.3B: Timestamp Anomaly & Watermark Protection Test Suite.

SIH Problem ID: SIH26145
Compliance: 26-dimensional forward-feature-only design; reverse-direction feature information excluded.

Verifies:
1. T01: Year-2099 future spike attack quarantined without buffer mutation or watermark regression.
2. T02: Legitimate multi-hour / overnight replay gaps (43,260s) accepted naturally without arbitrary thresholds.
3. T03: Exact future drift boundary in wall-clock mode accepted.
4. T04: Just-over future drift boundary in wall-clock mode quarantined.
5. T05: Direct NaT, None, NaN, Inf, boolean timestamp ingestion quarantined before state mutation.
6. T06: In-window out-of-order flows sorted chronologically without regressing watermark.
7. T07: Expired late arrivals evaluated in single-event isolation without entering buffer.
8. T08: Duplicate timestamps within a chunk receive identical complete aggregate matching offline bit-for-bit.
9. T09: Duplicate timestamps across chunk boundaries with flush=False maintain complete aggregation.
10. T10: Safe recovery after rejected future spike.
11. T11: Strict watermark monotonicity across adversarial stream.
12. T12: Buffer snapshot immutability on rejected flow arrival.
13. T13: Safe capacity overflow and degraded recovery across time jumps.
14. T14: Numerical parity between single-flow and batch processing across mixed anomalies.
15. T15: Warm-up buffer rejects NaT, unparseable strings, and horizon exceedances.
16. T16: Stream simulation trusted horizon integration and chunked resilience.
"""

import sys
import math
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (
    PURE_ONE_WAY_FEATURES,
    add_temporal_velocity_features,
    PureOneWayLogPreprocessingPipeline,
)
from streaming_features import (
    StreamingCausalContext,
    StreamingFeatureEngine,
    FlowQuarantineManager,
    validate_and_sanitize_flow_record,
    validate_timestamp_strict,
    prepare_chronological_stream,
    DEFAULT_EVENT_TIME_HORIZON,
)


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
        "Flow IAT Mean": 200.0,
        "Flow IAT Std": 50.0,
        "Flow IAT Max": 500.0,
        "Flow IAT Min": 10.0,
        "Fwd IAT Mean": 200.0,
        "Fwd IAT Std": 50.0,
        "Fwd IAT Max": 500.0,
        "Fwd IAT Min": 10.0,
        "Fwd Header Length": 100.0,
        "Min Packet Length": 20.0,
        "Max Packet Length": 100.0,
        "Packet Length Mean": 50.0,
        "Packet Length Std": 10.0,
        "Packet Length Variance": 100.0,
        "ACK Flag Count": 1.0,
        "Down/Up Ratio": 0.0,
        "Average Packet Size": 50.0,
        "Avg Fwd Segment Size": 50.0,
        "Init_Win_bytes_forward": 8192.0,
        "act_data_pkt_fwd": 4.0,
        "Subflow Fwd Packets": 5.0,
        "Subflow Fwd Bytes": 500.0,
        "Fwd PSH Flags": 0.0,
        "Fwd URG Flags": 0.0,
        "SYN Flag Count": 1.0,
        "min_seg_size_forward": 20.0,
        "Label": "BENIGN",
    }


# --------------------------------------------------------------------------
# T01: Year-2099 Future Spike Attack
# --------------------------------------------------------------------------
def test_01_year_2099_future_spike_attack(valid_flow_record):
    """
    Verifies that an adversarial or corrupted year-2099 timestamp is quarantined
    with TIMESTAMP_FUTURE_DRIFT, leaving buffer length, items, and watermark strictly unchanged.
    """
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(
        window_seconds=10,
        clock_mode="event_time",
        max_timestamp_horizon=pd.Timestamp("2030-01-01 00:00:00"),
        quarantine_manager=qm,
    )

    # Ingest 3 clean flows
    for i in range(3):
        rec = dict(valid_flow_record)
        rec["Timestamp"] = f"2026-11-04 10:00:0{i}"
        ctx.process_flow(rec)

    assert len(ctx.buffer) == 3
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:02")
    buf_snapshot = list(ctx.buffer)

    # Ingest rogue year-2099 flow
    rogue_rec = dict(valid_flow_record)
    rogue_rec["Timestamp"] = "2099-01-01 12:00:00"
    out = ctx.process_flow(rogue_rec)

    # Verification: quarantined, buffer untouched, watermark frozen
    assert out["_context_state"] == "QUARANTINED"
    assert out["_quarantine_reason"] == "TIMESTAMP_FUTURE_DRIFT"
    assert len(ctx.buffer) == 3
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:02")
    assert list(ctx.buffer) == buf_snapshot
    assert qm.total_quarantined_records == 1
    assert qm.quarantine_reason_counts["TIMESTAMP_FUTURE_DRIFT"] == 1
    assert qm.invalid_timestamp_count == 1


# --------------------------------------------------------------------------
# T02: Legitimate Multi-Hour Replay Gap (43,260s)
# --------------------------------------------------------------------------
def test_02_legitimate_multi_hour_replay_gap(valid_flow_record):
    """
    Verifies that a legitimate multi-hour / overnight replay pause (such as the 43,260s
    gap in CIC-IDS-2017) is accepted as monotonic progression without being rejected.
    """
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(
        window_seconds=10,
        clock_mode="event_time",
        max_timestamp_horizon=pd.Timestamp("2017-07-06 00:00:00"),
        quarantine_manager=qm,
    )

    # Tuesday evening flow: 17:00:00
    rec_tue = dict(valid_flow_record)
    rec_tue["Timestamp"] = "2017-07-04 17:00:00"
    out_tue = ctx.process_flow(rec_tue)
    assert out_tue["_context_state"] == "HEALTHY"
    assert ctx.latest_timestamp == pd.Timestamp("2017-07-04 17:00:00")
    assert len(ctx.buffer) == 1

    # Wednesday morning flow: 09:00:00 (43,260 seconds later)
    rec_wed = dict(valid_flow_record)
    rec_wed["Timestamp"] = "2017-07-05 09:00:00"
    out_wed = ctx.process_flow(rec_wed)
    assert out_wed["_context_state"] == "HEALTHY"

    # Watermark advanced to Wednesday, Tuesday flow cleanly evicted by 10s window
    assert ctx.latest_timestamp == pd.Timestamp("2017-07-05 09:00:00")
    assert len(ctx.buffer) == 1
    assert ctx.buffer[0][0] == pd.Timestamp("2017-07-05 09:00:00")
    assert qm.total_quarantined_records == 0


# --------------------------------------------------------------------------
# T03: Exact Future Drift Boundary (Wall-Clock Mode)
# --------------------------------------------------------------------------
def test_03_exact_future_drift_boundary_wall_clock(valid_flow_record):
    """
    Verifies that a flow arriving at exactly T_wall + max_future_drift_seconds (30.0s)
    is accepted in wall-clock mode using an injectable deterministic clock.
    """
    mock_wall = pd.Timestamp("2026-11-05 12:00:00")
    ctx = StreamingCausalContext(
        window_seconds=10,
        clock_mode="wall_clock",
        max_future_drift_seconds=30.0,
        current_time_fn=lambda: mock_wall,
    )

    # Flow at exactly T_wall + 30.0s
    rec = dict(valid_flow_record)
    rec["Timestamp"] = "2026-11-05 12:00:30.000000"
    out = ctx.process_flow(rec)

    assert out["_context_state"] == "HEALTHY"
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-05 12:00:30")
    assert len(ctx.buffer) == 1


# --------------------------------------------------------------------------
# T04: Just-Over Future Drift Boundary (Wall-Clock Mode)
# --------------------------------------------------------------------------
def test_04_just_over_future_drift_boundary_wall_clock(valid_flow_record):
    """
    Verifies that a flow arriving at T_wall + 30.001s (just over tolerance)
    is quarantined with TIMESTAMP_FUTURE_DRIFT in wall-clock mode.
    """
    mock_wall = pd.Timestamp("2026-11-05 12:00:00")
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(
        window_seconds=10,
        clock_mode="wall_clock",
        max_future_drift_seconds=30.0,
        current_time_fn=lambda: mock_wall,
        quarantine_manager=qm,
    )

    # Flow at T_wall + 30.001s
    rec = dict(valid_flow_record)
    rec["Timestamp"] = "2026-11-05 12:00:30.001000"
    out = ctx.process_flow(rec)

    assert out["_context_state"] == "QUARANTINED"
    assert out["_quarantine_reason"] == "TIMESTAMP_FUTURE_DRIFT"
    assert ctx.latest_timestamp is None
    assert len(ctx.buffer) == 0
    assert qm.total_quarantined_records == 1
    assert qm.quarantine_reason_counts["TIMESTAMP_FUTURE_DRIFT"] == 1


# --------------------------------------------------------------------------
# T05: Direct NaT, None, NaN, Inf, Boolean Ingestion Rejection
# --------------------------------------------------------------------------
def test_05_direct_nat_none_unparseable_ingestion():
    """
    Verifies that direct invocations of update() with invalid timestamps
    (NaT, None, NaN, Inf, bool, string garbage) never mutate buffer or watermark.
    """
    qm = FlowQuarantineManager()
    ctx = StreamingCausalContext(quarantine_manager=qm)

    invalid_inputs = [
        (pd.NaT, "INVALID_TIMESTAMP"),
        (None, "MISSING_TIMESTAMP"),
        (float("nan"), "INVALID_TIMESTAMP"),
        (float("inf"), "INVALID_TIMESTAMP"),
        (True, "INVALID_TIMESTAMP"),
        (False, "INVALID_TIMESTAMP"),
        ("not-a-timestamp", "INVALID_TIMESTAMP"),
        ("2099-01-01 00:00:00", "TIMESTAMP_FUTURE_DRIFT"),
    ]

    for val, expected_reason in invalid_inputs:
        rf, rb, rs = ctx.update(val, 1.0, 100.0)
        # Evaluated in safe single-flow isolation
        assert rf == 1.0
        assert rb == 100.0
        assert rs == 1.0
        assert len(ctx.buffer) == 0
        assert ctx.latest_timestamp is None

    assert qm.total_quarantined_records == len(invalid_inputs)

    # Ensure subsequent valid flow functions normally
    rf, rb, rs = ctx.update("2026-11-04 10:00:00", 2.0, 200.0)
    assert len(ctx.buffer) == 1
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:00")
    assert rf == 1.0
    assert rb == 200.0


# --------------------------------------------------------------------------
# T06: In-Window Out-of-Order Sorting
# --------------------------------------------------------------------------
def test_06_in_window_out_of_order_sorting(valid_flow_record):
    """
    Verifies that an out-of-order flow arriving within the 10-second causal window
    is inserted in sorted chronological order without regressing the watermark.
    """
    ctx = StreamingCausalContext(window_seconds=10)

    # Flow 1 at 10:00:10
    rec1 = dict(valid_flow_record)
    rec1["Timestamp"] = "2026-11-04 10:00:10"
    ctx.process_flow(rec1)

    # Flow 2 at 10:00:15 (watermark moves to 10:00:15)
    rec2 = dict(valid_flow_record)
    rec2["Timestamp"] = "2026-11-04 10:00:15"
    ctx.process_flow(rec2)
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:15")

    # Flow 3 at 10:00:12 (in-window late: 15 - 10s = 05 < 12 < 15)
    rec3 = dict(valid_flow_record)
    rec3["Timestamp"] = "2026-11-04 10:00:12"
    out3 = ctx.process_flow(rec3)

    assert out3["_context_state"] == "HEALTHY"
    # Watermark did NOT regress backward
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:15")
    # Buffer contains all 3 entries in sorted order
    assert len(ctx.buffer) == 3
    buf_ts = [item[0] for item in ctx.buffer]
    assert buf_ts == [
        pd.Timestamp("2026-11-04 10:00:10"),
        pd.Timestamp("2026-11-04 10:00:12"),
        pd.Timestamp("2026-11-04 10:00:15"),
    ]
    assert ctx.total_late_events == 1


# --------------------------------------------------------------------------
# T07: Expired Late Arrival Isolation
# --------------------------------------------------------------------------
def test_07_expired_late_arrival_isolation(valid_flow_record):
    """
    Verifies that an expired flow (>10s late) is evaluated in single-event isolation
    and never added to the rolling buffer.
    """
    ctx = StreamingCausalContext(window_seconds=10, max_late_seconds=10.0)

    # Flow 1 at 10:00:30 (watermark = 10:00:30)
    rec1 = dict(valid_flow_record)
    rec1["Timestamp"] = "2026-11-04 10:00:30"
    ctx.process_flow(rec1)
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:30")

    # Flow 2 at 10:00:10 (20s late, strictly <= 30 - 10s = 20)
    rec2 = dict(valid_flow_record)
    rec2["Timestamp"] = "2026-11-04 10:00:10"
    rec2["Total Length of Fwd Packets"] = 300.0
    out2 = ctx.process_flow(rec2)

    # Evaluated in single-event isolation
    assert out2["Rolling_Flow_Velocity_10s"] == 1.0
    assert out2["Rolling_Bytes_Velocity_10s"] == 300.0
    # Buffer retains ONLY flow 1
    assert len(ctx.buffer) == 1
    assert ctx.buffer[0][0] == pd.Timestamp("2026-11-04 10:00:30")
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:30")
    assert ctx.total_late_events == 1


# --------------------------------------------------------------------------
# T08: Duplicate Timestamps Within a Chunk
# --------------------------------------------------------------------------
def test_08_duplicate_timestamps_in_chunk():
    """
    Verifies that 5 flows sharing the identical timestamp in a simultaneous batch
    all receive the complete aggregate velocity (5.0), matching offline bit-for-bit.
    """
    ts = pd.Timestamp("2026-11-04 10:00:00")
    df = pd.DataFrame({
        "Timestamp": [ts] * 5,
        "Total Fwd Packets": [1.0, 2.0, 1.0, 4.0, 2.0],
        "Total Length of Fwd Packets": [100.0, 200.0, 100.0, 400.0, 200.0],
        "Fwd IAT Total": [10.0] * 5,
    })

    df_off = add_temporal_velocity_features(df, window_seconds=10)

    ctx = StreamingCausalContext(window_seconds=10)
    df_stream = ctx.process_batch(df, causal_mode="simultaneous", flush=True)

    np.testing.assert_array_equal(
        df_off["Rolling_Flow_Velocity_10s"].values,
        df_stream["Rolling_Flow_Velocity_10s"].values,
    )
    np.testing.assert_array_equal(
        df_off["Rolling_Bytes_Velocity_10s"].values,
        df_stream["Rolling_Bytes_Velocity_10s"].values,
    )
    assert (df_stream["Rolling_Flow_Velocity_10s"] == 5.0).all()
    assert (df_stream["Rolling_Bytes_Velocity_10s"] == 1000.0).all()


# --------------------------------------------------------------------------
# T09: Duplicate Timestamps Across Chunks with flush=False
# --------------------------------------------------------------------------
def test_09_duplicate_timestamps_across_chunks():
    """
    Verifies that simultaneous flows split across chunk boundaries with flush=False
    maintain aggregation continuity and emit the identical aggregate (5.0).
    """
    ts = pd.Timestamp("2026-11-04 10:00:00")
    df = pd.DataFrame({
        "Timestamp": [ts] * 5,
        "Total Fwd Packets": [1.0, 2.0, 1.0, 4.0, 2.0],
        "Total Length of Fwd Packets": [100.0, 200.0, 100.0, 400.0, 200.0],
        "Fwd IAT Total": [10.0] * 5,
    })

    chunk1 = df.iloc[:3].copy()
    chunk2 = df.iloc[3:].copy()

    ctx = StreamingCausalContext(window_seconds=10)

    # Chunk 1: flush=False leaves group pending
    out1 = ctx.process_batch(chunk1, causal_mode="simultaneous", flush=False)
    assert len(out1) == 0
    assert len(ctx.pending_group) == 3

    # Chunk 2: flush=True finalizes all 5 flows together
    out2 = ctx.process_batch(chunk2, causal_mode="simultaneous", flush=True)
    assert len(out2) == 5
    assert (out2["Rolling_Flow_Velocity_10s"] == 5.0).all()
    assert (out2["Rolling_Bytes_Velocity_10s"] == 1000.0).all()


# --------------------------------------------------------------------------
# T10: Recovery After Rejected Future Spike
# --------------------------------------------------------------------------
def test_10_recovery_after_rejected_future_spike(valid_flow_record):
    """
    Verifies that after an injected year-2099 spike is quarantined, the context
    smoothly processes the subsequent legitimate flow with correct cumulative metrics.
    """
    ctx = StreamingCausalContext(
        window_seconds=10,
        max_timestamp_horizon=pd.Timestamp("2030-01-01 00:00:00"),
    )

    # 1. Clean flow at 10:00:00
    f1 = dict(valid_flow_record)
    f1["Timestamp"] = "2026-11-04 10:00:00"
    out1 = ctx.process_flow(f1)
    assert out1["Rolling_Flow_Velocity_10s"] == 1.0

    # 2. Rogue year-2099 spike
    f2 = dict(valid_flow_record)
    f2["Timestamp"] = "2099-01-01 00:00:00"
    out2 = ctx.process_flow(f2)
    assert out2["_context_state"] == "QUARANTINED"

    # 3. Clean flow at 10:00:01
    f3 = dict(valid_flow_record)
    f3["Timestamp"] = "2026-11-04 10:00:01"
    out3 = ctx.process_flow(f3)
    assert out3["_context_state"] == "HEALTHY"
    # Velocity accumulates correctly over f1 and f3
    assert out3["Rolling_Flow_Velocity_10s"] == 2.0
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:01")


# --------------------------------------------------------------------------
# T11: Strict Watermark Monotonicity
# --------------------------------------------------------------------------
def test_11_strict_watermark_monotonicity(valid_flow_record):
    """
    Verifies that the event-time watermark W is strictly monotonic non-decreasing:
    W_{k+1} >= W_k holds for every processed step across an adversarial stream.
    """
    ctx = StreamingCausalContext(
        window_seconds=10,
        max_timestamp_horizon=pd.Timestamp("2030-01-01 00:00:00"),
    )

    adversarial_timestamps = [
        "2026-11-04 10:00:05",
        "2026-11-04 10:00:02",  # Late in-window
        "2026-11-04 10:00:08",  # Advance
        "2099-01-01 00:00:00",  # Future spike
        "2026-11-04 10:00:08",  # Duplicate
        "2026-11-04 09:50:00",  # Expired late
        "not-a-timestamp",      # Corrupt string
        "2026-11-04 10:00:15",  # Advance
        "2026-11-04 10:00:10",  # Late in-window
    ]

    last_watermark = None
    for ts_str in adversarial_timestamps:
        rec = dict(valid_flow_record)
        rec["Timestamp"] = ts_str
        ctx.process_flow(rec)

        curr_wm = ctx.latest_timestamp
        if last_watermark is not None and curr_wm is not None:
            assert curr_wm >= last_watermark, f"Watermark regressed from {last_watermark} to {curr_wm} on {ts_str}"
        if curr_wm is not None:
            last_watermark = curr_wm

    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:15")


# --------------------------------------------------------------------------
# T12: Buffer Snapshot Immutability on Rejected Flows
# --------------------------------------------------------------------------
def test_12_buffer_snapshot_immutability(valid_flow_record):
    """
    Verifies that rejected/quarantined flows leave the internal buffer state
    bit-for-bit identical to a pre-attack snapshot.
    """
    ctx = StreamingCausalContext(window_seconds=10)

    # Ingest 3 valid flows
    for i in range(3):
        rec = dict(valid_flow_record)
        rec["Timestamp"] = f"2026-11-04 10:00:0{i}"
        ctx.process_flow(rec)

    snapshot_buf = list(ctx.buffer)
    snapshot_len = len(ctx.buffer)
    snapshot_wm = ctx.latest_timestamp

    # Attempt to ingest 5 malformed/anomalous flows
    malformed_flows = [
        {"Timestamp": None, "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 100.0},
        {"Timestamp": "2099-01-01 00:00:00", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 100.0},
        {"Timestamp": "2026-11-04 10:00:05", "Total Fwd Packets": -5.0, "Total Length of Fwd Packets": 100.0},
        {"Timestamp": "invalid-time", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 100.0},
        {"Timestamp": "2026-11-04 10:00:05", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": float("nan")},
    ]

    for flow in malformed_flows:
        ctx.process_flow(flow)

    assert len(ctx.buffer) == snapshot_len
    assert list(ctx.buffer) == snapshot_buf
    assert ctx.latest_timestamp == snapshot_wm


# --------------------------------------------------------------------------
# T13: Safe Capacity Overflow and Degraded Recovery Across Time Jumps
# --------------------------------------------------------------------------
def test_13_capacity_and_overflow_recovery(valid_flow_record):
    """
    Verifies that buffer capacity enforcement degrades safely under traffic bursts
    and clears degraded state strictly when t > t_last_overflow + 10s.
    """
    cap = 10
    ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=cap)

    # Send 15 flows at 10:00:00 -> exceeds capacity
    for i in range(15):
        rec = dict(valid_flow_record)
        rec["Timestamp"] = "2026-11-04 10:00:00"
        ctx.process_flow(rec)

    assert len(ctx.buffer) == cap
    assert ctx.is_degraded is True
    assert ctx.total_dropped_overflow_events == 5
    assert ctx.last_overflow_timestamp == pd.Timestamp("2026-11-04 10:00:00")

    # Flow at 10:00:05 (<= 10:00:00 + 10s): still degraded, triggers overflow drop and updates last_overflow_timestamp
    rec_mid = dict(valid_flow_record)
    rec_mid["Timestamp"] = "2026-11-04 10:00:05"
    out_mid = ctx.process_flow(rec_mid)
    assert out_mid["_context_state"] == "DEGRADED"
    assert ctx.is_degraded is True
    assert ctx.last_overflow_timestamp == pd.Timestamp("2026-11-04 10:00:05")

    # Flow at 10:00:15 (exact boundary: 10:00:05 + 10.000s): remains DEGRADED per event-time safety contract
    rec_bound = dict(valid_flow_record)
    rec_bound["Timestamp"] = "2026-11-04 10:00:15"
    out_bound = ctx.process_flow(rec_bound)
    assert out_bound["_context_state"] == "DEGRADED"
    assert ctx.is_degraded is True

    # Flow at 10:00:16 (> 10:00:05 + 10s): recovered to HEALTHY
    rec_post = dict(valid_flow_record)
    rec_post["Timestamp"] = "2026-11-04 10:00:16"
    out_post = ctx.process_flow(rec_post)
    assert out_post["_context_state"] == "HEALTHY"
    assert ctx.is_degraded is False


# --------------------------------------------------------------------------
# T14: Numerical Parity Between Single-Flow and Batch Processing
# --------------------------------------------------------------------------
def test_14_single_flow_and_batch_numerical_parity(valid_flow_record):
    """
    Verifies numerical parity between process_flow() and process_batch()
    across a stream with duplicates, in-window late arrivals, and quarantined spikes.
    """
    records = [
        {"Timestamp": "2026-11-04 10:00:00", "Total Fwd Packets": 2.0, "Total Length of Fwd Packets": 100.0, "Fwd IAT Total": 100.0},
        {"Timestamp": "2026-11-04 10:00:00", "Total Fwd Packets": 3.0, "Total Length of Fwd Packets": 200.0, "Fwd IAT Total": 200.0},
        {"Timestamp": "2099-01-01 00:00:00", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 50.0, "Fwd IAT Total": 50.0},  # Quarantined
        {"Timestamp": "2026-11-04 10:00:05", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 150.0, "Fwd IAT Total": 150.0},
        {"Timestamp": "2026-11-04 10:00:02", "Total Fwd Packets": 4.0, "Total Length of Fwd Packets": 300.0, "Fwd IAT Total": 300.0},  # In-window late
    ]

    # Batch processing
    ctx_batch = StreamingCausalContext(
        window_seconds=10,
        max_timestamp_horizon=pd.Timestamp("2030-01-01 00:00:00"),
    )
    df_raw = pd.DataFrame(records)
    batch_res = ctx_batch.process_batch(df_raw, causal_mode="sequential", flush=True)

    # Single-flow processing
    ctx_single = StreamingCausalContext(
        window_seconds=10,
        max_timestamp_horizon=pd.Timestamp("2030-01-01 00:00:00"),
    )
    single_res = []
    for r in records:
        out = ctx_single.process_flow(r)
        if out.get("_context_state") != "QUARANTINED":
            single_res.append(out)

    df_single = pd.DataFrame(single_res)

    assert len(batch_res) == len(df_single) == 4
    np.testing.assert_allclose(
        batch_res["Rolling_Flow_Velocity_10s"].values,
        df_single["Rolling_Flow_Velocity_10s"].values,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        batch_res["Rolling_Bytes_Velocity_10s"].values,
        df_single["Rolling_Bytes_Velocity_10s"].values,
        rtol=1e-5,
    )


# --------------------------------------------------------------------------
# T15: Warm-Up Rejection of Corrupted Timestamps & Horizon Exceedance
# --------------------------------------------------------------------------
def test_15_warm_up_rejects_corrupted_timestamps_and_horizon_exceedance():
    """
    Verifies that warm_up() safely discards NaT, unparseable strings, and
    timestamps exceeding max_timestamp_horizon without corrupting buffer or watermark.
    """
    ctx = StreamingCausalContext(
        window_seconds=10,
        max_timestamp_horizon=pd.Timestamp("2026-11-04 12:00:00"),
    )

    df_hist = pd.DataFrame({
        "Timestamp": [
            "2026-11-04 10:00:00",
            "not-a-valid-date",     # NaT
            None,                    # NaT
            "2026-11-04 10:00:05",  # Valid in-window
            "2099-01-01 00:00:00",  # Exceeds horizon
        ],
        "Total Fwd Packets": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Total Length of Fwd Packets": [100.0, 200.0, 300.0, 400.0, 500.0],
    })

    n_buffered = ctx.warm_up(df_hist)

    # Valid timestamps: 10:00:00 and 10:00:05 (within 10s of max valid ts 10:00:05)
    assert n_buffered == 2
    assert ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:05")
    assert len(ctx.buffer) == 2


# --------------------------------------------------------------------------
# T16: Stream Simulation Trusted Horizon Integration & Chunked Resilience
# --------------------------------------------------------------------------
def test_16_stream_simulation_trusted_horizon_chunked(valid_flow_record):
    """
    Verifies that prepare_chronological_stream attaches trusted_horizon metadata
    and that StreamingFeatureEngine quarantines rogue future timestamps across chunks.
    """
    rows = []
    for i in range(3):
        r = dict(valid_flow_record)
        r["Timestamp"] = f"2026-11-04 10:00:0{i}"
        rows.append(r)
    df = pd.DataFrame(rows)

    pool = prepare_chronological_stream(df)
    trusted_horizon = pool.attrs.get("trusted_horizon")

    assert trusted_horizon == pd.Timestamp("2026-11-04 10:00:12")

    qm = FlowQuarantineManager()
    engine = StreamingFeatureEngine(
        window_seconds=10,
        clock_mode="event_time",
        max_timestamp_horizon=trusted_horizon,
        quarantine_manager=qm,
    )

    # Process chunk 1: legitimate flows
    df_c1 = pool.iloc[:2].copy()
    features_c1 = engine.extract_features_df(df_c1, causal_mode="simultaneous", flush=True)
    assert len(features_c1) == 2
    assert qm.total_quarantined_records == 0

    # Injected rogue chunk with 2099 flow
    r_rogue = dict(valid_flow_record)
    r_rogue["Timestamp"] = "2099-01-01 00:00:00"
    rogue_chunk = pd.DataFrame([r_rogue])
    features_rogue = engine.extract_features_df(rogue_chunk, causal_mode="simultaneous", flush=True)
    # Rogue flow quarantined, 0 valid features emitted
    assert len(features_rogue) == 0
    assert qm.total_quarantined_records == 1
    assert qm.quarantine_reason_counts["TIMESTAMP_FUTURE_DRIFT"] == 1

    # Chunk 3: subsequent legitimate flow continues without disruption
    df_c3 = pool.iloc[2:].copy()
    features_c3 = engine.extract_features_df(df_c3, causal_mode="simultaneous", flush=True)
    assert len(features_c3) == 1
    assert engine.context.latest_timestamp == pd.Timestamp("2026-11-04 10:00:02")
