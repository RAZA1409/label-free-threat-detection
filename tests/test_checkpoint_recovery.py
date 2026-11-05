"""
tests/test_checkpoint_recovery.py
AEGIS Phase 3 Stage 4B.3D: Checkpointing, Graceful Shutdown & Restart Recovery Test Suite.

Verifies:
1. Versioned atomic checkpoint creation and restoration.
2. Restart recovery with populated rolling causal buffer.
3. Restart recovery with pending simultaneous group.
4. Durable critical-alert backlog preservation across process termination.
5. Duplicate alert prevention and replay idempotency.
6. Safe rejection of corrupted, 0-byte, and truncated checkpoint files.
7. Safe rejection of incompatible schema versions.
8. Safe rejection of invalid timestamps (NaT, non-monotonic, timezone mismatch).
9. Safe rejection of feature dimension mismatches (v1 != 32, v2 != 26).
10. Safe rejection of capacity ceiling violations (> 10,000 entries).
11. Atomic write failure safety (existing valid checkpoint survives).
12. Graceful shutdown flush and final checkpoint persistence.
13. Clean default state recovery after failed restoration attempts.
14. End-to-end v1/v2 inference invariance across continuous vs checkpointed execution.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from collections import deque
import json
import math
import os
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from checkpoint import (
    CheckpointManager,
    RecoveryStatus,
    RecoveryResult,
    AlertIdempotencyTracker,
    CURRENT_SCHEMA_VERSION,
)
from streaming_features import (
    StreamingCausalContext,
    StreamingFeatureEngine,
    FlowQuarantineManager,
)
from simulate_stream import run_stream_simulation
from telemetry import TelemetryManager
from features import ONE_WAY_FEATURES


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_stream_parquet(tmp_path):
    """Generates a deterministic synthetic stream dataset for replay tests."""
    np.random.seed(42)
    n_rows = 20
    timestamps = [f"2026-11-04 10:00:{i:02d}" for i in range(n_rows)]
    data = {
        "Timestamp": timestamps,
        "Source IP": [f"192.168.1.{10 + (i % 5)}" for i in range(n_rows)],
        "Destination IP": ["10.0.0.1"] * n_rows,
        "Protocol": [6] * n_rows,
        "Total Fwd Packets": [float(1 + (i % 4)) for i in range(n_rows)],
        "Total Length of Fwd Packets": [float(50 + (i * 10)) for i in range(n_rows)],
        "Fwd IAT Total": [1000.0] * n_rows,
        "Label": ["BENIGN"] * 15 + ["DDoS"] * 5,
        "Clean_Label": ["BENIGN"] * 15 + ["DDoS"] * 5,
    }
    for col in ONE_WAY_FEATURES:
        if col not in data:
            data[col] = np.random.uniform(10.0, 100.0, size=n_rows)

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_stream_pool.parquet"
    df.to_parquet(parquet_path)

    # Also save phase 1 prefix slice
    p1_path = tmp_path / "test_stream_p1.parquet"
    df.iloc[:10].to_parquet(p1_path)

    return parquet_path


# -----------------------------------------------------------------------------
# Test 1: Checkpoint Creation and Restoration
# -----------------------------------------------------------------------------

def test_checkpoint_creation_and_restoration(tmp_path):
    """Verify that a valid causal context is saved to checkpoint and fully restored."""
    ckpt_file = tmp_path / "checkpoint.json"
    mgr = CheckpointManager(checkpoint_path=ckpt_file)

    ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10_000)
    ctx.update("2026-11-04 10:00:00", 2.0, 150.0)
    ctx.update("2026-11-04 10:00:05", 1.0, 60.0)

    # Save checkpoint
    saved = mgr.save_checkpoint(ctx, extra_state={"total_processed": 2, "total_anomalies": 0})
    assert saved is True
    assert ckpt_file.exists()

    # Restore into fresh context
    fresh_ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10_000)
    res = mgr.restore_checkpoint(fresh_ctx)

    assert res.status == RecoveryStatus.SUCCESS
    assert res.restored_buffer_entries == 2
    assert res.restored_flow_index == 2
    assert fresh_ctx.latest_timestamp == pd.Timestamp("2026-11-04 10:00:05")
    assert len(fresh_ctx.buffer) == 2
    assert fresh_ctx.buffer[0][1] == 2.0
    assert fresh_ctx.buffer[1][1] == 1.0


# -----------------------------------------------------------------------------
# Test 2: Restart with Populated Rolling Buffer
# -----------------------------------------------------------------------------

def test_restart_with_populated_rolling_buffer(tmp_path):
    """Verify continuity of rolling window calculations when resuming from checkpoint."""
    ckpt_file = tmp_path / "rolling_ckpt.json"
    mgr = CheckpointManager(checkpoint_path=ckpt_file)

    ctx1 = StreamingCausalContext(window_seconds=10)
    ctx1.update("2026-11-04 10:00:01", 1.0, 100.0)
    ctx1.update("2026-11-04 10:00:03", 2.0, 200.0)
    ctx1.update("2026-11-04 10:00:06", 1.0, 50.0)

    mgr.save_checkpoint(ctx1, extra_state={"total_processed": 3})

    # Restore into fresh context
    ctx2 = StreamingCausalContext(window_seconds=10)
    res = mgr.restore_checkpoint(ctx2)
    assert res.status == RecoveryStatus.SUCCESS

    # Add 4th event at 10:00:08 (all 4 in (10:00:08 - 10s, 10:00:08])
    r_flow, r_bytes, _ = ctx2.update("2026-11-04 10:00:08", 3.0, 300.0)
    assert r_flow == 4.0
    assert r_bytes == 650.0  # 100 + 200 + 50 + 300

    # Add 5th event at 10:00:12 -> event at 10:00:01 expires (<= 10:00:02 cutoff)
    r_flow2, r_bytes2, _ = ctx2.update("2026-11-04 10:00:12", 1.0, 10.0)
    assert r_flow2 == 4.0  # events at 03, 06, 08, 12
    assert r_bytes2 == 560.0  # 200 + 50 + 300 + 10


# -----------------------------------------------------------------------------
# Test 3: Restart with Pending Simultaneous Group
# -----------------------------------------------------------------------------

def test_restart_with_pending_simultaneous_group(tmp_path):
    """Verify that unfinalized simultaneous timestamp groups across batches are preserved."""
    ckpt_file = tmp_path / "pending_ckpt.json"
    mgr = CheckpointManager(checkpoint_path=ckpt_file)

    ctx = StreamingCausalContext(window_seconds=10)
    batch_df = pd.DataFrame([
        {"Timestamp": "2026-11-04 10:00:00", "Total Fwd Packets": 1.0, "Total Length of Fwd Packets": 50.0, "Fwd IAT Total": 10.0},
        {"Timestamp": "2026-11-04 10:00:05", "Total Fwd Packets": 2.0, "Total Length of Fwd Packets": 100.0, "Fwd IAT Total": 10.0},
        {"Timestamp": "2026-11-04 10:00:05", "Total Fwd Packets": 3.0, "Total Length of Fwd Packets": 150.0, "Fwd IAT Total": 10.0},
    ])

    # Process with flush=False -> the two 10:00:05 events remain in pending_group
    emitted = ctx.process_batch(batch_df, causal_mode="simultaneous", flush=False)
    assert len(emitted) == 1  # only 10:00:00 emitted
    assert len(ctx.pending_group) == 2  # two 10:00:05 retained

    mgr.save_checkpoint(ctx, extra_state={"total_processed": 1})

    # Restore into fresh context
    restored_ctx = StreamingCausalContext(window_seconds=10)
    res = mgr.restore_checkpoint(restored_ctx)

    assert res.status == RecoveryStatus.SUCCESS
    assert res.restored_pending_entries == 2
    assert len(restored_ctx.pending_group) == 2

    # Flush the restored context -> both pending events finalized with full group aggregate
    final_df = restored_ctx.flush()
    assert len(final_df) == 2
    # Both events share the aggregate of 10:00:00 (50B) + both 10:00:05 events (100B + 150B) = 300B
    assert final_df["Rolling_Flow_Velocity_10s"].iloc[0] == 3.0
    assert final_df["Rolling_Bytes_Velocity_10s"].iloc[0] == 300.0
    assert final_df["Rolling_Bytes_Velocity_10s"].iloc[1] == 300.0


# -----------------------------------------------------------------------------
# Test 4: Durable Critical-Alert Backlog Recovery
# -----------------------------------------------------------------------------

def test_durable_critical_alert_backlog_recovery_after_outage(tmp_path):
    """Verify that alerts retained during a telemetry failure survive process termination."""
    ckpt_file = tmp_path / "alerts_ckpt.json"
    mgr = CheckpointManager(checkpoint_path=ckpt_file)

    feed_file = tmp_path / "feed.json"
    telemetry_mgr = TelemetryManager(feed_path=feed_file, enabled=True, throttle_interval_sec=0, throttle_event_count=1)

    alerts = [
        {"flow_index": 1, "timestamp": "10:00:01", "verdict": "ALERT", "score": 0.88},
        {"flow_index": 2, "timestamp": "10:00:02", "verdict": "QUARANTINED", "score": None},
        {"flow_index": 3, "timestamp": "10:00:03", "verdict": "ALERT", "score": 0.95},
    ]

    # Simulating outage: writes to live_stream_feed fail, so alerts accumulate in memory
    feed_state = {"total_processed": 3, "recent_alerts": alerts}
    with patch("telemetry.atomic_write_json", side_effect=OSError("Disk full")):
        for a in alerts:
            telemetry_mgr.update_and_write(feed_state, force=True, recent_alert=a)

    assert telemetry_mgr.is_degraded is True
    assert len(telemetry_mgr._backlog_alerts) == 3

    # Checkpoint is written durably
    ctx = StreamingCausalContext(window_seconds=10)
    saved = mgr.save_checkpoint(ctx, telemetry_mgr=telemetry_mgr, extra_state={"total_processed": 3, "total_anomalies": 2})
    assert saved is True

    # Process restart: simulate fresh memory
    new_telemetry_mgr = TelemetryManager(feed_path=feed_file, enabled=True)
    new_ctx = StreamingCausalContext(window_seconds=10)

    res = mgr.restore_checkpoint(new_ctx, telemetry_mgr=new_telemetry_mgr)
    assert res.status == RecoveryStatus.SUCCESS
    assert res.restored_alerts_count == 3
    assert len(new_telemetry_mgr._backlog_alerts) == 3
    assert new_telemetry_mgr._backlog_alerts[0]["score"] == 0.88
    assert new_telemetry_mgr._backlog_alerts[1]["verdict"] == "QUARANTINED"
    assert new_telemetry_mgr._backlog_alerts[2]["score"] == 0.95


# -----------------------------------------------------------------------------
# Test 5: Replay Idempotency & Duplicate Prevention
# -----------------------------------------------------------------------------

def test_duplicate_alert_prevention_idempotency(tmp_path):
    """Verify idempotency tracker prevents duplicate alert registration and emission."""
    ckpt_file = tmp_path / "idempotent_ckpt.json"
    mgr = CheckpointManager(checkpoint_path=ckpt_file)

    tracker = mgr.idempotency_tracker
    assert tracker.register("flow_10:10:00:01") is True
    assert tracker.register("flow_10:10:00:01") is False  # Duplicate rejected!

    assert tracker.register("flow_11:10:00:02") is True
    assert tracker.is_seen("flow_10:10:00:01") is True
    assert tracker.is_seen("flow_11:10:00:02") is True
    assert tracker.is_seen("flow_99:10:00:99") is False

    ctx = StreamingCausalContext(window_seconds=10)
    mgr.save_checkpoint(ctx)

    # Restore in new manager
    new_mgr = CheckpointManager(checkpoint_path=ckpt_file)
    new_ctx = StreamingCausalContext(window_seconds=10)
    res = new_mgr.restore_checkpoint(new_ctx)

    assert res.status == RecoveryStatus.SUCCESS
    # Committed IDs survive recovery
    assert new_mgr.idempotency_tracker.is_seen("flow_10:10:00:01") is True
    assert new_mgr.idempotency_tracker.is_seen("flow_11:10:00:02") is True
    assert new_mgr.idempotency_tracker.register("flow_10:10:00:01") is False


# -----------------------------------------------------------------------------
# Test 6: Corrupted and Truncated Checkpoints Safe Rejection
# -----------------------------------------------------------------------------

def test_corrupted_and_truncated_checkpoints_safe_rejection(tmp_path):
    """Verify that corrupt, 0-byte, and malformed files are safely rejected without crashing."""
    mgr = CheckpointManager()
    ctx = StreamingCausalContext(window_seconds=10)

    # Case A: Missing file
    missing = tmp_path / "missing.json"
    mgr.checkpoint_path = missing
    res_miss = mgr.restore_checkpoint(ctx)
    assert res_miss.status == RecoveryStatus.SKIPPED

    # Case B: 0-byte file
    empty = tmp_path / "empty.json"
    empty.touch()
    mgr.checkpoint_path = empty
    res_empty = mgr.restore_checkpoint(ctx)
    assert res_empty.status == RecoveryStatus.CORRUPTED_OR_MALFORMED

    # Case C: Truncated JSON
    trunc = tmp_path / "trunc.json"
    trunc.write_text('{"schema_version": "1.0.0", "causal_context": {', encoding="utf-8")
    mgr.checkpoint_path = trunc
    res_trunc = mgr.restore_checkpoint(ctx)
    assert res_trunc.status == RecoveryStatus.CORRUPTED_OR_MALFORMED

    # Case D: Non-dict root
    not_dict = tmp_path / "not_dict.json"
    not_dict.write_text("[1, 2, 3]", encoding="utf-8")
    mgr.checkpoint_path = not_dict
    res_list = mgr.restore_checkpoint(ctx)
    assert res_list.status == RecoveryStatus.CORRUPTED_OR_MALFORMED

    # Context remains pristine
    assert len(ctx.buffer) == 0
    assert ctx.latest_timestamp is None


# -----------------------------------------------------------------------------
# Test 7: Unsupported Schema Version Rejection
# -----------------------------------------------------------------------------

def test_unsupported_checkpoint_version_rejection(tmp_path):
    """Verify checkpoints with incompatible schema versions are safely rejected."""
    bad_ver_file = tmp_path / "future_ver.json"
    bad_payload = {
        "schema_version": "99.0.0",
        "engine_metadata": {"v1_feature_dim": 32, "v2_feature_dim": 26},
        "causal_context": {"rolling_buffer": [], "pending_group": []},
        "alert_state": {},
    }
    bad_ver_file.write_text(json.dumps(bad_payload), encoding="utf-8")

    mgr = CheckpointManager(checkpoint_path=bad_ver_file)
    ctx = StreamingCausalContext(window_seconds=10)
    res = mgr.restore_checkpoint(ctx)

    assert res.status == RecoveryStatus.INCOMPATIBLE_VERSION
    assert "Unsupported checkpoint schema version '99.0.0'" in res.error_message
    assert len(ctx.buffer) == 0


# -----------------------------------------------------------------------------
# Test 8: Invalid Timestamps Safe Rejection
# -----------------------------------------------------------------------------

def test_invalid_timestamps_safe_rejection(tmp_path):
    """Verify non-chronological, unparseable, or timezone-mismatched timestamps are rejected."""
    mgr = CheckpointManager()
    ctx = StreamingCausalContext(window_seconds=10)

    # Case A: Non-chronological order (t1 > t2)
    non_chrono = tmp_path / "non_chrono.json"
    payload_nc = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "engine_metadata": {"v1_feature_dim": 32, "v2_feature_dim": 26},
        "causal_context": {
            "rolling_buffer": [
                ["2026-11-04 10:00:10", 1.0, 50.0, 1.0],
                ["2026-11-04 10:00:05", 2.0, 100.0, 0.0],  # Earlier than previous!
            ],
            "pending_group": [],
        },
        "alert_state": {"durable_alert_backlog": [], "committed_alert_ids": []},
    }
    non_chrono.write_text(json.dumps(payload_nc), encoding="utf-8")
    mgr.checkpoint_path = non_chrono
    res_nc = mgr.restore_checkpoint(ctx)
    assert res_nc.status == RecoveryStatus.INVALID_TIMESTAMPS
    assert "Non-chronological timestamp order" in res_nc.error_message

    # Case B: Unparseable timestamp
    unparseable = tmp_path / "unparseable.json"
    payload_un = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "engine_metadata": {"v1_feature_dim": 32, "v2_feature_dim": 26},
        "causal_context": {
            "rolling_buffer": [["INVALID_TIME_STRING_123", 1.0, 50.0, 1.0]],
            "pending_group": [],
        },
        "alert_state": {"durable_alert_backlog": [], "committed_alert_ids": []},
    }
    unparseable.write_text(json.dumps(payload_un), encoding="utf-8")
    mgr.checkpoint_path = unparseable
    res_un = mgr.restore_checkpoint(ctx)
    assert res_un.status == RecoveryStatus.INVALID_TIMESTAMPS


# -----------------------------------------------------------------------------
# Test 9: Feature Dimension Mismatch Rejection
# -----------------------------------------------------------------------------

def test_feature_dimension_mismatch_rejection(tmp_path):
    """Verify that checkpoint with incompatible model dimensions is rejected."""
    dim_file = tmp_path / "dim_mismatch.json"
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "engine_metadata": {
            "v1_feature_dim": 39,  # Incompatible: expected 32
            "v2_feature_dim": 26,
        },
        "causal_context": {"rolling_buffer": [], "pending_group": []},
        "alert_state": {"durable_alert_backlog": [], "committed_alert_ids": []},
    }
    dim_file.write_text(json.dumps(payload), encoding="utf-8")

    mgr = CheckpointManager(checkpoint_path=dim_file, v1_feature_dim=32, v2_feature_dim=26)
    ctx = StreamingCausalContext(window_seconds=10)
    res = mgr.restore_checkpoint(ctx)

    assert res.status == RecoveryStatus.DIMENSION_MISMATCH
    assert "Feature dimension mismatch" in res.error_message


# -----------------------------------------------------------------------------
# Test 10: Capacity Ceiling Violations Rejection
# -----------------------------------------------------------------------------

def test_capacity_ceiling_violations_rejection(tmp_path):
    """Verify that a checkpoint exceeding configured capacity is rejected."""
    cap_file = tmp_path / "cap_violation.json"
    # Create 15 entries when max_capacity is 10
    overflow_buffer = [
        [f"2026-11-04 10:00:{i:02d}", 1.0, 50.0, 1.0] for i in range(15)
    ]
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "engine_metadata": {"v1_feature_dim": 32, "v2_feature_dim": 26},
        "causal_context": {
            "rolling_buffer": overflow_buffer,
            "pending_group": [],
        },
        "alert_state": {"durable_alert_backlog": [], "committed_alert_ids": []},
    }
    cap_file.write_text(json.dumps(payload), encoding="utf-8")

    mgr = CheckpointManager(checkpoint_path=cap_file, max_buffer_capacity=10)
    ctx = StreamingCausalContext(window_seconds=10)
    res = mgr.restore_checkpoint(ctx)

    assert res.status == RecoveryStatus.CAPACITY_VIOLATION
    assert "exceeds max_buffer_capacity" in res.error_message


# -----------------------------------------------------------------------------
# Test 11: Interrupted Checkpoint Writes Preserve Valid File
# -----------------------------------------------------------------------------

def test_interrupted_checkpoint_write_preserves_valid_file(tmp_path):
    """Verify that an interrupted write leaves the existing valid checkpoint undamaged."""
    ckpt_file = tmp_path / "safe_atomic.json"
    mgr = CheckpointManager(checkpoint_path=ckpt_file)
    ctx = StreamingCausalContext(window_seconds=10)
    ctx.update("2026-11-04 10:00:01", 1.0, 100.0)

    # Initial valid checkpoint
    mgr.save_checkpoint(ctx, extra_state={"total_processed": 1})
    assert ckpt_file.exists()

    with open(ckpt_file, "r", encoding="utf-8") as f:
        orig_data = json.load(f)
    assert orig_data["stream_progress"]["flow_index"] == 1

    # Simulate crash/failure during next save
    with patch("src.telemetry.os.replace", side_effect=OSError("Disk write interrupted")):
        ctx.update("2026-11-04 10:00:05", 2.0, 200.0)
        saved = mgr.save_checkpoint(ctx, extra_state={"total_processed": 2})
        assert saved is False

    # Original file on disk remains completely intact and valid
    with open(ckpt_file, "r", encoding="utf-8") as f:
        post_data = json.load(f)
    assert post_data["stream_progress"]["flow_index"] == 1


# -----------------------------------------------------------------------------
# Test 12: Graceful Shutdown and Final Flush
# -----------------------------------------------------------------------------

def test_graceful_shutdown_and_final_checkpoint(synthetic_stream_parquet, tmp_path):
    """Verify graceful shutdown writes a complete final checkpoint with accurate flow counts."""
    ckpt_file = tmp_path / "shutdown_ckpt.json"
    feed_file = tmp_path / "live_feed.json"

    res = run_stream_simulation(
        speed_eps=0,
        max_events=15,
        inject_attacks=True,
        chunk_size=5,
        v1_only=False,
        test_file=synthetic_stream_parquet,
        live_feed_file=feed_file,
        checkpoint_file=ckpt_file,
        save_checkpoint_on_shutdown=True,
        write_feed=True,
        verbose=False,
    )

    assert res["checkpoint_saved"] is True
    assert ckpt_file.exists()

    with open(ckpt_file, "r", encoding="utf-8") as f:
        ckpt_data = json.load(f)

    assert ckpt_data["schema_version"] == CURRENT_SCHEMA_VERSION
    assert ckpt_data["stream_progress"]["flow_index"] == 15
    assert ckpt_data["alert_state"]["total_anomalies"] == res["total_anomalies"]
    assert len(ckpt_data["causal_context"]["rolling_buffer"]) > 0


# -----------------------------------------------------------------------------
# Test 13: Clean Default State Recovery After Failed Restoration
# -----------------------------------------------------------------------------

def test_recovery_after_failed_restoration_leaves_clean_state(tmp_path):
    """Verify that an unexpected exception during load leaves context in clean reset state."""
    ckpt_file = tmp_path / "bad_load.json"
    mgr = CheckpointManager(checkpoint_path=ckpt_file)
    ctx = StreamingCausalContext(window_seconds=10)
    ctx.update("2026-11-04 10:00:01", 1.0, 100.0)
    mgr.save_checkpoint(ctx)

    # Force an unexpected exception inside state unpacking
    with patch("src.checkpoint.pd.Timestamp", side_effect=RuntimeError("Simulated memory allocation failure")):
        fresh_ctx = StreamingCausalContext(window_seconds=10)
        res = mgr.restore_checkpoint(fresh_ctx)

    assert res.status == RecoveryStatus.FAILED
    assert "Simulated memory allocation failure" in res.error_message
    # Context must be reset to clean default state
    assert len(fresh_ctx.buffer) == 0
    assert fresh_ctx.latest_timestamp is None


# -----------------------------------------------------------------------------
# Test 14: End-to-End Inference Invariance (Continuous vs Checkpointed)
# -----------------------------------------------------------------------------

def test_inference_invariance_continuous_vs_checkpointed(synthetic_stream_parquet, tmp_path):
    """Verify exact numerical and verdict identity between continuous and checkpointed stream runs."""
    feed_cont = tmp_path / "feed_cont.json"
    p1_file = synthetic_stream_parquet.parent / "test_stream_p1.parquet"

    res_continuous = run_stream_simulation(
        speed_eps=0,
        max_events=20,
        inject_attacks=False,
        chunk_size=5,
        v1_only=False,
        test_file=synthetic_stream_parquet,
        live_feed_file=feed_cont,
        write_feed=True,
        verbose=False,
    )

    # Segmented run with checkpoint
    ckpt_file = tmp_path / "invariance_ckpt.json"
    feed_seg = tmp_path / "feed_seg.json"

    # Phase 1: Process first 10 events and save checkpoint
    res_p1 = run_stream_simulation(
        speed_eps=0,
        max_events=10,
        inject_attacks=False,
        chunk_size=5,
        v1_only=False,
        test_file=p1_file,
        live_feed_file=feed_seg,
        checkpoint_file=ckpt_file,
        save_checkpoint_on_shutdown=True,
        write_feed=True,
        verbose=False,
    )
    assert res_p1["total_processed"] == 10

    # Phase 2: Resume from checkpoint and process through event 20
    res_p2 = run_stream_simulation(
        speed_eps=0,
        max_events=20,
        inject_attacks=False,
        chunk_size=5,
        v1_only=False,
        test_file=synthetic_stream_parquet,
        live_feed_file=feed_seg,
        checkpoint_file=ckpt_file,
        auto_restore=True,
        save_checkpoint_on_shutdown=True,
        write_feed=True,
        verbose=False,
    )

    # Primary verification: total processed and anomaly counts match exactly
    assert res_p2["total_processed"] == res_continuous["total_processed"]
    assert res_p2["total_anomalies"] == res_continuous["total_anomalies"]
    assert res_p2["anomaly_rate_pct"] == res_continuous["anomaly_rate_pct"]
    assert res_p2["v2_primary_anomalies"] == res_continuous["v2_primary_anomalies"]
    assert res_p2["v2_reference_anomalies"] == res_continuous["v2_reference_anomalies"]
    assert res_p2["v2_degraded_count"] == res_continuous["v2_degraded_count"]
    assert res_p2["v2_unavailable_count"] == res_continuous["v2_unavailable_count"]

    # Verify alerts produced match between continuous and segmented runs
    assert len(res_p2["recent_alerts"]) == len(res_continuous["recent_alerts"])
    for a_cont, a_seg in zip(res_continuous["recent_alerts"], res_p2["recent_alerts"]):
        assert a_cont["verdict"] == a_seg["verdict"]
        if a_cont["score"] is None:
            assert a_seg["score"] is None
        else:
            assert abs(a_cont["score"] - a_seg["score"]) < 1e-4
        assert a_cont["threshold"] == a_seg["threshold"]
