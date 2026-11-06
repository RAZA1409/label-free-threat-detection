"""
AEGIS Streaming Stress & Performance Test Suite (Phase 3 Stage 4B.3E)

Verifies high-volume throughput, latency profiles, burst handling,
capacity ceiling enforcement, degraded recovery, telemetry failure resilience,
checkpoint resumption continuity, and memory boundedness.
"""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import psutil
import pytest

from checkpoint import (
    CheckpointManager,
    RecoveryStatus,
    RecoveryResult,
    AlertIdempotencyTracker,
)
from streaming_features import (
    StreamingCausalContext,
    StreamingFeatureEngine,
    FlowQuarantineManager,
)
from simulate_stream import run_stream_simulation
from telemetry import TelemetryManager, atomic_write_json
from features import ONE_WAY_FEATURES


def get_current_rss_mb() -> float:
    """Returns the current process resident set size (RSS) in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def generate_synthetic_flow_df(
    n_flows: int,
    base_time: str = "2026-11-04 10:00:00",
    interval_ms: float = 1.0,
    burst_mode: bool = False,
    attack_ratio: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a deterministic synthetic DataFrame of network flows."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(base_time)

    if burst_mode:
        timestamps = [start_ts] * n_flows
    else:
        timestamps = [start_ts + pd.Timedelta(milliseconds=i * interval_ms) for i in range(n_flows)]

    n_attacks = int(n_flows * attack_ratio)
    labels = ["BENIGN"] * (n_flows - n_attacks) + ["DDoS"] * n_attacks
    rng.shuffle(labels)

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Source IP": [f"192.168.1.{(i % 250) + 1}" for i in range(n_flows)],
        "Destination IP": ["10.0.0.1"] * n_flows,
        "Source Port": rng.integers(1024, 65535, size=n_flows).tolist(),
        "Destination Port": [80 if lbl == "BENIGN" else 443 for lbl in labels],
        "Protocol": [6] * n_flows,
        "Total Fwd Packets": rng.uniform(1.0, 10.0, size=n_flows).tolist(),
        "Total Length of Fwd Packets": rng.uniform(50.0, 1000.0, size=n_flows).tolist(),
        "Fwd IAT Total": rng.uniform(10.0, 2000.0, size=n_flows).tolist(),
        "Label": labels,
        "Clean_Label": labels,
    })

    for col in ONE_WAY_FEATURES:
        if col not in df.columns:
            df[col] = rng.uniform(10.0, 100.0, size=n_flows)

    return df


# -----------------------------------------------------------------------------
# Test 1: High-Volume Feature Engine Throughput and Latencies (10,000 flows)
# -----------------------------------------------------------------------------

def test_high_volume_feature_engine_throughput_and_latencies():
    """Verify that pure causal feature engine processes 10,000 flows with high throughput."""
    n_flows = 10_000
    df = generate_synthetic_flow_df(n_flows, interval_ms=1.0)
    ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10_000)

    t0 = time.perf_counter()
    res = ctx.process_batch(df, causal_mode="simultaneous", flush=True)
    t_total = time.perf_counter() - t0

    throughput = n_flows / t_total if t_total > 0 else 0.0

    # Assertions
    assert len(res) == n_flows
    assert throughput > 500.0, f"Expected throughput > 500 fps, got {throughput:.1f} fps"
    assert "Rolling_Flow_Velocity_10s" in res.columns
    assert "Rolling_Bytes_Velocity_10s" in res.columns
    assert "Rolling_Single_Pkt_Ratio_10s" in res.columns
    assert not res["Rolling_Flow_Velocity_10s"].isna().any()
    assert not res["Rolling_Bytes_Velocity_10s"].isna().any()
    assert not res["Rolling_Single_Pkt_Ratio_10s"].isna().any()
    assert ctx.total_processed == n_flows
    assert ctx.is_degraded is False


# -----------------------------------------------------------------------------
# Test 2: Burst Traffic Causal Window and Monotonic Ordering
# -----------------------------------------------------------------------------

def test_burst_traffic_causal_window_and_ordering():
    """Verify that a 1,000-flow burst at identical timestamp preserves order and aggregates correctly."""
    n_flows = 1_000
    df = generate_synthetic_flow_df(n_flows, burst_mode=True)
    ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10_000)

    # Ingest in chunks of 250 with flush=False
    chunk_size = 250
    n_chunks = n_flows // chunk_size
    emitted_dfs = []

    for i in range(n_chunks):
        sub_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        is_last = (i == n_chunks - 1)
        res_chunk = ctx.process_batch(sub_df, causal_mode="simultaneous", flush=is_last)
        if len(res_chunk) > 0:
            emitted_dfs.append(res_chunk)

    final_df = pd.concat(emitted_dfs, ignore_index=True)

    assert len(final_df) == n_flows
    # All 1,000 flows share timestamp t, so each must receive cumulative aggregate of 1,000
    assert (final_df["Rolling_Flow_Velocity_10s"] == float(n_flows)).all()
    # Ordering must match original input order exactly
    assert (final_df["Source IP"] == df["Source IP"]).all()
    assert ctx.total_processed == n_flows


# -----------------------------------------------------------------------------
# Test 3: Buffer Capacity Ceiling and Degraded Recovery Under Burst
# -----------------------------------------------------------------------------

def test_buffer_capacity_ceiling_and_degraded_recovery_under_burst():
    """Verify buffer cap at 10,000, degraded transition, and automatic recovery."""
    capacity = 10_000
    burst_size = 12_000
    df_burst = generate_synthetic_flow_df(burst_size, base_time="2026-11-04 10:00:00", interval_ms=0.5)
    ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=capacity)

    res_burst = ctx.process_batch(df_burst, causal_mode="simultaneous", flush=False)

    # 1. Capacity must strictly not exceed 10,000
    total_retained = len(ctx.buffer) + len(ctx.pending_group)
    assert total_retained <= capacity
    assert ctx.is_degraded is True
    assert ctx.total_dropped_overflow_events == (burst_size - capacity)
    assert ctx.last_overflow_timestamp is not None

    # 2. Advance time past recovery horizon (> last_overflow + 10s window) with normal rate
    recovery_flows = 2_000
    df_recovery = generate_synthetic_flow_df(
        recovery_flows,
        base_time="2026-11-04 10:00:18",  # 18s later (> 10s window)
        interval_ms=5.0,
    )
    _ = ctx.process_batch(df_recovery, causal_mode="simultaneous", flush=True)

    assert ctx.is_degraded is False
    assert len(ctx.buffer) <= capacity


# -----------------------------------------------------------------------------
# Test 4: Burst Traffic Pending Simultaneous Group Retention
# -----------------------------------------------------------------------------

def test_burst_traffic_pending_simultaneous_retention():
    """Verify that unfinalized simultaneous flows hold across chunk boundaries."""
    ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10_000)

    # Chunk 1: 100 flows at t0, then 50 flows at t1 (t1 > t0)
    df1_a = generate_synthetic_flow_df(100, base_time="2026-11-04 10:00:00", burst_mode=True)
    df1_b = generate_synthetic_flow_df(50, base_time="2026-11-04 10:00:02", burst_mode=True)
    df1 = pd.concat([df1_a, df1_b], ignore_index=True)

    res1 = ctx.process_batch(df1, causal_mode="simultaneous", flush=False)
    # The 100 flows at t0 must be emitted; the 50 flows at t1 must be retained in pending_group
    assert len(res1) == 100
    assert len(ctx.pending_group) == 50

    # Chunk 2: 30 flows at t2 > t1
    df2 = generate_synthetic_flow_df(30, base_time="2026-11-04 10:00:05", burst_mode=True)
    res2 = ctx.process_batch(df2, causal_mode="simultaneous", flush=True)

    # res2 must emit the 50 pending flows from Chunk 1 + 30 flows from Chunk 2
    assert len(res2) == 80
    assert len(ctx.pending_group) == 0
    assert ctx.total_processed == 180


# -----------------------------------------------------------------------------
# Test 5: Telemetry Persistence Failure Under Sustained Load
# -----------------------------------------------------------------------------

def test_telemetry_persistence_failure_under_sustained_load(tmp_path):
    """Verify that telemetry write failures do not crash inference and preserve critical alerts."""
    feed_path = tmp_path / "live_feed_stress.json"
    mgr = TelemetryManager(feed_path=feed_path, enabled=True)

    # 1. Initial write
    mgr.record_event(is_critical=False)
    assert mgr.update_and_write({"initial": True}, force=False) is True
    assert mgr.is_degraded is False

    # 2. Simulate 100 flows during persistence failure with 50 critical alerts
    with patch("telemetry.atomic_write_json", side_effect=PermissionError("Disk simulated write lock")):
        for i in range(100):
            is_critical = (i % 2 == 0)
            mgr.record_event(is_critical=is_critical)
            alert = {"flow_index": i, "score": 0.95, "verdict": "ANOMALOUS"} if is_critical else None
            # Must not throw exception
            ok = mgr.update_and_write({"flow": i}, force=is_critical, recent_alert=alert)
            assert ok is False if is_critical else True

    health_outage = mgr.get_health()
    assert health_outage["status"] == "DEGRADED"
    assert health_outage["backlog_alerts_retained"] == 50

    # 3. Restore write capability and flush
    success_flush = mgr.flush({"final": True, "status": "RECOVERED"})
    assert success_flush is True
    health_recovered = mgr.get_health()
    assert health_recovered["status"] == "HEALTHY"
    assert health_recovered["consecutive_failures"] == 0


# -----------------------------------------------------------------------------
# Test 6: Checkpoint and Restart Continuity Under Load
# -----------------------------------------------------------------------------

def test_checkpoint_and_restart_continuity_under_load(tmp_path):
    """Verify that saving and restoring a checkpoint midway produces identical results to continuous run."""
    n_flows = 40
    df = generate_synthetic_flow_df(n_flows, interval_ms=10.0, attack_ratio=0.25)
    p_full = tmp_path / "stream_full.parquet"
    df.to_parquet(p_full)

    p_p1 = tmp_path / "stream_p1.parquet"
    df.iloc[:20].to_parquet(p_p1)

    ckpt_file = tmp_path / "stress_checkpoint.json"
    feed_cont = tmp_path / "feed_cont.json"
    feed_seg = tmp_path / "feed_seg.json"

    # Continuous reference run
    res_cont = run_stream_simulation(
        speed_eps=0,
        max_events=40,
        inject_attacks=False,
        chunk_size=10,
        v1_only=False,
        test_file=p_full,
        live_feed_file=feed_cont,
        write_feed=True,
        verbose=False,
    )

    # Segmented Phase 1: 20 flows
    res_p1 = run_stream_simulation(
        speed_eps=0,
        max_events=20,
        inject_attacks=False,
        chunk_size=10,
        v1_only=False,
        test_file=p_p1,
        live_feed_file=feed_seg,
        checkpoint_file=ckpt_file,
        save_checkpoint_on_shutdown=True,
        write_feed=True,
        verbose=False,
    )
    assert res_p1["total_processed"] == 20

    # Segmented Phase 2: Resume from checkpoint through flow 40
    res_p2 = run_stream_simulation(
        speed_eps=0,
        max_events=40,
        inject_attacks=False,
        chunk_size=10,
        v1_only=False,
        test_file=p_full,
        live_feed_file=feed_seg,
        checkpoint_file=ckpt_file,
        auto_restore=True,
        save_checkpoint_on_shutdown=True,
        write_feed=True,
        verbose=False,
    )

    # Assertions
    assert res_p2["total_processed"] == res_cont["total_processed"] == 40
    assert res_p2["total_anomalies"] == res_cont["total_anomalies"]
    assert res_p2["v2_primary_anomalies"] == res_cont["v2_primary_anomalies"]
    assert res_p2["v2_reference_anomalies"] == res_cont["v2_reference_anomalies"]
    assert len(res_p2["recent_alerts"]) == len(res_cont["recent_alerts"])


# -----------------------------------------------------------------------------
# Test 7: Memory Boundedness and No Leak Under Sustained Stream
# -----------------------------------------------------------------------------

def test_memory_boundedness_and_no_leak_under_sustained_stream():
    """Verify that buffer size stays bounded and memory does not grow unbounded across 15,000 flows."""
    n_flows = 15_000
    df = generate_synthetic_flow_df(n_flows, interval_ms=10.0)
    ctx = StreamingCausalContext(window_seconds=10, max_buffer_capacity=10_000)

    rss_start = get_current_rss_mb()
    chunk_size = 3_000
    n_chunks = n_flows // chunk_size

    for i in range(n_chunks):
        sub_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        is_last = (i == n_chunks - 1)
        _ = ctx.process_batch(sub_df, causal_mode="simultaneous", flush=is_last)
        # Buffer must stay strictly bounded by the 10s sliding window
        assert len(ctx.buffer) <= 10_000

    rss_end = get_current_rss_mb()
    rss_growth = rss_end - rss_start

    # Net memory growth across 15,000 flows must be bounded (< 40 MB)
    assert rss_growth < 40.0, f"Memory growth too high: {rss_growth:.2f} MB"
    assert ctx.total_processed == n_flows
    assert ctx.is_degraded is False


# -----------------------------------------------------------------------------
# Test 8: End-to-End Streaming Simulation Stability
# -----------------------------------------------------------------------------

def test_end_to_end_streaming_simulation_stability(tmp_path):
    """Verify end-to-end streaming simulation stability on 100 flows with dual-model execution."""
    n_flows = 100
    df = generate_synthetic_flow_df(n_flows, interval_ms=10.0, attack_ratio=0.2)
    p_file = tmp_path / "e2e_stability.parquet"
    df.to_parquet(p_file)
    feed_path = tmp_path / "e2e_feed.json"

    t0 = time.perf_counter()
    summary = run_stream_simulation(
        speed_eps=0,
        max_events=n_flows,
        inject_attacks=False,
        chunk_size=25,
        v1_only=False,
        test_file=p_file,
        live_feed_file=feed_path,
        write_feed=True,
        verbose=False,
    )
    runtime = time.perf_counter() - t0

    assert summary["total_processed"] == n_flows
    assert summary["total_anomalies"] >= 0
    assert summary["telemetry_health"]["status"] == "HEALTHY"
    assert feed_path.exists()
    assert runtime < 15.0, f"End-to-end execution took {runtime:.2f}s, expected < 15.0s"
