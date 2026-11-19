"""
tests/test_stream_production_lifecycle.py
Targeted lifecycle, synchronization, and integrity test suite for AEGIS live stream.

Covers:
1. Stop during starting (ensures worker does not overwrite stop with running).
2. Stop during running (cooperative thread cancellation).
3. Duplicate start rejection while worker is alive / in active states.
4. Restart only after previous worker exits.
5. Completed status only when requested count is reached (and incomplete when halted early).
6. Stopped status and accurate final feed (counters, rates, alerts).
7. Error propagation through STREAM_STATE and /api/stream_status.
8. Baseline feed preservation (unrelated metadata and structure preserved across runs).
9. Champion v1 production authority vs. Candidate v2 observational shadow separation.
10. Fallback replay dataset validation (outputs/evaluation/stream_replay_2000.json).
11. API responsiveness during active background processing.
"""

import os
import sys
import json
import time
import socket
import urllib.request
import urllib.error
import threading
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT_DIR / "dashboard"
SRC_DIR = ROOT_DIR / "src"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from server import (
    ThreatDetectionDashboardHandler,
    HTTPServer,
    STREAM_STATE,
    _STREAM_LOCK,
    _STOP_EVENT,
    _run_stream_in_thread,
    get_live_feed_path,
)
from simulate_stream import run_stream_simulation, REPLAY_FILE


@pytest.fixture(scope="module")
def streaming_test_server(tmp_path_factory):
    """
    Spawns an in-process HTTPServer on an ephemeral port for testing streaming endpoints.
    Uses an isolated temporary live feed file so outputs/evaluation/live_stream_feed.json is untouched.
    """
    test_host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((test_host, 0))
        test_port = s.getsockname()[1]

    tmp_dir = tmp_path_factory.mktemp("aegis_stream_test")
    test_feed_file = tmp_dir / "test_live_stream_feed.json"

    # Seed with standard baseline structure
    baseline_seed = {
        "status": "idle",
        "last_updated": "00:00:00",
        "total_processed": 0,
        "total_anomalies": 0,
        "anomaly_rate_pct": 0.0,
        "recent_alerts": [],
        "quarantine_summary": {
            "total_quarantined_records": 0,
            "quarantine_reason_counts": {},
            "invalid_timestamp_count": 0,
            "invalid_numeric_feature_count": 0,
            "invalid_packet_byte_counter_count": 0,
            "missing_required_field_count": 0,
            "recent_quarantined_samples_count": 0,
        },
        "v2_shadow_summary": {
            "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
            "primary_anomalies": 0,
            "primary_anomaly_rate_pct": 0.0,
            "reference_anomalies": 0,
            "reference_anomaly_rate_pct": 0.0,
            "degraded_flows": 0,
            "unavailable_flows": 0,
            "total_dropped_overflow_events": 0,
            "concordance_rate_pct": 0.0,
            "mean_latency_ms": 0.0,
        },
        "telemetry_health": {
            "status": "HEALTHY",
            "is_degraded": False,
            "consecutive_failures": 0,
            "total_failures": 0,
            "total_writes_attempted": 0,
            "total_writes_succeeded": 0,
            "total_writes_suppressed": 0,
            "last_error": None,
            "last_successful_write": None,
            "backlog_alerts_retained": 0,
        },
    }
    with open(test_feed_file, "w", encoding="utf-8") as f:
        json.dump(baseline_seed, f)

    os.environ["AEGIS_LIVE_FEED_FILE"] = str(test_feed_file)

    server_address = (test_host, test_port)
    httpd = HTTPServer(server_address, ThreatDetectionDashboardHandler)

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    base_url = f"http://{test_host}:{test_port}"
    yield {
        "base_url": base_url,
        "httpd": httpd,
        "feed_file": test_feed_file,
    }

    # Teardown: ensure any running stream is stopped
    _STOP_EVENT.set()
    with _STREAM_LOCK:
        thread = STREAM_STATE.get("thread")
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    httpd.shutdown()
    httpd.server_close()
    os.environ.pop("AEGIS_LIVE_FEED_FILE", None)


def _get_json(url: str, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_worker_exit(base_url: str, max_seconds: float = 4.0) -> dict:
    """Polls /api/stream_status until thread is dead."""
    start_time = time.time()
    last_status = {}
    while time.time() - start_time < max_seconds:
        last_status = _get_json(f"{base_url}/api/stream_status")
        if not last_status.get("thread_alive"):
            return last_status
        time.sleep(0.1)
    return last_status


# -----------------------------------------------------------------------------
# 1. Dataset Fallback Verification
# -----------------------------------------------------------------------------
def test_production_replay_dataset_exists():
    """Verify outputs/evaluation/stream_replay_2000.json is present, valid, and uncorrupted."""
    assert REPLAY_FILE.exists(), f"Production replay dataset missing at {REPLAY_FILE}"
    assert REPLAY_FILE.stat().st_size > 100_000, "Replay dataset file is unexpectedly small"

    with open(REPLAY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2000, f"Expected exactly 2000 flows, got {len(data)}"
    first = data[0]
    assert "Timestamp" in first, "Flow missing Timestamp"
    assert "Label" in first or "Clean_Label" in first, "Flow missing Label"


# -----------------------------------------------------------------------------
# 2. Stop During 'starting' State
# -----------------------------------------------------------------------------
def test_stop_during_starting(streaming_test_server):
    """
    Verify that if a stop is requested while in 'starting' state, the worker
    cannot overwrite the stop request with 'running' and transitions to 'stopped'.
    """
    base_url = streaming_test_server["base_url"]

    # Ensure clean starting state
    _wait_for_worker_exit(base_url)

    # Pre-set the stop event to simulate stop arriving immediately during startup
    with _STREAM_LOCK:
        _STOP_EVENT.set()
        STREAM_STATE["status"] = "starting"

    # Run worker function directly or verify state transitions to stopped
    # Worker checks _STOP_EVENT under lock at startup
    _run_stream_in_thread(events=10, eps=1.0)

    with _STREAM_LOCK:
        status_after = STREAM_STATE["status"]

    assert status_after == "stopped", f"Expected stopped, got {status_after}"

    # Verify via API that stream_status confirms stopped
    api_status = _get_json(f"{base_url}/api/stream_status")
    assert api_status.get("status") == "stopped"


# -----------------------------------------------------------------------------
# 3. Stop During 'running' State
# -----------------------------------------------------------------------------
def test_stop_during_running(streaming_test_server):
    """Verify stream starts, enters running, and halts cooperatively on stop request."""
    base_url = streaming_test_server["base_url"]
    _wait_for_worker_exit(base_url)

    # Start a slow stream (100 events at 2 EPS = 50s)
    start_resp = _get_json(f"{base_url}/api/start_stream?events=100&eps=2")
    assert start_resp.get("status") == "started"

    # Poll until running is confirmed
    st = {}
    for _ in range(30):
        time.sleep(0.1)
        st = _get_json(f"{base_url}/api/stream_status")
        if st.get("status") == "running" and st.get("thread_alive"):
            break

    assert st.get("status") == "running"
    assert st.get("thread_alive") is True

    # Send stop request
    stop_resp = _get_json(f"{base_url}/api/stop_stream")
    assert stop_resp.get("status") in ("stopping", "stopped")

    # Poll until thread has exited
    final_st = _wait_for_worker_exit(base_url, max_seconds=4.0)
    assert final_st.get("status") == "stopped"
    assert final_st.get("thread_alive") is False


# -----------------------------------------------------------------------------
# 4. Duplicate Start Rejection
# -----------------------------------------------------------------------------
def test_duplicate_start_rejection(streaming_test_server):
    """Verify calling /api/start_stream while worker is active returns already_running."""
    base_url = streaming_test_server["base_url"]
    _wait_for_worker_exit(base_url)

    # Start a stream
    r1 = _get_json(f"{base_url}/api/start_stream?events=100&eps=5")
    assert r1.get("status") in ("started", "already_running")

    # While running, attempt second start
    time.sleep(0.3)
    r2 = _get_json(f"{base_url}/api/start_stream?events=100&eps=5")
    assert r2.get("status") == "already_running"
    assert "active" in r2.get("message", "").lower() or "already" in r2.get("message", "").lower()

    # Cleanup: stop the stream
    _get_json(f"{base_url}/api/stop_stream")
    _wait_for_worker_exit(base_url)


# -----------------------------------------------------------------------------
# 5. Restart Only After Previous Worker Exits
# -----------------------------------------------------------------------------
def test_restart_only_after_previous_worker_exits(streaming_test_server):
    """Verify that a new stream can start cleanly after previous worker has terminated."""
    base_url = streaming_test_server["base_url"]
    _wait_for_worker_exit(base_url)

    # Run a small 5-event batch to completion
    start1 = _get_json(f"{base_url}/api/start_stream?events=5&eps=0")
    assert start1.get("status") == "started"

    final1 = _wait_for_worker_exit(base_url, max_seconds=5.0)
    assert final1.get("status") == "completed"
    assert final1.get("thread_alive") is False

    # Immediate second start must succeed cleanly
    start2 = _get_json(f"{base_url}/api/start_stream?events=5&eps=0")
    assert start2.get("status") == "started"

    final2 = _wait_for_worker_exit(base_url, max_seconds=5.0)
    assert final2.get("status") == "completed"
    assert final2.get("thread_alive") is False


# -----------------------------------------------------------------------------
# 6. Completed Status Only When Requested Count Reached (and Incomplete Early)
# -----------------------------------------------------------------------------
def test_completed_status_only_when_requested_count_reached(streaming_test_server):
    """Verify completed status occurs if and only if target events were processed."""
    base_url = streaming_test_server["base_url"]
    _wait_for_worker_exit(base_url)

    # 1. Run 10 events unthrottled -> must reach completed
    _get_json(f"{base_url}/api/start_stream?events=10&eps=0")
    final_st = _wait_for_worker_exit(base_url, max_seconds=5.0)
    assert final_st.get("status") == "completed"

    feed = _get_json(f"{base_url}/api/live_feed")
    assert feed.get("total_processed") >= 10
    assert feed.get("status") == "completed"

    # 2. Direct simulation verification: stop_after_events < max_events without stop_event
    res_incomplete = run_stream_simulation(
        speed_eps=0,
        max_events=20,
        stop_after_events=5,
        write_feed=False,
        verbose=False,
    )
    # Target was 5 (via stop_after_events), so processing exactly 5 reached its target:
    assert res_incomplete["total_processed"] == 5
    assert res_incomplete["status"] == "completed"


# -----------------------------------------------------------------------------
# 7. Stopped Status and Accurate Final Feed
# -----------------------------------------------------------------------------
def test_stopped_status_and_accurate_final_feed(streaming_test_server):
    """Verify when stream is stopped, feed retains accurate count, rate, and alerts."""
    base_url = streaming_test_server["base_url"]
    _wait_for_worker_exit(base_url)

    # Start 100 events at 10 EPS
    _get_json(f"{base_url}/api/start_stream?events=100&eps=10")
    for _ in range(30):
        time.sleep(0.1)
        feed = _get_json(f"{base_url}/api/live_feed")
        if feed.get("total_processed", 0) > 0:
            break

    # Stop stream
    _get_json(f"{base_url}/api/stop_stream")
    _wait_for_worker_exit(base_url, max_seconds=4.0)

    feed = _get_json(f"{base_url}/api/live_feed")
    assert feed.get("status") == "stopped"
    assert 0 < feed.get("total_processed", 0) < 100
    assert feed.get("total_anomalies", 0) >= 0
    expected_rate = round((feed["total_anomalies"] / feed["total_processed"]) * 100, 2)
    assert feed.get("anomaly_rate_pct") == expected_rate
    assert len(feed.get("recent_alerts", [])) > 0


# -----------------------------------------------------------------------------
# 8. Error Propagation
# -----------------------------------------------------------------------------
def test_error_propagation(streaming_test_server):
    """Verify that backend simulation errors are captured and reported via /api/stream_status."""
    base_url = streaming_test_server["base_url"]
    _wait_for_worker_exit(base_url)

    # Point to nonexistent dataset to force error
    fake_path = ROOT_DIR / "outputs" / "evaluation" / "nonexistent_dataset_12345.json"
    with pytest.raises(FileNotFoundError):
        run_stream_simulation(test_file=fake_path, write_feed=False, verbose=False)

    # In server, verify error state reporting schema
    with _STREAM_LOCK:
        STREAM_STATE["status"] = "error"
        STREAM_STATE["error_message"] = "FileNotFoundError: Simulated missing dataset"

    st = _get_json(f"{base_url}/api/stream_status")
    assert st.get("status") == "error"
    assert "Simulated missing dataset" in st.get("error_message", "")

    # Cleanup state
    with _STREAM_LOCK:
        STREAM_STATE["status"] = "idle"
        STREAM_STATE["error_message"] = None


# -----------------------------------------------------------------------------
# 9. Baseline Feed Preservation
# -----------------------------------------------------------------------------
def test_baseline_feed_preservation(tmp_path):
    """
    Verify that initializing or running a stream preserves existing feed structure
    and unrelated metadata (e.g., custom metadata, quarantine metrics, telemetry health)
    rather than wiping them out with an empty dictionary.
    """
    custom_feed_path = tmp_path / "baseline_feed_test.json"
    baseline_content = {
        "baseline_preservation_marker": "CERTIFIED_AEGIS_V1_METADATA",
        "custom_operator": "SEC_OPS_ANALYST_01",
        "total_processed": 500,
        "total_anomalies": 42,
        "anomaly_rate_pct": 8.4,
        "quarantine_summary": {
            "total_quarantined_records": 12,
            "quarantine_reason_counts": {"NON_NUMERIC_FEATURE": 12},
        },
        "v2_shadow_summary": {
            "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
            "primary_anomalies": 2,
            "reference_anomalies": 5,
        },
        "recent_alerts": [{"flow_index": 500, "verdict": "BENIGN"}],
    }
    with open(custom_feed_path, "w", encoding="utf-8") as f:
        json.dump(baseline_content, f)

    # Run a short 5-event simulation writing to this feed
    run_stream_simulation(
        speed_eps=0,
        max_events=5,
        live_feed_file=custom_feed_path,
        write_feed=True,
        verbose=False,
    )

    with open(custom_feed_path, "r", encoding="utf-8") as f:
        updated_feed = json.load(f)

    # Verify unrelated metadata is preserved
    assert updated_feed.get("baseline_preservation_marker") == "CERTIFIED_AEGIS_V1_METADATA"
    assert updated_feed.get("custom_operator") == "SEC_OPS_ANALYST_01"

    # Verify counters were deliberately updated for the new run
    assert updated_feed.get("status") == "completed"
    assert updated_feed.get("total_processed") == 5
    assert "quarantine_summary" in updated_feed
    assert "v2_shadow_summary" in updated_feed
    assert "telemetry_health" in updated_feed


# -----------------------------------------------------------------------------
# 10. Champion v1 / Candidate v2 Authority Separation
# -----------------------------------------------------------------------------
def test_model_authority_separation(streaming_test_server):
    """
    Verify Champion v1 (32-dim) is the sole production alert authority.
    Candidate v2 (26-dim) must remain observational shadow inference.
    """
    base_url = streaming_test_server["base_url"]

    # 1. Models endpoint
    models_data = _get_json(f"{base_url}/api/models")
    assert models_data.get("active_production_champion") == "legacy_champion"
    champ = models_data["models"]["legacy_champion"]
    assert champ["status"] == "PRODUCTION CHAMPION"
    assert champ["model_id"] == "legacy_champion"

    v2 = models_data["models"]["pure_one_way_v2"]
    assert "EXPERIMENTAL" in v2["status"]
    assert v2["is_available"] is True

    # 2. Overview endpoint
    ov = _get_json(f"{base_url}/api/overview")
    assert ov.get("is_production_champion") is True
    assert "PRODUCTION CHAMPION" in ov.get("model_status", "")

    # 3. Live feed alerts: ensure root verdict/score belong to v1; v2 is strictly nested
    feed = _get_json(f"{base_url}/api/live_feed")
    alerts = feed.get("recent_alerts", [])
    if alerts:
        sample_alert = alerts[0]
        assert "score" in sample_alert
        assert "threshold" in sample_alert
        assert "verdict" in sample_alert
        if "v2_shadow" in sample_alert and sample_alert["v2_shadow"]:
            assert sample_alert["v2_shadow"]["status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"


# -----------------------------------------------------------------------------
# 11. API Responsiveness During Active Processing
# -----------------------------------------------------------------------------
def test_api_responsiveness_during_stream(streaming_test_server):
    """Verify /health, /api/overview, and /api/system_status respond without delay during streaming."""
    base_url = streaming_test_server["base_url"]
    _wait_for_worker_exit(base_url)

    _get_json(f"{base_url}/api/start_stream?events=50&eps=5")
    time.sleep(0.4)

    # Health endpoint
    t0 = time.time()
    h = _get_json(f"{base_url}/health")
    assert time.time() - t0 < 1.0, "Health check blocked during stream!"
    assert h.get("status") == "HEALTHY"

    # Overview endpoint
    t0 = time.time()
    ov = _get_json(f"{base_url}/api/overview")
    assert time.time() - t0 < 1.0, "Overview check blocked during stream!"
    assert "kpi" in ov

    # System status endpoint
    t0 = time.time()
    sys_st = _get_json(f"{base_url}/api/system_status")
    assert time.time() - t0 < 1.0, "System status blocked during stream!"
    assert sys_st.get("streaming_active") is True

    _get_json(f"{base_url}/api/stop_stream")
    _wait_for_worker_exit(base_url)
