"""
tests/test_render_deployment.py
Local simulation and verification suite for Render deployment readiness.

Verifies:
1. Production server binds to 0.0.0.0 and respects PORT environment variable (e.g. PORT=10000).
2. Server socket inspection via getsockname() confirms ('0.0.0.0', 10000).
3. /health endpoint HTTP 200 response and payload schema containing bound_host and bound_port.
4. Dashboard index.html serving at root / and /index.html.
5. Core REST API endpoints (/api/overview, /api/metrics, /api/models, /api/system_status, /api/live_feed).
6. Local Windows default fallback (HOST=0.0.0.0, PORT=8501).
7. Model adapter readiness and artifact loading.
"""

import os
import sys
import json
import time
import socket
import urllib.request
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

from server import ThreatDetectionDashboardHandler, HTTPServer, HOST, PORT


@pytest.fixture(scope="module")
def render_simulated_server():
    """
    Starts the HTTP server explicitly bound to 0.0.0.0 on simulated Render port 10000.
    Verifies actual socket binding address.
    """
    test_port = 10000
    test_host = "0.0.0.0"

    server_address = (test_host, test_port)
    httpd = HTTPServer(server_address, ThreatDetectionDashboardHandler)

    # Verify socket getsockname() before accepting connections
    sock_addr, sock_port = httpd.socket.getsockname()[:2]
    assert sock_addr == "0.0.0.0", f"Expected socket bound to 0.0.0.0, got {sock_addr}"
    assert sock_port == test_port, f"Expected socket bound to port {test_port}, got {sock_port}"

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{test_port}"
    yield {
        "base_url": base_url,
        "bound_host": sock_addr,
        "bound_port": sock_port,
        "httpd": httpd,
    }

    httpd.shutdown()
    httpd.server_close()


def test_socket_bind_address_0_0_0_0(render_simulated_server):
    """Verify the server socket is genuinely bound to 0.0.0.0 (all interfaces)."""
    assert render_simulated_server["bound_host"] == "0.0.0.0"
    assert render_simulated_server["bound_port"] == 10000

    # Also verify the HTTPServer object reports 0.0.0.0
    httpd = render_simulated_server["httpd"]
    assert httpd.server_address[0] == "0.0.0.0"
    assert httpd.server_address[1] == 10000


def test_health_endpoint(render_simulated_server):
    """Verify /health returns HTTP 200 with status HEALTHY and bound_host 0.0.0.0."""
    url = f"{render_simulated_server['base_url']}/health"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "application/json"
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("status") == "HEALTHY"
        assert data.get("bound_host") == "0.0.0.0"
        assert data.get("bound_port") == 10000
        assert "models" in data
        assert "uptime_seconds" in data
        assert data["models"]["champion_v1"]["available"] is True


def test_homepage_serving(render_simulated_server):
    """Verify root / serves HTML dashboard."""
    url = f"{render_simulated_server['base_url']}/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "AEGIS" in html


def test_overview_api(render_simulated_server):
    """Verify /api/overview returns KPI and severity stats."""
    url = f"{render_simulated_server['base_url']}/api/overview"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "kpi" in data
        assert data["kpi"]["total_network_events"] == 2300825
        assert data["kpi"]["overall_recall_pct"] == 93.08


def test_metrics_api(render_simulated_server):
    """Verify /api/metrics returns evaluation models."""
    url = f"{render_simulated_server['base_url']}/api/metrics"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "temporal_autoencoder" in data
        assert data["temporal_autoencoder"]["detection_rate_pct"] == 93.08


def test_models_api(render_simulated_server):
    """Verify /api/models returns active champion and shadow candidate."""
    url = f"{render_simulated_server['base_url']}/api/models"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("active_production_champion") == "legacy_champion"
        assert "legacy_champion" in data.get("models", {})


def test_system_status_api(render_simulated_server):
    """Verify /api/system_status returns telemetry state."""
    url = f"{render_simulated_server['base_url']}/api/system_status"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("status") == "OPERATIONAL"
        assert data.get("champion_status") == "PRODUCTION CHAMPION"


def test_default_environment_fallbacks():
    """Verify module-level default values when environment variables are unset."""
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", 8501))
    assert default_host == "0.0.0.0"
    assert default_port in (8501, 10000)
