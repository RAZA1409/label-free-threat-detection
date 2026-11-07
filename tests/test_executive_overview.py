"""
tests/test_executive_overview.py
================================

Verification suite for AEGIS Executive Overview integration:
- Real evaluation metrics connected to Overview KPI cards (flows, recall, FPR, precision, F1)
- Live optical diode streaming telemetry segregated from benchmark evaluation
- Model selector supporting v1 champion, v2 primary shadow, and v2 reference shadow
- Dominant attack classes table and severity distribution charts dynamic data
- Backward compatibility with existing dashboard test suite
"""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "dashboard"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from server import ThreatDetectionDashboardHandler, HTTPServer


@pytest.fixture(scope="module")
def live_server():
    server_addr = ("127.0.0.1", 8597)
    httpd = HTTPServer(server_addr, ThreatDetectionDashboardHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)
    yield "http://127.0.0.1:8597"
    httpd.shutdown()
    httpd.server_close()


# -----------------------------------------------------------------------------
# Test 1: Default call preserves v1 production champion metrics
# -----------------------------------------------------------------------------
def test_01_overview_default_champion_metrics(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/overview")
    assert req.getcode() == 200
    data = json.loads(req.read().decode("utf-8"))

    # Model metadata
    assert data["model_id"] == "legacy_champion_v1"
    assert data["is_production_champion"] is True
    assert data["model_status"] == "PRODUCTION CHAMPION"
    assert "Legacy Temporal Autoencoder" in data["model_name"]

    # Invariant: 2,300,825 total test flows
    kpi = data["kpi"]
    assert kpi["total_network_events"] == 2300825
    assert kpi["benign_events"] == 1743179
    assert kpi["total_attacks"] == 557646
    assert kpi["attacks_detected"] == 519031
    assert kpi["attacks_missed"] == 38615

    # Verification of key rates
    assert kpi["overall_recall_pct"] == 93.08
    assert kpi["false_positive_rate_pct"] == 12.48
    assert kpi["observed_fpr_pct"] == 12.48
    assert kpi["precision_pct"] == 70.46
    assert kpi["f1_score"] == 0.8020
    assert kpi["average_precision"] == 0.8783
    assert kpi["calibrated_threshold"] == 0.190885
    assert kpi["target_calibration_fpr"] == 0.002


# -----------------------------------------------------------------------------
# Test 2: Overview supports v2 primary shadow candidate
# -----------------------------------------------------------------------------
def test_02_overview_v2_primary_metrics(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/overview?model=v2_primary")
    assert req.getcode() == 200
    data = json.loads(req.read().decode("utf-8"))

    assert data["model_id"] == "pure_one_way_v2_primary"
    assert data["is_production_champion"] is False
    assert data["model_status"] in ("EXPERIMENTAL / SHADOW CANDIDATE", "EXPERIMENTAL / RESEARCH CANDIDATE")

    kpi = data["kpi"]
    assert kpi["total_network_events"] == 2300825
    assert kpi["attacks_detected"] == 476125
    assert kpi["attacks_missed"] == 81521
    assert kpi["overall_recall_pct"] == 85.38
    assert kpi["false_positive_rate_pct"] == 10.70
    assert kpi["precision_pct"] == 71.86
    assert kpi["f1_score"] == 0.7804
    assert kpi["average_precision"] == 0.8687
    assert kpi["calibrated_threshold"] == 0.314036
    assert kpi["target_calibration_fpr"] == 0.002


# -----------------------------------------------------------------------------
# Test 3: Overview supports v2 reference shadow candidate
# -----------------------------------------------------------------------------
def test_03_overview_v2_reference_metrics(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/overview?model=v2_reference")
    assert req.getcode() == 200
    data = json.loads(req.read().decode("utf-8"))

    assert data["model_id"] == "pure_one_way_v2_reference"
    assert data["is_production_champion"] is False
    assert data["model_status"] in ("EXPERIMENTAL / SHADOW CANDIDATE", "EXPERIMENTAL / RESEARCH CANDIDATE")

    kpi = data["kpi"]
    assert kpi["total_network_events"] == 2300825
    assert kpi["attacks_detected"] == 518647
    assert kpi["attacks_missed"] == 38999
    assert kpi["overall_recall_pct"] == 93.01
    assert kpi["false_positive_rate_pct"] == 17.35
    assert kpi["precision_pct"] == 63.16
    assert kpi["f1_score"] == 0.7523
    assert kpi["average_precision"] == 0.8687
    assert kpi["calibrated_threshold"] == 0.099018
    assert kpi["target_calibration_fpr"] == 0.010


# -----------------------------------------------------------------------------
# Test 4: Model aliases are normalized cleanly
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("alias,expected_id", [
    ("legacy", "legacy_champion_v1"),
    ("v1", "legacy_champion_v1"),
    ("temporal_ae", "legacy_champion_v1"),
    ("v2", "pure_one_way_v2_primary"),
    ("pure_oneway_v2", "pure_one_way_v2_primary"),
    ("v2_primary", "pure_one_way_v2_primary"),
    ("v2_ref", "pure_one_way_v2_reference"),
    ("v2_reference", "pure_one_way_v2_reference"),
    ("pure_oneway_v2_ref", "pure_one_way_v2_reference"),
])
def test_04_model_aliases_support(live_server, alias, expected_id):
    req = urllib.request.urlopen(f"{live_server}/api/overview?model={alias}")
    assert req.getcode() == 200
    data = json.loads(req.read().decode("utf-8"))
    assert data["model_id"] == expected_id


# -----------------------------------------------------------------------------
# Test 5: Invalid model query returns 400 Bad Request
# -----------------------------------------------------------------------------
def test_05_invalid_model_returns_400(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/api/overview?model=nonexistent_model")
    assert exc_info.value.code == 400
    body = json.loads(exc_info.value.read().decode("utf-8"))
    assert body["status"] == "ERROR"
    assert "Unknown model" in body["error"]


# -----------------------------------------------------------------------------
# Test 6: Dominant attack classes breakdown is genuine and empirical
# -----------------------------------------------------------------------------
def test_06_dominant_attacks_consistency(live_server):
    req_v1 = urllib.request.urlopen(f"{live_server}/api/overview?model=legacy")
    data_v1 = json.loads(req_v1.read().decode("utf-8"))
    assert "dominant_attacks" in data_v1
    dom_v1 = {item["attack_class"]: item for item in data_v1["dominant_attacks"]}

    assert len(dom_v1) == 5
    assert "DoS Hulk" in dom_v1
    assert "PortScan" in dom_v1
    assert "DDoS" in dom_v1
    assert "Infiltration" in dom_v1
    assert "Heartbleed" in dom_v1

    # In v1 champion:
    assert dom_v1["DoS Hulk"]["total_volume"] == 231073
    assert dom_v1["DoS Hulk"]["detected"] == 230726
    assert dom_v1["PortScan"]["detected"] == 158601
    assert dom_v1["Infiltration"]["detected"] == 21

    # In v2 primary shadow:
    req_v2 = urllib.request.urlopen(f"{live_server}/api/overview?model=v2_primary")
    data_v2 = json.loads(req_v2.read().decode("utf-8"))
    dom_v2 = {item["attack_class"]: item for item in data_v2["dominant_attacks"]}
    assert dom_v2["PortScan"]["detected"] == 157568
    assert dom_v2["Infiltration"]["detected"] == 17


# -----------------------------------------------------------------------------
# Test 7: Severity distribution accurately reflects model performance
# -----------------------------------------------------------------------------
def test_07_severity_distribution_consistency(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/overview?model=legacy")
    data = json.loads(req.read().decode("utf-8"))
    assert "severity_distribution" in data
    sev = data["severity_distribution"]

    assert sev["Critical"] == 125259  # DDoS (125253) + Heartbleed (6)
    assert sev["High"] == 389327      # DoS Hulk (230726) + PortScan (158601)
    assert sev["Medium"] == 4199
    assert sev["Low"] == 246


# -----------------------------------------------------------------------------
# Test 8: Live telemetry is segregated from benchmark test metrics
# -----------------------------------------------------------------------------
def test_08_live_telemetry_segregation(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/overview")
    data = json.loads(req.read().decode("utf-8"))

    assert "live_telemetry" in data
    telemetry = data["live_telemetry"]
    assert "processed_flows" in telemetry
    assert "alert_count" in telemetry
    assert "buffer_status" in telemetry
    assert "streaming_active" in telemetry
    assert "model_availability" in telemetry

    # Verify that offline benchmark total flows are never altered by live ingestion
    assert data["kpi"]["total_network_events"] == 2300825


# -----------------------------------------------------------------------------
# Test 9: Frontend HTML elements for Executive Overview are present
# -----------------------------------------------------------------------------
def test_09_frontend_overview_elements_exist():
    html_path = REPO_ROOT / "dashboard" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Banner & switcher buttons
    assert 'id="ov-banner-container"' in content
    assert 'id="ov-banner-title"' in content
    assert 'id="ov-banner-badge"' in content
    assert 'id="ov-btn-v1"' in content
    assert 'id="ov-btn-v2-primary"' in content
    assert 'id="ov-btn-v2-ref"' in content

    # Live telemetry strip
    assert 'id="ov-live-processed"' in content
    assert 'id="ov-live-alerts"' in content
    assert 'id="ov-live-buffer"' in content
    assert 'id="ov-model-availability"' in content

    # 6 KPI cards
    assert 'id="ov-kpi-total"' in content
    assert 'id="ov-kpi-recall"' in content
    assert 'id="ov-kpi-fpr"' in content
    assert 'id="ov-kpi-prec"' in content
    assert 'id="ov-kpi-f1"' in content
    assert 'id="ov-kpi-ap"' in content

    # Dynamic dominant attacks table & donut severity
    assert 'id="ov-dominant-tbody"' in content
    assert 'id="ov-donut-crit"' in content
    assert 'id="ov-donut-high"' in content
    assert 'id="ov-donut-med"' in content
    assert 'id="ov-donut-low"' in content
    assert 'id="ov-donut-total-attacks"' in content

    # JavaScript functions
    assert "function switchOverviewModel" in content
    assert "async function fetchOverviewData" in content
