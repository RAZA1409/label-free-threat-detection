"""
tests/test_threat_analytics.py
AEGIS Dynamic Threat Analytics Verification Suite

Verifies:
1. Backend /api/threat_analytics and /api/attack_metrics endpoints:
   - Returns HTTP 200 with authentic evaluation data for Legacy v1 Champion.
   - Returns identical contract for /api/attack_metrics alias.
   - Supports ?model=all returning all evaluated models.
2. Category-level metrics fidelity:
   - DDoS, DoS Hulk, PortScan, Bot, Infiltration, Heartbleed, and Web Attacks match empirical source splits.
   - Total count, detected (TP), missed (FN), recall, and precision only exposed when supported by source data.
3. Missing categories and error handling:
   - Requesting non-existent model returns HTTP 404 with structured error JSON.
   - Non-existent category query returns UNAVAILABLE status with null metrics (never invented).
4. v1 Production Champion vs. v2 Experimental Shadow Isolation:
   - v1 has is_production_champion=True, threshold 0.190885, 32 dims.
   - v2 primary has is_production_champion=False, threshold 0.314036, 26 dims.
   - v2 reference has is_production_champion=False, threshold 0.099018, 26 dims.
5. Frontend HTML integration:
   - Model switcher buttons present.
   - Dynamic confusion matrix element IDs present.
   - Per-attack breakdown table (#analytics-attack-tbody) present.
   - Shadow notice banner and error status banner present.
"""

import sys
import json
import urllib.request
import urllib.error
import threading
import time
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DASHBOARD_DIR = ROOT_DIR / "dashboard"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from server import ThreatDetectionDashboardHandler, HTTPServer, get_threat_analytics, CANONICAL_ATTACK_CATEGORIES


@pytest.fixture(scope="module")
def live_server():
    """Launches test HTTP server on localhost:8596 for isolated threat analytics testing."""
    server_addr = ("127.0.0.1", 8596)
    httpd = HTTPServer(server_addr, ThreatDetectionDashboardHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)
    yield "http://127.0.0.1:8596"
    httpd.shutdown()
    httpd.server_close()


# -----------------------------------------------------------------------------
# TEST 1: Backend Threat Analytics API Endpoint & Aliasing
# -----------------------------------------------------------------------------
def test_01_threat_analytics_endpoint_and_alias(live_server):
    """Verify GET /api/threat_analytics and /api/attack_metrics return 200 with authentic v1 data."""
    # 1. Primary endpoint
    req1 = urllib.request.urlopen(f"{live_server}/api/threat_analytics")
    assert req1.getcode() == 200
    data1 = json.loads(req1.read().decode("utf-8"))

    assert data1["status"] == "SUCCESS"
    assert data1["model_id"] == "legacy_champion_v1"
    assert data1["model_name"] == "Legacy Temporal Autoencoder (v1)"
    assert data1["model_status"] == "PRODUCTION CHAMPION"
    assert data1["is_production_champion"] is True
    assert data1["feature_count"] == 32
    assert round(float(data1["threshold"]), 6) == 0.190885
    assert data1["total_flows"] == 2300825
    assert data1["attack_flows"] == 557646
    assert data1["benign_flows"] == 1743179

    cm = data1["confusion_matrix"]
    assert cm["tp"] == 519031
    assert cm["fp"] == 217598
    assert cm["fn"] == 38615
    assert cm["tn"] == 1525581

    # 2. Alias endpoint /api/attack_metrics
    req2 = urllib.request.urlopen(f"{live_server}/api/attack_metrics")
    assert req2.getcode() == 200
    data2 = json.loads(req2.read().decode("utf-8"))
    assert data2["model_id"] == data1["model_id"]
    assert data2["confusion_matrix"]["tp"] == data1["confusion_matrix"]["tp"]


# -----------------------------------------------------------------------------
# TEST 2: All 14 Empirical Attack Categories Present with Authentic Metrics
# -----------------------------------------------------------------------------
def test_02_all_14_attack_categories_authentic_metrics(live_server):
    """Verify all 14 canonical attack categories are evaluated with exact empirical numbers."""
    req = urllib.request.urlopen(f"{live_server}/api/threat_analytics?model=legacy")
    data = json.loads(req.read().decode("utf-8"))

    categories = {c["category"]: c for c in data["categories"]}
    assert len(categories) == 14
    for cat_name in CANONICAL_ATTACK_CATEGORIES:
        assert cat_name in categories, f"Missing canonical attack category: {cat_name}"
        assert categories[cat_name]["status"] == "AVAILABLE"

    # Verify DDoS
    ddos = categories["DDoS"]
    assert ddos["total_count"] == 128027
    assert ddos["detected_count"] == 125253
    assert ddos["missed_count"] == 2774
    assert abs(ddos["recall_pct"] - 97.83) < 0.05
    assert abs(ddos["precision_pct"] - 17.00) < 0.05

    # Verify DoS Hulk
    hulk = categories["DoS Hulk"]
    assert hulk["total_count"] == 231073
    assert hulk["detected_count"] == 230726
    assert hulk["missed_count"] == 347
    assert abs(hulk["recall_pct"] - 99.85) < 0.05

    # Verify PortScan
    pscan = categories["PortScan"]
    assert pscan["total_count"] == 158930
    assert pscan["detected_count"] == 158601
    assert pscan["missed_count"] == 329
    assert abs(pscan["recall_pct"] - 99.79) < 0.05

    # Verify Infiltration
    infil = categories["Infiltration"]
    assert infil["total_count"] == 36
    assert infil["detected_count"] == 21
    assert infil["missed_count"] == 15
    assert abs(infil["recall_pct"] - 58.33) < 0.05

    # Verify Heartbleed
    hb = categories["Heartbleed"]
    assert hb["total_count"] == 11
    assert hb["detected_count"] == 6
    assert hb["missed_count"] == 5
    assert abs(hb["recall_pct"] - 54.55) < 0.05

    # Verify Bot
    bot = categories["Bot"]
    assert bot["total_count"] == 1966
    assert bot["detected_count"] == 45
    assert bot["missed_count"] == 1921
    assert abs(bot["recall_pct"] - 2.29) < 0.05


# -----------------------------------------------------------------------------
# TEST 3: Multi-Model Query (?model=all)
# -----------------------------------------------------------------------------
def test_03_multi_model_query_all(live_server):
    """Verify ?model=all returns all 3 models with explicit champion designation."""
    req = urllib.request.urlopen(f"{live_server}/api/threat_analytics?model=all")
    assert req.getcode() == 200
    data = json.loads(req.read().decode("utf-8"))

    assert data["status"] == "SUCCESS"
    assert data["production_champion_id"] == "legacy_champion_v1"
    models = data["models"]
    assert "legacy_champion_v1" in models
    assert "pure_one_way_v2_primary" in models
    assert "pure_one_way_v2_reference" in models

    assert models["legacy_champion_v1"]["is_production_champion"] is True
    assert models["pure_one_way_v2_primary"]["is_production_champion"] is False
    assert models["pure_one_way_v2_reference"]["is_production_champion"] is False


# -----------------------------------------------------------------------------
# TEST 4: Missing Categories and Error Handling
# -----------------------------------------------------------------------------
def test_04_missing_categories_and_error_handling(live_server):
    """Verify unknown model returns 404 and unknown category returns UNAVAILABLE."""
    # 1. Unknown model -> HTTP 404
    try:
        urllib.request.urlopen(f"{live_server}/api/threat_analytics?model=nonexistent_xyz")
        pytest.fail("Expected 404 for invalid model")
    except urllib.error.HTTPError as e:
        assert e.code == 404
        err_data = json.loads(e.read().decode("utf-8"))
        assert err_data["status"] == "NOT_FOUND"
        assert "available_models" in err_data

    # 2. Unknown category query -> UNAVAILABLE status, null metrics
    req = urllib.request.urlopen(f"{live_server}/api/threat_analytics?category=ZeroDayExploit")
    assert req.getcode() == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "filtered_category" in data
    fc = data["filtered_category"]
    assert fc["category"] == "ZeroDayExploit"
    assert fc["status"] == "UNAVAILABLE"
    assert fc["total_count"] is None
    assert fc["detected_count"] is None
    assert fc["recall_pct"] is None
    assert fc["precision_pct"] is None


# -----------------------------------------------------------------------------
# TEST 5: v1 Champion vs. v2 Shadow Isolation
# -----------------------------------------------------------------------------
def test_05_v1_v2_model_isolation(live_server):
    """Verify v1 is champion while v2 primary and reference remain shadow experimental."""
    # 1. v1 Champion
    req_v1 = urllib.request.urlopen(f"{live_server}/api/threat_analytics?model=v1")
    v1 = json.loads(req_v1.read().decode("utf-8"))
    assert v1["is_production_champion"] is True
    assert v1["feature_count"] == 32
    assert round(float(v1["threshold"]), 6) == 0.190885
    assert v1["confusion_matrix"]["tp"] == 519031
    assert v1["confusion_matrix"]["fp"] == 217598

    # 2. v2 Primary (Shadow)
    req_v2p = urllib.request.urlopen(f"{live_server}/api/threat_analytics?model=v2_primary")
    v2p = json.loads(req_v2p.read().decode("utf-8"))
    assert v2p["is_production_champion"] is False
    assert v2p["model_status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"
    assert v2p["feature_count"] == 26
    assert round(float(v2p["threshold"]), 6) == 0.314036
    assert v2p["confusion_matrix"]["tp"] == 476125
    assert v2p["confusion_matrix"]["fp"] == 186489

    # 3. v2 Reference (Shadow)
    req_v2r = urllib.request.urlopen(f"{live_server}/api/threat_analytics?model=v2_reference")
    v2r = json.loads(req_v2r.read().decode("utf-8"))
    assert v2r["is_production_champion"] is False
    assert v2r["model_status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"
    assert v2r["feature_count"] == 26
    assert round(float(v2r["threshold"]), 6) == 0.099018
    assert v2r["confusion_matrix"]["tp"] == 518647
    assert v2r["confusion_matrix"]["fp"] == 302461


# -----------------------------------------------------------------------------
# TEST 6: Frontend HTML Elements & Dynamic Binding
# -----------------------------------------------------------------------------
def test_06_frontend_html_analytics_elements():
    """Verify dashboard/index.html includes all required Threat Analytics elements."""
    index_path = ROOT_DIR / "dashboard" / "index.html"
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Switcher buttons
    assert 'id="btn-analytics-v1"' in html
    assert 'id="btn-analytics-v2-primary"' in html
    assert 'id="btn-analytics-v2-ref"' in html

    # 2. Badges and banners
    assert 'id="analytics-active-badge"' in html
    assert 'id="analytics-authority-note"' in html
    assert 'id="analytics-shadow-banner"' in html
    assert 'id="analytics-status-banner"' in html

    # 3. Dynamic Confusion Matrix elements
    assert 'id="analytics-cm-tp"' in html
    assert 'id="analytics-cm-fp"' in html
    assert 'id="analytics-cm-fn"' in html
    assert 'id="analytics-cm-tn"' in html
    assert 'id="analytics-precision"' in html
    assert 'id="analytics-recall"' in html
    assert 'id="analytics-prauc"' in html
    assert 'id="analytics-threshold"' in html

    # 4. Breakdown Table
    assert 'id="analytics-attack-tbody"' in html
    assert 'id="analytics-table-model-tag"' in html
    assert 'id="analytics-table-summary"' in html

    # 5. Core JS functions
    assert "function fetchThreatAnalytics(" in html
    assert "function switchAnalyticsModel(" in html
    assert "switchView('analytics')" in html or "switchView(" in html

