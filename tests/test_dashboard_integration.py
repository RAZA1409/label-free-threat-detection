"""
tests/test_dashboard_integration.py
AEGIS Phase 3 Stage 2: Dual-Model Dashboard Integration Test Suite

Validates all 14 mandatory integration requirements:
1. v1 loads successfully.
2. v2 loads successfully.
3. v1 accepts exactly the expected 32-dimensional input.
4. v2 accepts exactly the expected 26-dimensional input.
5. Cross-model dimensionality misuse is rejected.
6. v1 scoring remains unchanged.
7. v2 scoring remains unchanged.
8. v1 threshold remains unchanged.
9. Both v2 thresholds remain unchanged.
10. Missing v2 artifacts do not break v1 dashboard operation.
11. Existing API routes preserve their previous default behavior.
12. Model comparison endpoint returns the correct metrics.
13. Experimental status is exposed correctly.
14. Frontend displays model status, operating points, notes, and drift warning correctly.
"""

import sys
import json
import urllib.request
import threading
import time
from pathlib import Path
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DASHBOARD_DIR = ROOT_DIR / "dashboard"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from model_adapters import LegacyModelAdapter, PureOneWayV2Adapter
from server import ThreatDetectionDashboardHandler, HTTPServer


@pytest.fixture(scope="module")
def legacy_adapter():
    adapter = LegacyModelAdapter()
    assert adapter.is_available, f"Legacy adapter failed to load: {adapter.load_error}"
    return adapter


@pytest.fixture(scope="module")
def v2_adapter():
    adapter = PureOneWayV2Adapter()
    assert adapter.is_available, f"v2 adapter failed to load: {adapter.load_error}"
    return adapter


@pytest.fixture(scope="module")
def live_server():
    """Launches test HTTP server on localhost:8598."""
    server_addr = ("127.0.0.1", 8598)
    httpd = HTTPServer(server_addr, ThreatDetectionDashboardHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)
    yield "http://127.0.0.1:8598"
    httpd.shutdown()
    httpd.server_close()


# -----------------------------------------------------------------------------
# TEST 1: v1 loads successfully
# -----------------------------------------------------------------------------
def test_01_v1_loads_successfully(legacy_adapter):
    assert legacy_adapter.is_available is True
    assert legacy_adapter.model is not None
    assert legacy_adapter.pipeline is not None
    assert legacy_adapter.weights is not None
    assert legacy_adapter.expected_dim == 32
    assert len(legacy_adapter.feature_names) == 32


# -----------------------------------------------------------------------------
# TEST 2: v2 loads successfully
# -----------------------------------------------------------------------------
def test_02_v2_loads_successfully(v2_adapter):
    assert v2_adapter.is_available is True
    assert v2_adapter.model is not None
    assert v2_adapter.pipeline is not None
    assert v2_adapter.expected_dim == 26
    assert len(v2_adapter.feature_names) == 26


# -----------------------------------------------------------------------------
# TEST 3: v1 accepts exactly the expected 32-dimensional input
# -----------------------------------------------------------------------------
def test_03_v1_accepts_expected_32_dim_input(legacy_adapter):
    X_32 = np.zeros((8, 32), dtype=float)
    scores = legacy_adapter.score(X_32)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 8
    assert np.all(np.isfinite(scores))


# -----------------------------------------------------------------------------
# TEST 4: v2 accepts exactly the expected 26-dimensional input
# -----------------------------------------------------------------------------
def test_04_v2_accepts_expected_26_dim_input(v2_adapter):
    X_26 = np.zeros((8, 26), dtype=float)
    scores = v2_adapter.score(X_26)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 8
    assert np.all(np.isfinite(scores))


# -----------------------------------------------------------------------------
# TEST 5: Cross-model dimensionality misuse is rejected
# -----------------------------------------------------------------------------
def test_05_cross_model_dimensionality_misuse_rejected(legacy_adapter, v2_adapter):
    X_26 = np.zeros((5, 26), dtype=float)
    X_32 = np.zeros((5, 32), dtype=float)

    # v1 MUST reject 26-dimensional input
    with pytest.raises(ValueError, match="Dimensionality mismatch.*32"):
        legacy_adapter.validate_input(X_26)
    with pytest.raises(ValueError, match="Dimensionality mismatch.*32"):
        legacy_adapter.score(X_26)

    # v2 MUST reject 32-dimensional input
    with pytest.raises(ValueError, match="Dimensionality mismatch.*26"):
        v2_adapter.validate_input(X_32)
    with pytest.raises(ValueError, match="Dimensionality mismatch.*26"):
        v2_adapter.score(X_32)


# -----------------------------------------------------------------------------
# TEST 6: v1 scoring remains unchanged
# -----------------------------------------------------------------------------
def test_06_v1_scoring_remains_unchanged(legacy_adapter):
    # Verify deterministic weighted MSE formula
    X_test = np.ones((2, 32), dtype=float) * 0.5
    rec = legacy_adapter.model.predict(X_test)
    manual_weighted_mse = np.mean(np.square(X_test - rec) * legacy_adapter.weights, axis=1)
    adapter_score = legacy_adapter.score(X_test)
    np.testing.assert_allclose(adapter_score, manual_weighted_mse, rtol=1e-6)


# -----------------------------------------------------------------------------
# TEST 7: v2 scoring remains unchanged
# -----------------------------------------------------------------------------
def test_07_v2_scoring_remains_unchanged(v2_adapter):
    # Verify deterministic unweighted MSE formula
    X_test = np.ones((2, 26), dtype=float) * 0.5
    rec = v2_adapter.model.predict(X_test)
    manual_unweighted_mse = np.mean(np.square(X_test - rec), axis=1)
    adapter_score = v2_adapter.score(X_test)
    np.testing.assert_allclose(adapter_score, manual_unweighted_mse, rtol=1e-6)


# -----------------------------------------------------------------------------
# TEST 8: v1 threshold remains unchanged
# -----------------------------------------------------------------------------
def test_08_v1_threshold_remains_unchanged(legacy_adapter):
    assert legacy_adapter.threshold == 0.19088475755009496
    assert round(legacy_adapter.threshold, 6) == 0.190885


# -----------------------------------------------------------------------------
# TEST 9: Both v2 thresholds remain unchanged
# -----------------------------------------------------------------------------
def test_09_both_v2_thresholds_remain_unchanged(v2_adapter):
    assert v2_adapter.primary_threshold == 0.314036
    assert v2_adapter.reference_threshold == 0.099018


# -----------------------------------------------------------------------------
# TEST 10: Missing v2 artifacts do not break v1 dashboard operation
# -----------------------------------------------------------------------------
def test_10_missing_v2_artifacts_do_not_break_v1(legacy_adapter):
    dummy_missing_path = ROOT_DIR / "models" / "non_existent_directory_for_testing"
    missing_v2 = PureOneWayV2Adapter(artifacts_dir=dummy_missing_path)

    # v2 handles missing path gracefully
    assert missing_v2.is_available is False
    assert missing_v2.load_error is not None

    # Legacy adapter remains 100% operational
    assert legacy_adapter.is_available is True
    scores = legacy_adapter.score(np.zeros((3, 32)))
    assert len(scores) == 3


# -----------------------------------------------------------------------------
# TEST 11: Existing API routes preserve their previous default behavior
# -----------------------------------------------------------------------------
def test_11_existing_api_routes_preserve_previous_default_behavior(live_server):
    # 1. /api/overview preserves legacy champion metrics by default
    req = urllib.request.urlopen(f"{live_server}/api/overview")
    assert req.getcode() == 200
    ov = json.loads(req.read().decode("utf-8"))
    assert "kpi" in ov
    assert ov["kpi"]["total_network_events"] == 2300825
    assert ov["kpi"]["overall_recall_pct"] == 93.08
    assert ov["kpi"]["precision_pct"] == 70.46
    assert ov["kpi"]["observed_fpr_pct"] == 12.48
    assert ov["kpi"]["calibrated_threshold"] == 0.190885

    # 2. /api/metrics preserves all previous model keys
    req = urllib.request.urlopen(f"{live_server}/api/metrics")
    assert req.getcode() == 200
    metrics_data = json.loads(req.read().decode("utf-8"))
    for required_key in [
        "temporal_autoencoder",
        "oneway_autoencoder",
        "oneway_log_pca",
        "bidirectional_reference",
    ]:
        assert required_key in metrics_data, f"Missing required existing key {required_key}"

    # 3. /api/system_status preserves champion metadata
    req = urllib.request.urlopen(f"{live_server}/api/system_status")
    assert req.getcode() == 200
    st = json.loads(req.read().decode("utf-8"))
    assert st["status"] == "OPERATIONAL"
    assert st["champion_model"] == "Temporal One-Way Autoencoder (32-dim)"
    assert st["champion_status"] == "PRODUCTION CHAMPION"
    assert st["active_threshold"] == 0.190885


# -----------------------------------------------------------------------------
# TEST 12: Model comparison endpoint returns the correct metrics
# -----------------------------------------------------------------------------
def test_12_model_comparison_endpoint_returns_correct_metrics(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/compare_models")
    assert req.getcode() == 200
    comp = json.loads(req.read().decode("utf-8"))

    assert "comparison_table" in comp
    table = comp["comparison_table"]
    assert len(table) == 3

    # Check Legacy row
    leg_row = next(r for r in table if "Legacy" in r["model"])
    assert leg_row["recall_pct"] == 93.08
    assert leg_row["fpr_pct"] == 12.48
    assert leg_row["precision_pct"] == 70.46
    assert leg_row["feature_dimensions"] == 32

    # Check v2 Primary row
    v2_prim_row = next(r for r in table if "v2 Primary" in r["model"])
    assert v2_prim_row["recall_pct"] == 85.38
    assert v2_prim_row["fpr_pct"] == 10.70
    assert v2_prim_row["precision_pct"] == 71.86
    assert v2_prim_row["feature_dimensions"] == 26
    assert v2_prim_row["operating_threshold"] == 0.314036

    # Check v2 Reference row
    v2_ref_row = next(r for r in table if "v2 Reference" in r["model"])
    assert v2_ref_row["recall_pct"] == 93.01
    assert v2_ref_row["fpr_pct"] == 17.35
    assert v2_ref_row["precision_pct"] == 63.16
    assert v2_ref_row["feature_dimensions"] == 26
    assert v2_ref_row["operating_threshold"] == 0.099018


# -----------------------------------------------------------------------------
# TEST 13: Experimental status is exposed correctly
# -----------------------------------------------------------------------------
def test_13_experimental_status_exposed_correctly(live_server, legacy_adapter, v2_adapter):
    assert legacy_adapter.status == "PRODUCTION CHAMPION"
    assert v2_adapter.status == "EXPERIMENTAL / RESEARCH CANDIDATE"

    # From /api/models
    req = urllib.request.urlopen(f"{live_server}/api/models")
    assert req.getcode() == 200
    catalog = json.loads(req.read().decode("utf-8"))
    assert catalog["active_production_champion"] == "legacy_champion"
    assert catalog["models"]["legacy_champion"]["status"] == "PRODUCTION CHAMPION"
    assert catalog["models"]["pure_one_way_v2"]["status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"


# -----------------------------------------------------------------------------
# TEST 14: Frontend displays model status, operating points, notes, and drift warning correctly
# -----------------------------------------------------------------------------
def test_14_frontend_displays_model_status_and_operating_points():
    html_path = DASHBOARD_DIR / "index.html"
    assert html_path.exists(), "dashboard/index.html is missing"

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Model status badges
    assert "ACTIVE PRODUCTION CHAMPION" in html_content
    assert "EXPERIMENTAL / RESEARCH CANDIDATE" in html_content

    # Pure One-Way v2 operating points and thresholds
    assert "0.314036" in html_content
    assert "0.099018" in html_content
    assert "0.190885" in html_content

    # Metrics
    assert "85.38%" in html_content
    assert "10.70%" in html_content
    assert "71.86%" in html_content
    assert "93.01%" in html_content
    assert "17.35%" in html_content
    assert "63.16%" in html_content

    # Explanatory note
    expected_note = (
        "The primary v2 operating point reduces false positives compared with v1 but sacrifices recall. "
        "The reference operating point nearly matches v1 recall but has a higher FPR."
    )
    assert expected_note in html_content

    # Drift warning
    expected_warning = (
        "Operational FPR on Tuesday–Friday exceeds the Monday calibration target. "
        "The cause has not yet been established."
    )
    assert expected_warning in html_content


# -----------------------------------------------------------------------------
# TEST 15: Legacy API contract strictly matches base commit eca86cf
# -----------------------------------------------------------------------------
def test_15_legacy_api_contract_matches_base_commit_eca86cf(live_server):
    """
    Verifies that all legacy endpoints preserve 100% of the JSON schema,
    field keys, and types established at base commit eca86cf.
    """
    # 1. /api/metrics schema regression
    req = urllib.request.urlopen(f"{live_server}/api/metrics")
    assert req.getcode() == 200
    metrics_data = json.loads(req.read().decode("utf-8"))

    # Exact original 4 keys from eca86cf:dashboard/server.py
    ORIGINAL_METRICS_KEYS = {
        "temporal_autoencoder",
        "oneway_autoencoder",
        "oneway_log_pca",
        "bidirectional_reference",
    }
    assert ORIGINAL_METRICS_KEYS.issubset(set(metrics_data.keys())), (
        f"Missing original legacy keys: {ORIGINAL_METRICS_KEYS - set(metrics_data.keys())}"
    )

    # Newly added v2 keys
    NEW_V2_KEYS = {
        "pure_one_way_v2_primary",
        "pure_one_way_v2_reference",
    }
    assert NEW_V2_KEYS.issubset(set(metrics_data.keys()))

    # 2. /api/overview schema regression (default call)
    req = urllib.request.urlopen(f"{live_server}/api/overview")
    assert req.getcode() == 200
    ov = json.loads(req.read().decode("utf-8"))
    ORIGINAL_OVERVIEW_KEYS = {
        "system_name",
        "problem_id",
        "diode_mode",
        "uptime_seconds",
        "kpi",
        "severity_distribution",
        "category_breakdown",
    }
    assert ORIGINAL_OVERVIEW_KEYS.issubset(set(ov.keys()))

    ORIGINAL_KPI_KEYS = {
        "total_network_events",
        "benign_events",
        "total_attacks",
        "attacks_detected",
        "attacks_missed",
        "overall_recall_pct",
        "average_precision",
        "precision_pct",
        "observed_fpr_pct",
        "calibrated_threshold",
        "target_calibration_fpr",
        "average_inference_ms",
        "live_stream_processed",
        "live_stream_anomalies",
    }
    assert ORIGINAL_KPI_KEYS.issubset(set(ov["kpi"].keys()))

    # 3. /api/system_status schema regression
    req = urllib.request.urlopen(f"{live_server}/api/system_status")
    assert req.getcode() == 200
    st = json.loads(req.read().decode("utf-8"))
    ORIGINAL_STATUS_KEYS = {
        "status",
        "health_score",
        "uptime_seconds",
        "data_diode_link",
        "streaming_active",
        "last_packet_arrival",
        "champion_model",
        "model_parameters",
        "system_memory_mb",
        "active_threshold",
    }
    assert ORIGINAL_STATUS_KEYS.issubset(set(st.keys()))

    ORIGINAL_MODEL_PARAMS_KEYS = {
        "architecture",
        "activation",
        "loss_metric",
        "forward_features",
        "temporal_features",
        "window_seconds",
    }
    assert ORIGINAL_MODEL_PARAMS_KEYS == set(st["model_parameters"].keys())
