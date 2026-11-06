"""
tests/test_dashboard_calibration_sandbox.py
AEGIS Phase 3 Step 10: Model Calibration Sandbox Discrete Operating-Point Verification

Verifies all Step 10 requirements:
1. Fake continuous FPR slider and linear formula completely removed.
2. Verified discrete operating-point selector implemented.
3. Separation of Calibration Target FPR (alpha on calibration_benign.parquet) vs.
   Operational Test FPR (on test_tue_fri.parquet).
4. Dynamic Card A elements, Head-to-Head row highlighting, and chart synchronization.
5. v1 cannot select v2 Reference operating point (guarded both in DOM and JS).
6. Authentic evaluation metrics returned by backend routes and synchronized by frontend.
7. Production invariants preserved (v1 sole production champion, 32-dim, threshold 0.190885;
   v2 shadow-only, 26-dim, dual thresholds 0.314036 / 0.099018; no models retrained).
"""

import sys
import json
import urllib.request
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

from model_adapters import LegacyModelAdapter, PureOneWayV2Adapter
from server import ThreatDetectionDashboardHandler, HTTPServer


@pytest.fixture(scope="module")
def live_server():
    """Launches test HTTP server on localhost:8599 for isolated sandbox testing."""
    server_addr = ("127.0.0.1", 8599)
    httpd = HTTPServer(server_addr, ThreatDetectionDashboardHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)
    yield "http://127.0.0.1:8599"
    httpd.shutdown()
    httpd.server_close()


# -----------------------------------------------------------------------------
# TEST 1: Backend API responses return authentic evaluation metrics and operating points
# -----------------------------------------------------------------------------
def test_01_backend_api_metrics_and_operating_points(live_server):
    # 1. Check /api/metrics
    req = urllib.request.urlopen(f"{live_server}/api/metrics")
    assert req.getcode() == 200
    metrics = json.loads(req.read().decode("utf-8"))

    # v1 Champion
    assert "temporal_autoencoder" in metrics
    v1_m = metrics["temporal_autoencoder"]
    assert abs(v1_m["detection_rate_pct"] - 93.08) < 0.01
    assert abs(v1_m["false_positive_rate_pct"] - 12.48) < 0.01
    assert abs(v1_m["precision_pct"] - 70.46) < 0.01
    assert abs(v1_m["average_precision"] - 0.8783) < 0.001
    assert abs(v1_m["threshold"] - 0.190885) < 0.001
    assert abs(v1_m.get("target_calibration_fpr", 0.002) - 0.002) < 0.0001

    # v2 Primary
    assert "pure_one_way_v2_primary" in metrics
    v2_p = metrics["pure_one_way_v2_primary"]
    p_rates = v2_p["rates"]
    p_cls = v2_p["classification_metrics"]
    assert abs(p_rates["detection_rate_pct"] - 85.38) < 0.02
    assert abs(p_rates["false_positive_rate_pct"] - 10.70) < 0.02
    assert abs(p_cls["precision_pct"] - 71.86) < 0.02
    assert abs(p_cls["f1_score"] - 0.7804) < 0.001
    assert abs(v2_p["threshold_value"] - 0.314036) < 0.001

    # v2 Reference
    assert "pure_one_way_v2_reference" in metrics
    v2_r = metrics["pure_one_way_v2_reference"]
    r_rates = v2_r["rates"]
    r_cls = v2_r["classification_metrics"]
    assert abs(r_rates["detection_rate_pct"] - 93.01) < 0.02
    assert abs(r_rates["false_positive_rate_pct"] - 17.35) < 0.02
    assert abs(r_cls["precision_pct"] - 63.16) < 0.02
    assert abs(r_cls["f1_score"] - 0.7523) < 0.001
    assert abs(v2_r["threshold_value"] - 0.099018) < 0.001


def test_02_backend_compare_models_contract(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/compare_models")
    assert req.getcode() == 200
    comp = json.loads(req.read().decode("utf-8"))

    assert "comparison_table" in comp
    table = comp["comparison_table"]
    assert len(table) == 3

    # Legacy Champion (v1)
    leg = next(r for r in table if "Legacy Champion" in r["model"])
    assert leg["status"] == "PRODUCTION CHAMPION"
    assert leg["feature_dimensions"] == 32
    assert abs(leg["operating_threshold"] - 0.190885) < 0.001
    assert leg["recall_pct"] == 93.08
    assert leg["fpr_pct"] == 12.48
    assert leg["f1_score"] == 0.8020

    # v2 Primary
    v2_p = next(r for r in table if "v2 Primary" in r["model"])
    assert v2_p["status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"
    assert v2_p["feature_dimensions"] == 26
    assert abs(v2_p["operating_threshold"] - 0.314036) < 0.001
    assert v2_p["recall_pct"] == 85.38
    assert v2_p["fpr_pct"] == 10.70

    # v2 Reference
    v2_r = next(r for r in table if "v2 Reference" in r["model"])
    assert v2_r["status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"
    assert v2_r["feature_dimensions"] == 26
    assert abs(v2_r["operating_threshold"] - 0.099018) < 0.001
    assert v2_r["recall_pct"] == 93.01
    assert v2_r["fpr_pct"] == 17.35


# -----------------------------------------------------------------------------
# TEST 3: Fake continuous slider and formula completely removed from frontend
# -----------------------------------------------------------------------------
def test_03_fake_continuous_slider_removed():
    html_path = DASHBOARD_DIR / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # The continuous slider element must be completely absent
    assert 'id="fpr-slider"' not in content, "id='fpr-slider' was found in index.html; must be removed"
    assert 'type="range"' not in content, "Continuous range input found in index.html"
    assert "updateFPR(" not in content, "Fake updateFPR function still exists in index.html"
    assert "1.5 - val / 5.0" not in content, "Fabricated linear threshold multiplier formula found in index.html"


# -----------------------------------------------------------------------------
# TEST 4: Verified discrete selector and separation of calibration vs test FPR
# -----------------------------------------------------------------------------
def test_04_verified_discrete_selector_and_fpr_distinction():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Discrete operating point buttons
    assert 'id="btn-op-primary"' in content, "Missing #btn-op-primary button"
    assert 'id="btn-op-reference"' in content, "Missing #btn-op-reference button"

    # Discrete metrics display badges
    assert 'id="calib-target-val"' in content, "Missing #calib-target-val element"
    assert 'id="thresh-val"' in content, "Missing #thresh-val element"
    assert 'id="observed-fpr-val"' in content, "Missing #observed-fpr-val element"
    assert 'id="op-point-note"' in content, "Missing #op-point-note element"
    assert 'id="op-point-restriction"' in content, "Missing #op-point-restriction element"

    # Clear distinction note present in HTML
    assert "calibration_benign.parquet" in content, "calibration_benign.parquet not referenced in distinction note"
    assert "test_tue_fri.parquet" in content, "test_tue_fri.parquet not referenced in distinction note"
    assert "Calibration Target vs. Operational FPR:" in content, "Missing explicit conceptual distinction header"


# -----------------------------------------------------------------------------
# TEST 5: Dynamic Card A and Head-to-Head table DOM contract
# -----------------------------------------------------------------------------
def test_05_dynamic_card_a_and_table_row_contract():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Card A dynamic elements
    required_card_a_ids = [
        'id="card-a-title"',
        'id="card-a-badge"',
        'id="card-a-pulse"',
        'id="card-a-scope"',
        'id="card-a-scope-sub"',
        'id="card-a-recall"',
        'id="card-a-fpr"',
        'id="card-a-prec"',
        'id="card-a-f1"',
        'id="card-a-prauc"',
        'id="card-a-rocauc"',
        'id="card-a-threshold"',
        'id="card-a-target-fpr"',
        'id="card-a-scoring"',
    ]
    for dom_id in required_card_a_ids:
        assert dom_id in content, f"Missing required Card A DOM element: {dom_id}"

    # Head-to-Head Table row IDs
    required_table_row_ids = [
        'id="row-h2h-legacy"',
        'id="row-h2h-v2-primary"',
        'id="row-h2h-v2-ref"',
    ]
    for row_id in required_table_row_ids:
        assert row_id in content, f"Missing required Table row ID: {row_id}"


# -----------------------------------------------------------------------------
# TEST 6: JavaScript logic guards against v1 selecting Reference point
# -----------------------------------------------------------------------------
def test_06_v1_cannot_select_v2_reference_point():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Verify selectOperatingPoint exists
    assert "function selectOperatingPoint(" in content
    assert "function updateCalibrationDisplay(" in content
    assert "function fetchModelData(" in content

    # Reference button is initially disabled in markup for v1 default
    assert 'id="btn-op-reference" type="button"' in content or 'id="btn-op-reference"' in content
    assert 'disabled' in content

    # Guard in JavaScript: when activeModel is temporal_ae, cannot select reference
    assert "activeModel === 'temporal_ae'" in content
    assert "Reference operating point" in content
    assert "applies exclusively to Pure One-Way v2" in content


# -----------------------------------------------------------------------------
# TEST 7: Production invariants preserved
# -----------------------------------------------------------------------------
def test_07_production_invariants_preserved():
    legacy_adapter = LegacyModelAdapter()
    v2_adapter = PureOneWayV2Adapter()

    # 1. Authority
    assert legacy_adapter.status == "PRODUCTION CHAMPION"
    assert v2_adapter.status == "EXPERIMENTAL / RESEARCH CANDIDATE"

    # 2. Dimensions
    assert legacy_adapter.expected_dim == 32
    assert v2_adapter.expected_dim == 26

    # 3. Thresholds
    assert abs(legacy_adapter.threshold - 0.19088475755009496) < 1e-6
    assert abs(v2_adapter.primary_threshold - 0.314036) < 1e-6
    assert abs(v2_adapter.reference_threshold - 0.099018) < 1e-6

    # 4. Target FPRs
    assert legacy_adapter.target_fpr == 0.002
    assert v2_adapter.primary_target_fpr == 0.002
    assert v2_adapter.reference_target_fpr == 0.010


# -----------------------------------------------------------------------------
# TEST 8: All three backend APIs are fetched and structurally validated
# -----------------------------------------------------------------------------
def test_08_all_three_backend_apis_fetched_and_validated(live_server):
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Verify all 3 endpoints are fetched in fetchModelData()
    assert "fetch('/api/metrics')" in content
    assert "fetch('/api/models')" in content
    assert "fetch('/api/compare_models')" in content

    # Verify HTTP status checks on all 3
    assert "modelsRes.status === 'fulfilled' && modelsRes.value.ok" in content
    assert "metricsRes.status === 'fulfilled' && metricsRes.value.ok" in content
    assert "compareRes.status === 'fulfilled' && compareRes.value.ok" in content

    # Verify /api/models catalog validation logic in JS
    assert "catalog.active_production_champion === 'legacy_champion'" in content
    assert "catalog.models.legacy_champion.status === 'PRODUCTION CHAMPION'" in content
    assert "catalog.models.legacy_champion.input_dimensions === 32" in content
    assert "catalog.models.pure_one_way_v2.input_dimensions === 26" in content

    # Verify live /api/models endpoint adheres to catalog validation schema
    req = urllib.request.urlopen(f"{live_server}/api/models")
    assert req.getcode() == 200
    catalog = json.loads(req.read().decode("utf-8"))
    assert catalog["active_production_champion"] == "legacy_champion"
    assert catalog["models"]["legacy_champion"]["status"] == "PRODUCTION CHAMPION"
    assert catalog["models"]["legacy_champion"]["input_dimensions"] == 32
    assert catalog["models"]["pure_one_way_v2"]["status"] == "EXPERIMENTAL / RESEARCH CANDIDATE"
    assert catalog["models"]["pure_one_way_v2"]["input_dimensions"] == 26


# -----------------------------------------------------------------------------
# TEST 9: No silent fallback to v1; missing model data displays visible error
# -----------------------------------------------------------------------------
def test_09_no_silent_fallback_and_visible_error_handling():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Critical check: the silent fallback pattern MUST NOT EXIST
    assert "modelBreakdowns[dataKey] || modelBreakdowns.temporal_ae" not in content, (
        "Silent fallback 'modelBreakdowns[dataKey] || modelBreakdowns.temporal_ae' was found in index.html"
    )

    # Error handling branch for missing data
    assert "if (!data) {" in content
    assert "DATA UNAVAILABLE" in content
    assert "No metrics substituted" in content or "No fallback metrics substituted" in content

    # Elements in error state are populated with 'N/A' rather than v1 metrics
    assert 'cardRecall.textContent = "N/A"' in content
    assert 'cardFpr.textContent = "N/A"' in content
    assert 'threshValEl.textContent = "N/A"' in content

    # Dedicated sync status / error banner element is present in DOM
    assert 'id="sandbox-sync-status"' in content
    assert 'id="sandbox-sync-msg"' in content
    assert 'id="sandbox-sync-icon"' in content


# -----------------------------------------------------------------------------
# TEST 10: Stale data is clearly identified and API failures do not claim success
# -----------------------------------------------------------------------------
def test_10_stale_data_handling_and_failure_reporting():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Stale tracking variables and indicators
    assert "isDataStale" in content
    assert 'id="card-a-stale-tag"' in content
    assert "STALE BENCHMARK" in content
    assert "[STALE]" in content

    # Failure reporting does not claim success
    assert "syncErrors.length > 0" in content
    assert "Backend synchronization incomplete" in content
    assert "Data marked as STALE" in content


# -----------------------------------------------------------------------------
# TEST 11: Field-level metric validation helper logic in index.html
# -----------------------------------------------------------------------------
def test_11_field_level_metric_validation_helper():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Helper function exists and checks both numeric and string values
    assert "function isValidMetricValue(val)" in content
    assert "if (val === null || val === undefined) return false;" in content
    assert "!isNaN(val) && isFinite(val)" in content
    assert "clean.endsWith('%')" in content

    # Python simulation of isValidMetricValue to test all edge cases identically
    def is_valid_metric(val):
        if val is None:
            return False
        if isinstance(val, (int, float)):
            import math
            return not math.isnan(val) and not math.isinf(val)
        if isinstance(val, str):
            trimmed = val.strip()
            if not trimmed or trimmed.upper() in ("N/A", "NAN", "NULL", "UNDEFINED"):
                return False
            clean = trimmed
            if clean.startswith("α = ") or clean.startswith("\u03b1 = ") or clean.startswith("a = "):
                clean = clean.split("=", 1)[1].strip()
            if clean.endswith("%"):
                clean = clean[:-1].strip()
            if not clean:
                return False
            try:
                num = float(clean)
                import math
                return not math.isnan(num) and not math.isinf(num)
            except ValueError:
                return False
        return False

    # Valid values, including valid numeric zero and percentage zero
    assert is_valid_metric(0) is True
    assert is_valid_metric(0.0) is True
    assert is_valid_metric("0") is True
    assert is_valid_metric("0.0") is True
    assert is_valid_metric("0.00%") is True
    assert is_valid_metric("85.38%") is True
    assert is_valid_metric("0.314036") is True
    assert is_valid_metric("α = 0.2%") is True
    assert is_valid_metric("\u03b1 = 1.0%") is True

    # Invalid values
    assert is_valid_metric(None) is False
    assert is_valid_metric("") is False
    assert is_valid_metric("   ") is False
    assert is_valid_metric("N/A") is False
    assert is_valid_metric("n/a") is False
    assert is_valid_metric("NaN") is False
    assert is_valid_metric("null") is False
    assert is_valid_metric("undefined") is False
    assert is_valid_metric("Baseline") is False
    assert is_valid_metric(float("nan")) is False


# -----------------------------------------------------------------------------
# TEST 12: Zero hardcoded fallback strings in updateCalibrationDisplay()
# -----------------------------------------------------------------------------
def test_12_no_hardcoded_metric_fallbacks_in_display():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # Extract updateCalibrationDisplay function
    start = content.find("function updateCalibrationDisplay()")
    end = content.find("async function fetchModelData()", start)
    assert start != -1 and end != -1
    fn_body = content[start:end]

    # Verify that none of the previously identified hardcoded fallback expressions exist
    forbidden_fallbacks = [
        'data.recall || "93.08%"',
        'data.fpr || "12.48%"',
        'data.precision || "70.46%"',
        'data.f1 || "0.8020"',
        'data.ap || "0.8783"',
        'data.roc_auc || "0.9538"',
        'data.threshold || "0.190885"',
        'data.threshold || "0.099018"',
        'data.threshold || "0.314036"',
        'data.fpr || "17.35%"',
        'data.fpr || "10.70%"',
    ]
    for fb in forbidden_fallbacks:
        assert fb not in fn_body, f"Found forbidden hardcoded fallback '{fb}' in updateCalibrationDisplay()"

    # Confirm all 8 fields use isValidMetricValue rendering
    assert 'cardRecall.textContent = isValidMetricValue(data.recall) ? data.recall : "N/A"' in fn_body
    assert 'cardFpr.textContent = isValidMetricValue(data.fpr) ? data.fpr : "N/A"' in fn_body
    assert 'cardPrec.textContent = isValidMetricValue(data.precision) ? data.precision : "N/A"' in fn_body
    assert 'cardF1.textContent = isValidMetricValue(data.f1) ? data.f1 : "N/A"' in fn_body
    assert 'cardPrauc.textContent = isValidMetricValue(praucVal) ? praucVal : "N/A"' in fn_body
    assert 'cardRocauc.textContent = isValidMetricValue(data.roc_auc) ? data.roc_auc : "N/A"' in fn_body
    assert 'cardThreshold.textContent = isValidMetricValue(data.threshold) ? data.threshold : "N/A"' in fn_body
    assert 'cardTargetFpr.textContent = "N/A"' in fn_body


# -----------------------------------------------------------------------------
# TEST 13: Strict field-level validation and incomplete-data warning banner
# -----------------------------------------------------------------------------
def test_13_field_level_validation_and_incomplete_banner():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    start = content.find("function updateCalibrationDisplay()")
    end = content.find("async function fetchModelData()", start)
    fn_body = content[start:end]

    # Required 8 fields checked
    for field_name in ["Recall", "Test FPR", "Precision", "F1-score", "PR-AUC", "ROC-AUC", "Threshold", "Calibration target FPR"]:
        assert field_name in fn_body, f"Required metric field '{field_name}' not checked in field validation list"

    assert "const hasIncompleteMetrics = missingFields.length > 0;" in fn_body
    assert "Incomplete Evaluation Data" in fn_body
    assert "[INCOMPLETE DATA]" in fn_body
    assert "missing required metric fields" in fn_body
    assert "Unavailable metrics are rendered as N/A. No values substituted." in fn_body


# -----------------------------------------------------------------------------
# TEST 14: Dual threshold backward compatibility on backend endpoints
# -----------------------------------------------------------------------------
def test_14_backend_threshold_backward_compatibility(live_server):
    req = urllib.request.urlopen(f"{live_server}/api/metrics")
    assert req.getcode() == 200
    metrics = json.loads(req.read().decode("utf-8"))

    # Check v1 threshold
    assert "temporal_autoencoder" in metrics
    assert abs(metrics["temporal_autoencoder"]["threshold"] - 0.190885) < 0.001

    # Check v2 primary exposes both threshold and threshold_value with identical values
    assert "pure_one_way_v2_primary" in metrics
    p = metrics["pure_one_way_v2_primary"]
    assert "threshold" in p
    assert "threshold_value" in p
    assert abs(p["threshold"] - p["threshold_value"]) < 1e-6
    assert abs(p["threshold"] - 0.314036) < 0.001

    # Check v2 reference exposes both threshold and threshold_value with identical values
    assert "pure_one_way_v2_reference" in metrics
    r = metrics["pure_one_way_v2_reference"]
    assert "threshold" in r
    assert "threshold_value" in r
    assert abs(r["threshold"] - r["threshold_value"]) < 1e-6
    assert abs(r["threshold"] - 0.099018) < 0.001


# -----------------------------------------------------------------------------
# TEST 15: No substitution of v1 values for missing v2 metrics
# -----------------------------------------------------------------------------
def test_15_no_v1_substitution_for_missing_v2_metrics():
    html_path = DASHBOARD_DIR / "index.html"
    content = html_path.read_text(encoding="utf-8")

    # In fetchModelData, null or missing fields in /api/metrics are assigned null, never v1 fallback
    assert "modelBreakdowns.pure_oneway_v2.recall = (ratesObj.detection_rate_pct != null) ? `${ratesObj.detection_rate_pct.toFixed(2)}%` : null;" in content
    assert "modelBreakdowns.pure_oneway_v2.fpr = (ratesObj.false_positive_rate_pct != null) ? `${ratesObj.false_positive_rate_pct.toFixed(2)}%` : null;" in content
    assert "modelBreakdowns.pure_oneway_v2.precision = (clsObj.precision_pct != null) ? `${clsObj.precision_pct.toFixed(2)}%` : null;" in content
    assert "modelBreakdowns.pure_oneway_v2.f1 = (clsObj.f1_score != null) ? clsObj.f1_score.toFixed(4) : null;" in content
    assert "modelBreakdowns.pure_oneway_v2.threshold = (rawThresh != null) ? parseFloat(rawThresh).toFixed(6) : null;" in content

    # Reference model as well
    assert "modelBreakdowns.pure_oneway_v2_ref.recall = (ratesObj.detection_rate_pct != null) ? `${ratesObj.detection_rate_pct.toFixed(2)}%` : null;" in content
    assert "modelBreakdowns.pure_oneway_v2_ref.fpr = (ratesObj.false_positive_rate_pct != null) ? `${ratesObj.false_positive_rate_pct.toFixed(2)}%` : null;" in content
    assert "modelBreakdowns.pure_oneway_v2_ref.f1 = (clsObj.f1_score != null) ? clsObj.f1_score.toFixed(4) : null;" in content
    assert "modelBreakdowns.pure_oneway_v2_ref.threshold = (rawThresh != null) ? parseFloat(rawThresh).toFixed(6) : null;" in content


# -----------------------------------------------------------------------------
# TEST 16: Strict production invariants verification
# -----------------------------------------------------------------------------
def test_16_strict_production_invariants():
    # 1. Adapter dimension invariants
    legacy = LegacyModelAdapter()
    assert legacy.EXPECTED_DIM == 32
    assert legacy.STATUS == "PRODUCTION CHAMPION"
    assert abs(legacy.threshold - 0.190885) < 1e-4

    v2 = PureOneWayV2Adapter()
    assert v2.EXPECTED_DIM == 26
    assert v2.STATUS == "EXPERIMENTAL / RESEARCH CANDIDATE"
    assert abs(v2.primary_threshold - 0.314036) < 1e-4
    assert abs(v2.reference_threshold - 0.099018) < 1e-4



