# AEGIS Phase 3 — Stage 2: Dual-Model Dashboard Integration Report
**Project:** AEGIS – Label-Free Threat Detection in One-Way Network Traffic
**SIH Problem ID:** SIH26145
**Current Branch:** `feat/phase-2-pure-one-way-model`
**Base Commit:** `eca86cf`
**Implementation Date:** 2026-11-04
**Author:** Principal Machine Learning Engineer, Systems Architect, and Full-Stack QA Engineer

---

## 1. Executive Summary & Verification of Mandates

In Phase 3 Stage 2, the **Pure One-Way 26-Feature Autoencoder (v2)** was integrated into the AEGIS dashboard alongside the existing **Legacy Champion (v1)**.

The implementation strictly honors all architectural safety and isolation constraints:
- **Legacy Champion Preservation:** The Legacy Champion (32 features, `32 -> 24 -> 8 -> 24 -> 32`) remains the active production champion. Its model weights, scalers, residual weights, threshold (`0.190885`), metrics, and endpoints remain completely unchanged and operational.
- **Experimental Status Enforcement:** The v2 model is explicitly marked and surfaced across the backend API and frontend UI as **`[EXPERIMENTAL / RESEARCH CANDIDATE]`**. It does not supersede v1 in default views or operational decisions.
- **Strict Dimensional & Pipeline Isolation:** Isolated adapters (`LegacyModelAdapter` and `PureOneWayV2Adapter`) encapsulate feature requirements. Input matrices are strictly guarded: passing 26 features to v1 or 32 features to v2 immediately raises a deterministic `ValueError` with clear guidance.
- **Operational Drift Transparency:** The exact required diagnostic wording is surfaced in both the API and the frontend comparison view:

  > *"Operational FPR on Tuesday–Friday exceeds the Monday calibration target. The cause has not yet been established."*

  *(No unproven speculative claims attributing drift to diurnal loads or active browsing are made.)*
- **Backward Compatibility:** All existing endpoints (`/api/overview`, `/api/metrics`, `/api/system_status`, `/api/live_feed`, `/api/start_stream`) preserve 100% of their legacy response schemas, enabling existing downstream consumers to function without modification.
- **Comprehensive Automated Test Coverage:** 14 dedicated integration tests were implemented in `tests/test_dashboard_integration.py`, verifying model loading, dimensionality enforcement, threshold invariance, score invariance, missing artifact resilience, API schemas, and frontend rendering. All 14 integration tests passed, and the complete test suite (44/44 tests) passed with 0 failures and 0 warnings.

---

## 2. Inventory of File Changes

| File | Status | Description |
|---|---|---|
| `src/model_adapters.py` | **New** | Object-oriented adapter layer providing `BaseModelAdapter`, `LegacyModelAdapter`, and `PureOneWayV2Adapter` with strict dimensionality guards, independent scoring functions, and safe metadata generation. |
| `tests/test_dashboard_integration.py` | **New** | Comprehensive pytest test suite with 14 test functions validating dual-model loading, cross-model input rejection, score/threshold invariance, API backward compatibility, and frontend rendering integrity. |
| `dashboard/server.py` | **Modified** | Integrated model adapters, added `/api/models` and `/api/compare_models`, extended `/api/overview` with `?model=v2` query parameter support, extended `/api/metrics` with v2 operating points, and enhanced `/api/system_status` with champion and experimental metadata. |
| `dashboard/index.html` | **Modified** | Added Dual-Model Intelligence Cards (Champion v1 vs Pure One-Way v2 Primary & Reference), Head-to-Head Comparison Table, Model Trade-off Note, Operational Drift Warning, v2 Model Selector button in Calibration Sandbox, and v2 rows in the Leaderboard. |
| `reports/PHASE_3_STAGE_2_DASHBOARD_IMPLEMENTATION_REPORT.md` | **New** | This comprehensive engineering report documenting architecture, APIs, frontend changes, test execution, limitations, rollback, and recommendation. |

---

## 3. Architecture & Adapter Pattern

### 3.1 Architectural Flow Diagram

```
                               DUAL-MODEL DASHBOARD ARCHITECTURE

   Browser (dashboard/index.html)             dashboard/server.py             src/model_adapters.py
         |                                           |                                  |
         |---- GET /api/overview?model=v2 ---------->|                                  |
         |                                           |---- Query Adapter -------------->| PureOneWayV2Adapter
         |<--- JSON (v2 KPIs & Severities) ----------|                                  | (26 dims, unweighted MSE)
         |                                           |                                  |
         |---- GET /api/overview (default) --------->|                                  |
         |                                           |---- Query Adapter -------------->| LegacyModelAdapter
         |<--- JSON (v1 KPIs & Severities) ----------|                                  | (32 dims, weighted MSE)
         |                                           |                                  |
         |---- GET /api/models --------------------->|                                  |
         |<--- JSON (Catalog: v1 Champ, v2 Exp) -----|                                  |
         |                                           |                                  |
         |---- GET /api/compare_models ------------->|                                  |
         |<--- JSON (Head-to-head table + Drift) ----|                                  |
         |                                           |                                  |
         |---- GET /api/metrics -------------------->|                                  |
         |<--- JSON (Legacy 4 + v2 Primary & Ref) ---|                                  |
```

### 3.2 Adapter Implementation Details (`src/model_adapters.py`)

1. **`BaseModelAdapter` (Abstract Interface):**
   - Defines standard contract: `load()`, `score(X)`, `predict(X)`, `validate_input(X)`, `get_metadata()`.
   - Strips absolute local filesystem paths from all client-facing metadata dictionaries to prevent internal path leaks.

2. **`LegacyModelAdapter`:**
   - **Target Architecture:** `32 -> 24 -> 8 -> 24 -> 32` (`models/temporal_autoencoder.pkl`).
   - **Feature Specification:** `AUGMENTED_ONE_WAY_FEATURES` (32 dimensions).
   - **Preprocessing:** MinMax scaler (`models/scaler_temporal.pkl`).
   - **Scoring Function:** Inverse-variance residual weighted MSE:
     $$\text{Score}(x) = \frac{1}{d} \sum_{j=1}^{d} w_j (x_j - \hat{x}_j)^2$$
     using residual weights from `models/residual_weights.npy`.
   - **Threshold:** $\tau = 0.19088475755009496$ with decision rule $\text{Score} \ge \tau$.
   - **Status:** `"PRODUCTION CHAMPION"`.

3. **`PureOneWayV2Adapter`:**
   - **Target Architecture:** `26 -> 20 -> 8 -> 20 -> 26` (`models/pure_oneway_autoencoder.pkl`).
   - **Feature Specification:** `PURE_ONE_WAY_FEATURES` (26 dimensions).
   - **Preprocessing:** Robust scaler (`models/scaler_pure_oneway.pkl`).
   - **Scoring Function:** Unweighted Mean Squared Error (MSE):
     $$\text{Score}(x) = \frac{1}{d} \sum_{j=1}^{d} (x_j - \hat{x}_j)^2$$
   - **Dual Operating Point Thresholds:**
     - **Primary (99.8% Benign Target):** $\tau_{\text{primary}} = 0.314036$ (Operational Recall: 85.38%, FPR: 10.70%, F1: 0.7804).
     - **Reference (Legacy Recall-Matched):** $\tau_{\text{ref}} = 0.099018$ (Operational Recall: 93.01%, FPR: 17.35%, F1: 0.7523).
     - Decision rule: $\text{Score} > \tau$.
   - **Status:** `"EXPERIMENTAL / RESEARCH CANDIDATE"`.

---

## 4. Model Isolation & Safety Mechanisms

### 4.1 Strict Dimensionality Guards
Both adapters implement strict pre-inference input validation via `validate_input(X)`:
- `LegacyModelAdapter.validate_input(X)` requires `X.shape[1] == 32`. If 26 features are passed, it raises:
  `ValueError: Dimension mismatch for Legacy Champion v1: expected 32 features (AUGMENTED_ONE_WAY_FEATURES), but received 26. Feature pipelines must not be crossed.`
- `PureOneWayV2Adapter.validate_input(X)` requires `X.shape[1] == 26`. If 32 features are passed, it raises:
  `ValueError: Dimension mismatch for Pure One-Way v2: expected 26 features (PURE_ONE_WAY_FEATURES), but received 32. Feature pipelines must not be crossed.`

### 4.2 Threshold and Scoring Independence
- The Legacy adapter maintains its calibrated float64 threshold and residual weights independently.
- The v2 adapter maintains its two distinct operating points without residual weights.
- Neither model's scoring function or threshold evaluation interacts with or depends on the other.

### 4.3 Missing Artifact Resilience & Graceful Degradation
- In `dashboard/server.py`, both adapters are instantiated inside isolated `try-except` blocks.
- If the Pure One-Way v2 model artifact is missing or corrupted, `V2_ADAPTER` is set to `None`.
- The legacy dashboard remains 100% operational, the server starts normally, and endpoints fall back gracefully without crashing.

---

## 5. Backend REST API Extensions & Schemas

### 5.1 `GET /api/models` (New)
Returns the model catalog, highlighting the active champion and any experimental models:
```json
{
  "active_champion": "temporal_autoencoder",
  "models": {
    "temporal_autoencoder": {
      "model_name": "Legacy Temporal Autoencoder (v1)",
      "version": "v1.0-legacy-champion",
      "status": "PRODUCTION CHAMPION",
      "expected_dim": 32,
      "feature_spec": "AUGMENTED_ONE_WAY_FEATURES",
      "architecture": "32 -> 24 -> 8 -> 24 -> 32",
      "threshold": 0.19088475755009496,
      "scoring_method": "Inverse-Variance Weighted MSE"
    },
    "pure_one_way_v2": {
      "model_name": "Pure One-Way Autoencoder (v2)",
      "version": "v2.0-pure-oneway-candidate",
      "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
      "expected_dim": 26,
      "feature_spec": "PURE_ONE_WAY_FEATURES",
      "architecture": "26 -> 20 -> 8 -> 20 -> 26",
      "primary_threshold": 0.314036,
      "reference_threshold": 0.099018,
      "scoring_method": "Unweighted MSE"
    }
  }
}
```

### 5.2 `GET /api/compare_models` (New)
Provides head-to-head operational metrics and transparent drift diagnostics:
```json
{
  "legacy_champion": {
    "name": "Legacy Champion (v1)",
    "status": "PRODUCTION CHAMPION",
    "feature_dim": 32,
    "architecture": "32 -> 24 -> 8 -> 24 -> 32",
    "threshold": 0.190885,
    "recall": 0.9308,
    "fpr": 0.1248,
    "precision": 0.7046,
    "f1": 0.8020,
    "pr_auc": 0.8783,
    "roc_auc": 0.9538
  },
  "pure_one_way_v2_primary": {
    "name": "Pure One-Way v2 (Primary)",
    "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
    "feature_dim": 26,
    "architecture": "26 -> 20 -> 8 -> 20 -> 26",
    "threshold": 0.314036,
    "recall": 0.8538,
    "fpr": 0.1070,
    "precision": 0.7186,
    "f1": 0.7804,
    "pr_auc": 0.8687,
    "roc_auc": 0.9478
  },
  "pure_one_way_v2_reference": {
    "name": "Pure One-Way v2 (Recall-Matched Ref)",
    "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
    "feature_dim": 26,
    "architecture": "26 -> 20 -> 8 -> 20 -> 26",
    "threshold": 0.099018,
    "recall": 0.9301,
    "fpr": 0.1735,
    "precision": 0.6316,
    "f1": 0.7523,
    "pr_auc": 0.8687,
    "roc_auc": 0.9478
  },
  "drift_warning": "Operational FPR on Tuesday–Friday exceeds the Monday calibration target. The cause has not yet been established.",
  "trade_off_note": "The primary v2 operating point reduces false positives compared with v1 but sacrifices recall. The reference operating point nearly matches v1 recall but has a higher FPR.",
  "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
  "compliance_note": "Feature-level compliance only; distinct from validation on a physical one-way capture device.",
  "drift_warning": "Operational FPR on Tuesday–Friday exceeds the Monday calibration target. The cause has not yet been established."
}
```

### 5.3 `GET /api/overview?model=v2` (Extended)
- When called without arguments (`/api/overview`), returns the unaltered Legacy Champion overview data (2,300,825 total flows, 519,031 detected, recall 93.08%, FPR 12.48%, threshold 0.190885).
- When called with `?model=v2`, returns the Pure One-Way v2 primary operational overview (detected 476,125, recall 85.38%, FPR 10.70%, threshold 0.314036) along with severity breakdowns and model metadata (`model_version: "v2.0-pure-oneway-candidate"`, `model_status: "EXPERIMENTAL / RESEARCH CANDIDATE"`).

### 5.4 `GET /api/metrics` (Extended)
Preserves existing top-level keys (`temporal_autoencoder`, `oneway_autoencoder`, `oneway_log_pca`, `reduced_pca`) while adding:
- `pure_one_way_v2_primary`: v2 primary operating point (threshold 0.314036, recall 0.8538, FPR 10.70%, F1 0.7804, precision 71.86%, PR-AUC 0.8687, ROC-AUC 0.9478).
- `pure_one_way_v2_reference`: v2 recall-matched reference operating point (threshold 0.099018, recall 93.01%, FPR 17.35%, F1 0.7523, precision 63.16%, PR-AUC 0.8687, ROC-AUC 0.9478).

### 5.5 `GET /api/system_status` (Extended)
Preserves all legacy status fields (`link_status`, `data_source`, `active_threat_level`, `model_parameters`, `anomaly_threshold`, `uptime_seconds`) while adding:
- `champion_status`: `"ACTIVE PRODUCTION CHAMPION"`
- `experimental_model`:
  - `model_name`: `"Pure One-Way Autoencoder (v2)"`
  - `status`: `"EXPERIMENTAL / RESEARCH CANDIDATE"`
  - `feature_dim`: `26`
  - `primary_threshold`: `0.314036`

---

## 6. Frontend Enhancements (`dashboard/index.html`)

1. **Dual-Model Intelligence Section:**
   - Placed in the Model Management view above the existing calibration controls.
   - **Card A (Legacy Champion v1):** Cyan-themed card highlighting active production status, 32-dim pipeline, 93.08% recall, 12.48% FPR, F1 0.8020.
   - **Card B (Pure One-Way v2):** Amber-themed card clearly badged `EXPERIMENTAL / RESEARCH CANDIDATE`, displaying the 26-dim forward-feature-only scope, Primary operating point ($\tau=0.314036$, Recall 85.38%, FPR 10.70%), and Reference operating point ($\tau=0.099018$, Recall 93.01%, FPR 17.35%).
2. **Head-to-Head Comparison Table:**
   - Displays all 8 key metrics side-by-side across Legacy Champion v1, Pure One-Way v2 (Primary), and Pure One-Way v2 (Reference).
   - Features visual badges distinguishing production and experimental statuses.
3. **Model Trade-off Note & Scope Disclaimer:**
   - Clearly explains the forward-feature design scope: 26-dimensional forward-feature-only design; reverse-direction feature information excluded (feature-level compliance; distinct from validation on a physical one-way capture device).
   - Explains the trade-off between lower FPR (10.70% vs 12.48%) and recall (85.38% vs 93.08%).
4. **Mandatory Operational Drift Warning:**
   - High-visibility amber alert box containing the exact required text:
     *"Operational FPR on Tuesday–Friday exceeds the Monday calibration target. The cause has not yet been established."*
5. **Model Calibration Sandbox Switcher:**
   - Added `btn-pure-oneway-v2` button in the model switch ribbon with amber highlight styling.
   - Informational selector behavior: Updates the threshold display and comparative rate charts in the calibration sandbox; does NOT alter backend live streaming or production inference.
   - Triggers an informational toast when clicked:
     *"Viewing EXPERIMENTAL candidate: Pure One-Way Autoencoder (v2). Production Champion remains Legacy v1."*
6. **Consolidated Leaderboard Extension:**
   - Appended `pure_one_way_v2_primary` and `pure_one_way_v2_reference` rows to the leaderboard table.

---

## 7. Head-to-Head Operational Performance Comparison

Evaluated on the full 2,300,825 Tuesday–Friday CIC-IDS2017 operational dataset (1,743,179 Benign, 557,646 Attacks across 14 categories):

| Evaluation Metric | Legacy Champion (v1) | Pure One-Way v2 (Primary) | Pure One-Way v2 (Reference) |
|---|---|---|---|
| **Model Status** | `PRODUCTION CHAMPION` | `EXPERIMENTAL / RESEARCH CANDIDATE` | `EXPERIMENTAL / RESEARCH CANDIDATE` |
| **Feature Dimensions** | 32 (Augmented) | 26 (Pure Forward-Only) | 26 (Pure Forward-Only) |
| **Compliance Scope** | Non-compliant (32 dims, derived bi-flow fields) | **26-dim forward-only features** (reverse info excluded)* | **26-dim forward-only features** (reverse info excluded)* |
| **Operating Threshold ($\tau$)** | `0.190885` | `0.314036` | `0.099018` |
| **True Positives (TP)** | **519,031** | 476,125 | 518,647 |
| **False Positives (FP)** | 217,598 | **186,489** | 302,461 |
| **True Negatives (TN)** | 1,525,581 | **1,556,690** | 1,440,718 |
| **False Negatives (FN)** | **38,615** | 81,521 | 38,999 |
| **Confusion Matrix Total** | **2,300,825** | **2,300,825** | **2,300,825** |
| **Operational Recall** | **93.08%** (519,031 / 557,646) | 85.38% (476,125 / 557,646) | **93.01%** (518,647 / 557,646) |
| **Operational FPR** | 12.48% (217,598 / 1,743,179) | **10.70%** (186,489 / 1,743,179) | 17.35% (302,461 / 1,743,179) |
| **Operational Precision** | 70.46% | **71.86%** | 63.16% |
| **F1 Score** | **0.8020** | 0.7804 | 0.7523 |
| **PR-AUC** | **0.8783** | 0.8687 | 0.8687 |
| **ROC-AUC** | **0.9538** | 0.9478 | 0.9478 |

*\*Note: Feature-level compliance only; distinct from validation on a physical one-way capture device.*

---

## 8. Automated Test Execution & Verification

### 8.1 Dedicated Integration Tests (`tests/test_dashboard_integration.py`)
All 14 integration tests passed:

```
tests/test_dashboard_integration.py::test_01_v1_loads_successfully PASSED              [  7%]
tests/test_dashboard_integration.py::test_02_v2_loads_successfully PASSED              [ 14%]
tests/test_dashboard_integration.py::test_03_v1_accepts_expected_32_dim_input PASSED    [ 21%]
tests/test_dashboard_integration.py::test_04_v2_accepts_expected_26_dim_input PASSED    [ 28%]
tests/test_dashboard_integration.py::test_05_cross_model_dimensionality_misuse_rejected PASSED [ 35%]
tests/test_dashboard_integration.py::test_06_v1_scoring_remains_unchanged PASSED       [ 42%]
tests/test_dashboard_integration.py::test_07_v2_scoring_remains_unchanged PASSED       [ 50%]
tests/test_dashboard_integration.py::test_08_v1_threshold_remains_unchanged PASSED     [ 57%]
tests/test_dashboard_integration.py::test_09_both_v2_thresholds_remain_unchanged PASSED [ 64%]
tests/test_dashboard_integration.py::test_10_missing_v2_artifacts_do_not_break_v1 PASSED [ 71%]
tests/test_dashboard_integration.py::test_11_existing_api_routes_preserve_previous_default_behavior PASSED [ 78%]
tests/test_dashboard_integration.py::test_12_model_comparison_endpoint_returns_correct_metrics PASSED [ 85%]
tests/test_dashboard_integration.py::test_13_experimental_status_exposed_correctly PASSED [ 92%]
tests/test_dashboard_integration.py::test_14_frontend_displays_model_status_and_operating_points PASSED [100%]
```

### 8.2 Full Repository Regression Test Run
The entire AEGIS test suite was executed across all unit, feature, causality, training, evaluation, and dashboard integration tests:
- **Command:** `pytest -q`
- **Result:** **44 passed in 11.67s**
- **Failures:** 0
- **Warnings:** 0

---

## 9. Known Limitations

1. **Live Streaming Daemon Unchanged:**
   The background streaming simulator (`src/simulate_stream.py`) remains bound to Legacy Champion (v1). As required by the safety constraints, v2 was not connected to the real-time simulation stream in this stage.

2. **Operational Drift Root Cause Open:**
   The gap between Monday calibration FPR (0.20%) and Tuesday–Friday operational FPR (10.70% on v2 primary, 12.48% on v1) is explicitly disclosed. Its exact root cause remains under investigation and has not yet been established.

3. **Static Evaluation Feed:**
   The comparative metrics displayed in the dashboard represent static evaluation results from the full Tuesday–Friday benchmark dataset. Dynamic live side-by-side stream scoring is scheduled for a subsequent phase.

---

## 10. Rollback Procedure

If rollback of Phase 3 Stage 2 is ever required:
1. Revert modified dashboard files:
   ```bash
   git checkout HEAD -- dashboard/server.py dashboard/index.html
   ```
2. Remove newly created adapter and test files:
   ```bash
   rm src/model_adapters.py tests/test_dashboard_integration.py
   ```
3. Run `pytest -q` to confirm the test suite returns to the 30-test Phase 2 baseline.

---

## 11. Final Recommendation: GO

**Status:** **GO**

### Justification:
1. **Safety Compliance:** Legacy Champion v1 remains completely untouched, fully operational, and designated as the active production champion.
2. **Strict Isolation:** The adapter layer prevents any cross-contamination of feature vectors or scoring routines between 32-dimensional and 26-dimensional pipelines.
3. **Rigorous Verification:** 100% of integration tests (14/14) and full regression tests (44/44) pass cleanly.
4. **Transparent Governance:** The experimental status, trade-offs, and operational drift disclaimer are prominently displayed across the dashboard backend and frontend UI.
5. **Clean Git State:** No untracked build artifacts or unintentional modifications were introduced. The working directory is ready for review.
