# AEGIS — Step 10, 10.1 & 10.2: Model Calibration Sandbox Discrete Operating-Point, API Resilience & Strict Metric Validation Report

**Project:** AEGIS — Label-Free Threat Detection in One-Way Network Traffic  
**Problem ID:** SIH26145  
**Baseline Git HEAD:** `740ade7`  
**Branch:** `feat/phase-2-pure-one-way-model`  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Executive Summary

In Phase 3 Steps 10, 10.1, and 10.2, the Model Calibration Sandbox in `dashboard/index.html` and supporting backend endpoints in `dashboard/server.py` were corrected to:
1. Eliminate the fabricated continuous False Positive Rate (FPR) slider (`#fpr-slider`) and its synthetic linear formula (`baseThresh * (1.5 - val / 5.0)`).
2. Implement a verified discrete operating-point selector integrating directly with all three authentic backend evaluation routes (`/api/metrics`, `/api/models`, `/api/compare_models`).
3. Validate HTTP status and JSON schemas across all three endpoints asynchronously, consuming `/api/models` for catalog and active production champion validation without allowing its metadata to overwrite verified evaluation metrics.
4. Eliminate silent fallback to v1 metrics (`modelBreakdowns[dataKey] || modelBreakdowns.temporal_ae`). When data for a selected model is missing or invalid, the dashboard surfaces a clear visible error state, identifies the unavailable model, and refuses to substitute or fabricate metrics.
5. Track and visibly mark stale data (`[STALE]`) if an API request fails, preventing false claims of successful synchronization.
6. **Strict Field-Level Metric Validation (Step 10.2):** Implemented `isValidMetricValue()` to strictly validate all 8 required evaluation fields before rendering:
   - Recall
   - Test FPR
   - Precision
   - F1-score
   - PR-AUC
   - ROC-AUC
   - Threshold
   - Calibration target FPR
   If any required field is missing, null, nonnumeric, or invalid, the model is not displayed as fully synchronized, missing metrics display `"N/A"`, a visible warning banner (`#sandbox-sync-status`) names the missing fields, and no values are substituted from any other model.
7. **Complete Elimination of Hardcoded Fallbacks (Step 10.2):** Completely removed all hardcoded rendering fallbacks (`data.recall || "93.08%"`, `data.fpr || "12.48%"`, `data.precision || "70.46%"`, `data.f1 || "0.8020"`, `data.threshold || "0.190885"`, `data.threshold || "0.099018"`, `data.threshold || "0.314036"`, etc.). Valid numeric zero (e.g. `0`, `0.0`, `"0.00%"`) is correctly recognized as valid.
8. **Dual-Threshold Backward Compatibility (Step 10.2):** `dashboard/server.py` exposes both `threshold` and `threshold_value` with identical floating-point values across all `/api/metrics` code paths.

All production invariants remain strictly preserved: Legacy Temporal Autoencoder (v1) remains the sole production alert authority (32 dimensions, threshold $0.190885$), while Pure One-Way Autoencoder (v2) remains experimental and shadow-only (26 dimensions). Zero model retraining, zero weights modifications, and zero streaming changes were made.

---

## 2. Changes Implemented

### A. Frontend: `dashboard/index.html`
1. **Removed Fake Slider & Linear Formula:**
   - Deleted `<input id="fpr-slider" type="range">` and `<span id="fpr-slider-val">`.
   - Completely eliminated the JavaScript function `updateFPR(val)` and its linear scaling arithmetic `(baseThresh * (1.5 - val / 5.0))`.
2. **Added Verified Discrete Operating-Point Selector:**
   - `#btn-op-primary`: Operational Primary button ($\alpha = 0.2\%$).
   - `#btn-op-reference`: High-Recall Reference button ($\alpha = 1.0\%$, applies exclusively to v2; disabled and guarded for v1).
   - Added discrete badges:
     - `#calib-target-val`: Calibration Target ($\alpha = 0.2\%$ or $1.0\%$) on Monday benign set (`calibration_benign.parquet`).
     - `#thresh-val`: Calibrated Cutoff ($\tau$) anomaly threshold.
     - `#observed-fpr-val`: Operational Test FPR on held-out Tuesday–Friday evaluation traffic (`test_tue_fri.parquet`, 2,300,825 flows).
   - Added explicit conceptual distinction callout `#op-point-note` and `#op-point-restriction`.
3. **Synchronization & Incomplete Data Banner (`#sandbox-sync-status`):**
   - Added dynamic banner `#sandbox-sync-status`, icon `#sandbox-sync-icon`, and message `#sandbox-sync-msg`.
   - Displays clear error message if selected model data is unavailable, if required fields are missing, or if backend synchronization fails.
   - Includes a manual "Retry Sync" action.
4. **Dynamic Model Inspection Card (Card A) & Incomplete / Stale Indicators:**
   - Added semantic DOM IDs: `#card-a-title`, `#card-a-badge`, `#card-a-stale-tag`, `#card-a-pulse`, `#card-a-scope`, `#card-a-scope-sub`, `#card-a-recall`, `#card-a-fpr`, `#card-a-prec`, `#card-a-f1`, `#card-a-prauc`, `#card-a-rocauc`, `#card-a-threshold`, `#card-a-target-fpr`, `#card-a-scoring`.
   - Dynamically updates Card A to display verified metrics for the currently selected model and operating point without affecting live production authority.
   - When retained data is incomplete, displays `#card-a-stale-tag` (`INCOMPLETE DATA`) and badge suffix `[INCOMPLETE DATA]`.
   - When retained data is stale, displays `#card-a-stale-tag` (`STALE BENCHMARK`) and badge suffix `[STALE]`.
5. **Strict Field-Level Metric Validation (`isValidMetricValue`):**
   - Implemented helper `isValidMetricValue(val)` handling numeric finite values, string percentage values, and stripping prefix notations (`α = `) while strictly rejecting `null`, `undefined`, `NaN`, `""`, `"N/A"`, `"NaN"`, `"null"`, `"undefined"`, and nonnumeric strings.
   - Preserves valid numeric zero (`0`, `0.0`, `"0"`, `"0.0"`, `"0.00%"` are treated as valid).
6. **Elimination of All Hardcoded Fallbacks:**
   - Replaced all fallback expressions (`|| "93.08%"`, `|| "12.48%"`, `|| "0.190885"`, etc.) with ternary rendering: `isValidMetricValue(val) ? val : "N/A"`.
   - Removed silent fallback `modelBreakdowns[dataKey] || modelBreakdowns.temporal_ae`.
   - When `hasIncompleteMetrics` is true, surfaces visible incomplete-data warning banner, lists missing fields, and displays `"N/A"` for unavailable metrics with zero cross-model substitution.
7. **Triple API Synchronization (`fetchModelData`):**
   - Uses `Promise.allSettled` to query `/api/metrics`, `/api/models`, and `/api/compare_models`.
   - Validates HTTP status and response schema for each endpoint.
   - Validates model catalog from `/api/models`: verifies `active_production_champion === 'legacy_champion'`, legacy status is `'PRODUCTION CHAMPION'` with 32 dimensions, and v2 candidate with 26 dimensions. Does not overwrite evaluation metrics with catalog metadata.
   - Synchronizes evaluated metrics directly from API responses, storing `null` for missing fields so stale values cannot mask incomplete backend data.
   - Tracks `syncErrors`. If any endpoint fails, marks retained data as stale, displays failure banner, and avoids claiming successful synchronization.
8. **Head-to-Head Table Highlighting:**
   - Added row IDs: `#row-h2h-legacy`, `#row-h2h-v2-primary`, `#row-h2h-v2-ref`.
   - Active row dynamically receives highlighting rings (`ring-2 ring-cyan-400` / `ring-2 ring-amber-400`).

### B. Backend: `dashboard/server.py`
- Maintained 100% backward compatibility for all existing API routes (`/api/overview`, `/api/metrics`, `/api/models`, `/api/compare_models`, `/api/live_feed`, `/api/system_status`).
- Updated v2 integration in `/api/metrics` to expose both `threshold` and `threshold_value` with identical numerical values across file-read and adapter fallback branches.
- Added defensive fallback for `pure_one_way_v2_reference` in the adapter fallback branch.

---

## 3. Verified Operating Points & API Contracts

All displayed values originate from empirical evaluation on 2,300,825 flows (`outputs/experiment/test_tue_fri.parquet`):

| Model Architecture | Calibration Target ($\alpha$) | Score Cutoff ($\tau$) | Test Recall (%) | Test FPR (%) | Precision (%) | F1-Score | PR-AUC | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Legacy Champion (v1)** | 0.2% | 0.190885 | 93.08% | 12.48% | 70.46% | 0.8020 | 0.8783 | 0.9538 | Production Champion |
| **Pure One-Way v2 (Primary)** | 0.2% | 0.314036 | 85.38% | 10.70% | 71.86% | 0.7804 | 0.8687 | 0.9478 | Experimental Candidate |
| **Pure One-Way v2 (Reference)** | 1.0% | 0.099018 | 93.01% | 17.35% | 63.16% | 0.7523 | 0.8687 | 0.9478 | Experimental Candidate |

### API Contracts Utilized:
- `GET /api/metrics`: Returns JSON dictionary containing `temporal_autoencoder`, `pure_one_way_v2_primary`, `pure_one_way_v2_reference`, and baseline models with authentic confusion matrices, rates, and per-category detection breakdowns. Exposes both `threshold` and `threshold_value` for all models.
- `GET /api/models`: Returns catalog indicating `"active_production_champion": "legacy_champion"` with metadata for `legacy_champion` (32 dims) and `pure_one_way_v2` (26 dims).
- `GET /api/compare_models`: Returns comparative table and trade-off notes between v1 and v2 dual operating points.

---

## 4. Verification & Testing

### A. Calibration Sandbox Test Suite
Test suite: `tests/test_dashboard_calibration_sandbox.py` (16/16 passed in 3.83s):
1. `test_01_backend_api_metrics_and_operating_points`: Verifies `/api/metrics` returns exact verified evaluation rates, thresholds, and calibration targets for v1, v2 Primary, and v2 Reference.
2. `test_02_backend_compare_models_contract`: Confirms `/api/compare_models` returns all 3 models with matching dimensions and metrics.
3. `test_03_fake_continuous_slider_removed`: Proves `#fpr-slider`, `type="range"`, `updateFPR()`, and `1.5 - val / 5.0` are completely purged from `dashboard/index.html`.
4. `test_04_verified_discrete_selector_and_fpr_distinction`: Validates discrete selector buttons, discrete badges, and explicit differentiation between Monday calibration target and Tuesday–Friday test FPR.
5. `test_05_dynamic_card_a_and_table_row_contract`: Asserts all required Card A dynamic DOM IDs and Head-to-Head table row IDs are present.
6. `test_06_v1_cannot_select_v2_reference_point`: Verifies both DOM `disabled` attribute and JavaScript logic guard against v1 selecting the v2 1.0% reference point.
7. `test_07_production_invariants_preserved`: Validates model adapter dimensions (32 vs 26), thresholds, and authority statuses.
8. `test_08_all_three_backend_apis_fetched_and_validated`: Verifies all three endpoints are consumed, structurally validated, and `/api/models` catalog validation confirms champion identity and feature dimensions without overwriting evaluation metrics.
9. `test_09_no_silent_fallback_and_visible_error_handling`: Confirms removal of `modelBreakdowns[dataKey] || modelBreakdowns.temporal_ae`, verifying visible error state and `N/A` metrics when data is unavailable.
10. `test_10_stale_data_handling_and_failure_reporting`: Validates stale data indicators (`[STALE]`, `#card-a-stale-tag`) and verifies that failed API requests do not falsely claim synchronization success.
11. `test_11_field_level_metric_validation_helper`: Validates `isValidMetricValue()` logic across valid numeric zeros (`0`, `0.0`, `"0.00%"`), positive metrics, and invalid values (`null`, `undefined`, `NaN`, `""`, `"N/A"`, `"Baseline"`).
12. `test_12_no_hardcoded_metric_fallbacks_in_display`: Asserts zero hardcoded metric fallbacks in `updateCalibrationDisplay()` and confirms all 8 fields use `isValidMetricValue()` ternary rendering.
13. `test_13_field_level_validation_and_incomplete_banner`: Confirms all 8 required fields are checked, `hasIncompleteMetrics` is computed, and incomplete evaluation banner is displayed.
14. `test_14_backend_threshold_backward_compatibility`: Confirms both `threshold` and `threshold_value` fields exist with identical values on all `/api/metrics` model records.
15. `test_15_no_v1_substitution_for_missing_v2_metrics`: Confirms `/api/metrics` sync sets missing or null fields to `null`, completely preventing substitution of v1 values.
16. `test_16_strict_production_invariants`: Asserts v1 32 features / threshold 0.190885 and v2 26 features / dual thresholds 0.314036 and 0.099018.

### B. Integration & Full Regression Suite
- `tests/test_dashboard_integration.py`: 15/15 passed in 3.75s.
- Full pytest test suite: **196/196 passed in 63.11s**.
- `git diff --check`: 0 warnings/errors (clean whitespace).

---

## 5. Production Invariants Preserved

1. **v1 Alert Authority:** `LegacyModelAdapter` remains the sole production alert authority. Dashboard model switching is strictly isolated to sandbox visual inspection.
2. **v2 Shadow Status:** `PureOneWayV2Adapter` remains experimental/shadow-only and does not emit production verdicts.
3. **Dimensional Integrity:** v1 strictly enforces 32 dimensions; v2 strictly enforces 26 forward-only dimensions.
4. **No Synthetic Calibration:** No continuous interpolation or linear approximations exist. Only authentic empirical points are selectable.
5. **Zero Model Modifications:** Model artifacts, scalers, weights, and thresholds are untouched.
6. **Strict Field-Level Validation:** No model is rendered as synchronized unless all 8 required metrics are valid; missing metrics render as `"N/A"`, and no cross-model values are substituted.

---

## 6. Known Limitations

1. **Discrete vs Continuous Calibration:** Only 2 empirical calibration operating points ($\alpha = 0.2\%$ and $\alpha = 1.0\%$) exist for v2, and 1 ($\alpha = 0.2\%$) for v1. Additional operating points would require running full offline threshold sweeps on raw reconstruction score distributions.
2. **Operational Drift:** Operational FPR on Tuesday–Friday traffic (10.70%–12.48%) exceeds the Monday calibration target (0.2%). This drift is documented in the UI and test assertions; the underlying cause remains under investigation in Phase 3.
