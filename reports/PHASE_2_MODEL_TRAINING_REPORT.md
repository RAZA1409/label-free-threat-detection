# AEGIS – PHASE 2 STAGE 2 REPORT: PURE ONE-WAY AUTOENCODER TRAINING & CALIBRATION

**Project:** AEGIS – Label-Free Threat Detection in One-Way Network Traffic\
**Problem Statement:** SIH 2026 (Problem ID: SIH26145)\
**Phase:** Phase 2 (Model Training & Evaluation) — Stage 2 (Autoencoder Implementation & Calibration)\
**Date:** November 4, 2026\
**Active Git Branch:** `feat/phase-2-pure-one-way-model` (based on `aa99971`)\
**Status:** TRAINING & CALIBRATION COMPLETE — 24/24 AUTOMATED TESTS PASSED\

---

## 1. Executive Summary

In Phase 2 Stage 2, the new **Pure One-Way Temporal Autoencoder (v2.0)** was implemented, trained, and calibrated in complete isolation from the legacy production model. The model operates exclusively on the verified **`PURE_ONE_WAY_FEATURES` (26 dimensions)** specification engineered in Phase 1, using strictly forward-only physical telemetry and causal rolling temporal context ($(t - 10\text{s}, t]$).

### Key Accomplishments:
1. **Dedicated Architecture:** A symmetric bottleneck autoencoder ($26 \to 20 \to 8 \to 20 \to 26$) was trained via `src/train_pure_oneway_autoencoder.py`.
2. **Time-Aware Training & Validation:** Trained on 120,000 Monday benign flows drawn from `train_benign.parquet` (`01:00:01` to `09:44:59`), evaluated on a separate chronological validation holdout of 25,000 flows (`09:45:00` to `10:44:33`).
3. **Loss Diagnostics & Convergence:** Converged in **16.57 seconds** at iteration 39 (before `max_iter=45`).
   - Training Reconstruction MSE: **`0.012244`**
   - Validation Reconstruction MSE: **`0.012927`**
   - Generalization Gap: **`0.000683`** (exhibiting exceptional stability across the chronological time boundary).
4. **Independent Threshold Calibration:** Anomaly thresholds were calibrated on the complete **166,321 Monday benign calibration flows** using unweighted MSE scoring with `method="higher"`:
   - Primary Threshold ($\alpha = 0.002$): **`0.314036`** (332 false alarms / 166,321 = **`0.1996%`** empirical FPR).
   - Reference Threshold ($\alpha = 0.010$): **`0.099018`** (1,663 false alarms / 166,321 = **`0.9999%`** empirical FPR).
5. **Artifact Isolation:** All v2 models and configs were atomically saved to `models/pure_one_way_v2/`. The legacy 32-dim champion models in `models/` remain **100% untouched and operational**.
6. **Testing Verification:** **24/24 automated pytest tests passed cleanly** (including 9 dedicated Phase 2 tests, 12 Phase 1 feature tests, and 3 streaming tests).

---

## 2. Git Working Tree & Safety State

All modifications remain strictly isolated on the current feature branch:

```text
$ git branch --show-current
feat/phase-2-pure-one-way-model

$ git log -1 --oneline
aa99971 feat: implement pure one-way feature engineering

$ git status --short
 M src/features.py
 M tests/test_pure_one_way_features.py
?? archive/backups/
?? reports/PHASE_2_MODEL_TRAINING_REPORT.md
?? reports/PHASE_2_TRAINING_PIPELINE_AUDIT.md
?? src/train_pure_oneway_autoencoder.py
?? tests/test_pure_one_way_autoencoder.py
```

* **No Production Artifacts Overwritten:** `models/temporal_autoencoder_*` timestamps and dimensions (32-dim) were verified untouched.
* **No Commits or Pushes:** Code and reports are staged for user review.

---

## 3. Model Architecture & Hyperparameter Configuration

The model is implemented using `sklearn.neural_network.MLPRegressor` with a deep symmetric bottleneck topology:

$$\mathbf{x} \in \mathbb{R}^{26} \xrightarrow{\text{Dense}(26, 20)} \mathbf{h}_1 \xrightarrow{\text{Dense}(20, 8)} \mathbf{z} \in \mathbb{R}^8 \xrightarrow{\text{Dense}(8, 20)} \mathbf{h}_2 \xrightarrow{\text{Dense}(20, 26)} \mathbf{\hat{x}} \in \mathbb{R}^{26}$$

```
Input Layer:        [26 Dimensions]  (Standardized, log1p-transformed forward flow metrics)
                          │
Encoder Layer 1:    [20 Neurons]     (Dense + ReLU) — Captures non-linear feature interactions
                          │
Latent Bottleneck:  [ 8 Neurons]     (Dense + ReLU) — Compresses normal traffic into core manifold
                          │
Decoder Layer 1:    [20 Neurons]     (Dense + ReLU) — Reconstructs correlated physical parameters
                          │
Output Layer:       [26 Dimensions]  (Linear / Identity) — Reconstructed feature vector x_hat
```

### Hyperparameters:
* **Hidden Layers:** `(20, 8, 20)`
* **Activation:** `relu`
* **Optimizer:** `adam`
* **Initial Learning Rate:** `0.003` with `learning_rate="adaptive"`
* **L2 Regularization ($\alpha$):** `0.0001`
* **Batch Size:** `256`
* **Maximum Iterations:** `45`
* **Early Stopping:** `False` (iterations managed via loss plateau tolerance `tol=1e-4` over 10 consecutive epochs)
* **Random State:** `42`

---

## 4. Dataset Partitioning, Sampling & Time-Aware Validation

The training pipeline uses only the Monday benign partition (`train_benign.parquet`):

```
Monday Benign Training Partition (363,597 flows, 01:00:01 -> 10:44:33)
                              │
               Chronological Cutoff: 09:45:00
                              ├───────────────────────────────────────┐
                              ▼                                       ▼
                 Training Pool (270,161 flows)             Validation Pool (93,436 flows)
                   [01:00:01 -> 09:44:59]                     [09:45:00 -> 10:44:33]
                              │                                       │
                Sampled: 120,000 flows (random_state=42)   Sampled: 25,000 flows (random_state=42)
```

### Partition Boundaries & Integrity:
* **Training Sample Size:** 120,000 flows. Spans `2017-07-03 01:00:01` to `2017-07-03 09:44:59`. Preserves diurnal diversity across 8.75 hours of traffic.
* **Chronological Validation Segment:** 25,000 flows. Spans `2017-07-03 09:45:00` to `2017-07-03 10:44:33`.
* **Temporal Overlap:** Exactly **0 flows**. $\max(t_{\text{train}}) < \min(t_{\text{val}})$.
* **Unsupervised Purity:** 100% of rows in both pools are `"BENIGN"`. Zero attack labels were exposed.
* **Preprocessing Isolation:** `pipeline.fit()` was called strictly on the 120,000 training flows. The validation flows were strictly transformed.

---

## 5. Feature Engineering Pipeline Verification

The model ingests the 26 pure one-way forward features defined in `src/features.py`:

```
1.  Total Fwd Packets             14. Fwd IAT Total
2.  Total Length of Fwd Packets   15. Fwd IAT Mean
3.  act_data_pkt_fwd              16. Fwd IAT Std
4.  Subflow Fwd Packets           17. Fwd IAT Max
5.  Subflow Fwd Bytes             18. Fwd IAT Min
6.  Fwd Header Length             19. Fwd_Bytes_Per_Sec
7.  min_seg_size_forward          20. Fwd_Packets_Per_Sec
8.  Avg Fwd Segment Size          21. Fwd PSH Flags
9.  Init_Win_bytes_forward        22. Fwd URG Flags
10. Fwd Packet Length Max         23. SYN Flag Count
11. Fwd Packet Length Min         24. Rolling_Flow_Velocity_10s
12. Fwd Packet Length Mean        25. Rolling_Bytes_Velocity_10s
13. Fwd Packet Length Std         26. Rolling_Single_Pkt_Ratio_10s
```

* **Preprocessing Pipeline:** `PureOneWayLogPreprocessingPipeline`
  - Median imputation (`keep_empty_features=True`).
  - Derived pure forward rates (`Fwd_Bytes_Per_Sec`, `Fwd_Packets_Per_Sec`) guarded against division by zero.
  - Safe log1p transform applied to 14 skewed power-law forward features.
  - Standard scaling (`StandardScaler`) fitted exclusively on normal training data.
* **Causal Rolling Engine:** Causal rolling window $(t - 10\text{s}, t]$ with optional historical context buffer to prevent cold-start boundary distortion.

---

## 6. Training Convergence & Reconstruction Loss Diagnostics

```text
Training on 120,000 observations (max_iter=45, batch_size=256)...
Iteration 1, loss = 0.14780942
Iteration 2, loss = 0.05942138
Iteration 3, loss = 0.04587746
...
Iteration 37, loss = 0.00837079
Iteration 38, loss = 0.00779776
Iteration 39, loss = 0.00799840
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
Training finished in 16.57s across 39 iterations.
```

### Loss Metrics:
* **Convergence Status:** **CONVERGED** (reached tolerance plateau at iteration 39 without hitting `max_iter=45`).
* **Final Training Objective Loss:** `0.007998`
* **Training Reconstruction MSE:** **`0.012244`**
* **Validation Reconstruction MSE:** **`0.012927`**
* **Generalization Gap:** $|MSE_{\text{val}} - MSE_{\text{train}}| = 0.000683$ ($5.58\%$ relative difference).
* **Finding:** The model reconstructs unseen future benign network traffic from Monday late morning with virtually the same accuracy as the training data, confirming that the 8-dimensional bottleneck captures intrinsic benign manifold invariants without overfitting.

---

## 7. Threshold Calibration on Monday Benign Calibration Data

Threshold calibration was executed across the complete **166,321 Monday benign calibration flows** (`calibration_benign.parquet`):

* **Scoring Function:** Unweighted Mean Squared Error (MSE):
  $$S(\mathbf{x}) = \frac{1}{26} \sum_{j=1}^{26} (x_j - \hat{x}_j)^2$$
* **Quantile Method:** `np.quantile(scores, 1 - alpha, method="higher")`
* **Decision Rule:** $\text{score} > \text{threshold}$ (strictly greater than).

```
+----------------------------------------------------------------------------------------------------+
| Calibration Metric              | Primary Operating Point (0.2%) | Reference Operating Point (1.0%)|
+---------------------------------+--------------------------------+---------------------------------+
| Target Calibration FPR          | 0.2000%                        | 1.0000%                         |
| Calibrated Threshold            | 0.314036                       | 0.099018                        |
| Calibration Observations        | 166,321                        | 166,321                         |
| False Positives (Score > Thresh)| 332                            | 1,663                           |
| Empirical Calibration FPR       | 0.1996%                        | 0.9999%                         |
| Relative Target Error           | -0.0004%                       | -0.0001%                        |
+----------------------------------------------------------------------------------------------------+
```

### Calibration Score Distribution Statistics:
* Minimum Score: `0.000291`
* Median Score: `0.005303`
* Mean Score: `0.012346`
* 99th Percentile: `0.099012`
* Maximum Score: `2.145047`

* **Independence Guarantee:** Calibration data was never used to fit the scaler, imputer, or autoencoder weights.
* **Operational Test Isolation:** Zero Tuesday–Friday flows or attack labels were touched during threshold setting.

---

## 8. Artifact Isolation & Integrity

All v2 artifacts were safely written to `models/pure_one_way_v2/`:

```text
models/pure_one_way_v2/
├── model.joblib              [54,855 bytes]  Fitted MLPRegressor autoencoder (26 -> 20 -> 8 -> 20 -> 26)
├── pipeline.joblib            [3,163 bytes]  Fitted PureOneWayLogPreprocessingPipeline (26 features)
├── feature_manifest.json      [1,794 bytes]  Strict ordered manifest of 26 features
├── train_config.json          [1,322 bytes]  Architecture, hyperparameters, and convergence diagnostics
└── threshold_config.json        [723 bytes]  Primary (0.314036) and Reference (0.099018) thresholds
```

### Non-Disruption Verification:
* Legacy champion model `models/temporal_autoencoder_model.joblib` (32 dims, Last Modified: `03-Nov-26 2:32:48 PM`) remains completely unmodified.
* Active dashboard server on `:8501` remains unaffected.

---

## 9. Automated Testing Results

The comprehensive test suite now contains **24 automated tests** across three modules:

```text
$ python -m pytest -v
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\D drive\Label-free-threat-detection
collected 24 items

tests/test_pure_one_way_autoencoder.py::test_01_model_input_and_reconstruction_dimensions PASSED [  4%]
tests/test_pure_one_way_autoencoder.py::test_02_unweighted_mse_scoring PASSED [  8%]
tests/test_pure_one_way_autoencoder.py::test_03_threshold_calculation_and_ties PASSED [ 12%]
tests/test_pure_one_way_autoencoder.py::test_04_calibration_independence PASSED [ 16%]
tests/test_pure_one_way_autoencoder.py::test_05_feature_dimension_mismatch_and_guard PASSED [ 20%]
tests/test_pure_one_way_autoencoder.py::test_06_deterministic_training_sampling PASSED [ 25%]
tests/test_pure_one_way_autoencoder.py::test_07_no_attack_labels_in_monday_training PASSED [ 29%]
tests/test_pure_one_way_autoencoder.py::test_08_no_overwrite_of_old_champion PASSED [ 33%]
tests/test_pure_one_way_autoencoder.py::test_09_artifact_save_and_load_integrity PASSED [ 37%]
tests/test_pure_one_way_features.py::test_01_feature_specification_validation PASSED [ 41%]
tests/test_pure_one_way_features.py::test_02_reverse_direction_exclusion PASSED [ 45%]
tests/test_pure_one_way_features.py::test_03_forward_only_rate_calculation PASSED [ 50%]
tests/test_pure_one_way_features.py::test_04_directional_invariance PASSED [ 54%]
tests/test_pure_one_way_features.py::test_05_feature_order_consistency PASSED [ 58%]
tests/test_pure_one_way_features.py::test_06_preprocessing_robustness PASSED [ 62%]
tests/test_pure_one_way_features.py::test_07_rolling_feature_correctness PASSED [ 66%]
tests/test_pure_one_way_features.py::test_08_pipeline_output_shape_and_validity PASSED [ 70%]
tests/test_pure_one_way_features.py::test_09_feature_reproducibility PASSED [ 75%]
tests/test_pure_one_way_features.py::test_10_model_compatibility_guard PASSED [ 79%]
tests/test_pure_one_way_features.py::test_11_temporal_causality_no_future_leakage PASSED [ 83%]
tests/test_pure_one_way_features.py::test_12_cross_partition_boundary_policy PASSED [ 87%]
tests/test_streaming_compatibility.py::test_streaming_single_forward_flow_transformation PASSED [ 91%]
tests/test_streaming_compatibility.py::test_streaming_multi_packet_forward_burst PASSED [ 95%]
tests/test_streaming_compatibility.py::test_streaming_forward_only_absence_of_reverse_fields PASSED [100%]

============================= 24 passed in 42.32s =============================
```# AEGIS PHASE 2 — STAGE 3: FULL OPERATIONAL EVALUATION

You are working on the AEGIS Label-Free Threat Detection project.

Repository: `RAZA1409/label-free-threat-detection`

Current branch: `feat/phase-2-pure-one-way-model`

Base commit: `aa99971`

## OBJECTIVE

Perform a rigorous, leakage-free operational evaluation of:

1. The new Pure One-Way 26-feature autoencoder.
2. The existing 32-feature legacy champion model.

Use the Tuesday–Friday CIC-IDS2017 operational dataset containing approximately 2,300,825 flows and 14 attack categories.

This stage is evaluation only. Do not retrain either model or tune thresholds using operational labels.

## STEP 1 — SAFETY AND DATA AUDIT

Before implementation:

- Inspect Git branch, status and recent commits.
- Read `reports/PHASE_2_MODEL_TRAINING_REPORT.md`.
- Inspect `reports/PHASE_2_TRAINING_PIPELINE_AUDIT.md`.
- Inspect the Phase 1 feature pipeline and its tests.
- Verify all artifact paths and model compatibility.
- Identify the exact Tuesday–Friday operational dataset and its labels.
- Verify there are no training/calibration/test row overlaps.
- Verify timestamps and label distributions.
- Check the modifications to `src/features.py` and its tests.

Do not modify or overwrite any existing model artifacts.

Do not commit or push anything.

## STEP 2 — NEW MODEL EVALUATION

Load the exact saved artifacts from:

`models/pure_one_way_v2/`

Use:

- `model.joblib`
- `pipeline.joblib`
- `feature_manifest.json`
- `threshold_config.json`

Verify feature order, 26-dimensional input and model compatibility.

Transform the operational dataset using the saved pipeline only.

Use the pre-calibrated thresholds:

- Primary: 0.314036
- Reference: 0.099018

Anomaly rule:

`score > threshold`

Do not fit or update the scaler, preprocessing pipeline or autoencoder.

Do not change the thresholds.

## STEP 3 — LEGACY CHAMPION EVALUATION

Identify and load the existing 32-feature champion and its exact preprocessing and threshold artifacts.

Do not assume that the new 26-feature pipeline is compatible with the legacy model.

Use the legacy model's own feature transformation, feature order and previously established threshold.

If the legacy threshold's provenance cannot be verified, document this limitation clearly. Do not invent a threshold or silently recalibrate it using test labels.

Both models must be evaluated on the same operational flows wherever their pipelines permit a fair comparison.

## STEP 4 — METRICS

Calculate for both models:

- True Positives
- False Positives
- True Negatives
- False Negatives
- Precision
- Recall / Detection Rate
- F1-score
- False Positive Rate
- False Negative Rate
- Accuracy
- Balanced Accuracy
- PR-AUC / Average Precision
- ROC-AUC

Use the appropriate continuous anomaly scores for ROC-AUC and PR-AUC, not binary predictions.

Also calculate:

1. Per-attack-category recall, precision and F1.
2. Per-day metrics for Tuesday, Wednesday, Thursday and Friday.
3. Benign-only operational false positive rate.
4. Attack-only detection rate.
5. Confusion matrices.
6. Inference time and throughput.

Clearly distinguish micro, macro and weighted averages.

Handle class imbalance correctly. Do not rely on accuracy alone.

## STEP 5 — THRESHOLD SENSITIVITY

For the new model, evaluate both already-calibrated thresholds:

- 0.314036
- 0.099018

Do not choose a new threshold based on operational labels.

Explain the precision/recall and false-positive trade-off between the two operating points.

Do not present the best-performing test threshold as a deployable threshold.

## STEP 6 — TEMPORAL AND STREAMING VALIDATION

Verify that operational feature generation respects event-time causality.

- Do not use future events to calculate rolling features.
- Preserve timestamp ordering.
- Respect the 10-second rolling window.
- Verify overnight gaps naturally expire historical context.
- Do not carry Monday calibration history into Tuesday after the long overnight gap.
- Check that offline batch transformations and streaming transformations follow equivalent semantics where applicable.

Document any limitation caused by using offline CIC-IDS2017 flow records rather than a live network sensor.

## STEP 7 — REPORTS AND ARTIFACTS

Create:

`reports/PHASE_2_OPERATIONAL_EVALUATION_REPORT.md`

Include:

- Dataset provenance and partition validation.
- Exact evaluation row counts.
- Label distribution.
- Model artifact versions.
- Feature compatibility.
- Threshold provenance.
- Confusion matrices.
- Overall metrics.
- Per-attack results.
- Per-day results.
- Threshold comparison.
- Runtime and throughput.
- Limitations and methodology risks.
- Final comparison between the two models.

Save machine-readable results in:

`reports/phase_2_operational_results/`

Include JSON or CSV outputs for metrics, per-class results and per-day results.

Do not save large generated datasets or duplicate model files.

## STRICT RULES

- Do not train or retrain either model.
- Do not fit preprocessing on Tuesday–Friday data.
- Do not use attack labels for threshold selection.
- Do not tune hyperparameters on operational results.
- Do not modify the legacy champion.
- Do not modify the dashboard or streaming integration.
- Do not overwrite Phase 1 or Phase 2 artifacts.
- Do not commit or push.
- If any artifact, partition or threshold is ambiguous, stop and report the ambiguity rather than guessing.

## FINAL DELIVERABLE

Show:

1. Complete evaluation report.
2. Overall comparison table.
3. Per-attack detection table.
4. Per-day performance table.
5. Threshold sensitivity comparison.
6. Test results and Git status.
7. A clear verdict on whether the new 26-feature model outperforms, matches or underperforms the existing 32-feature champion.

Remember: this is an evaluation stage, not a model optimization stage.

---

## 10. Unresolved Limitations & Stage 3 Evaluation Roadmap

### Identified Limitations:
1. **Operational Test Evaluation Pending:** In accordance with constraints, the model has NOT yet been evaluated on the 2,300,825 Tuesday–Friday attack flows.
2. **Benign Temporal Drift:** Calibrating at 0.2% on Monday calibration may experience elevated false alarm rates on subsequent weekdays due to organic business shift. Stage 3 will explicitly evaluate threshold sensitivity across $\alpha \in [0.001, 0.05]$.
3. **Forensic Attribution Support:** `ThreatExplainer` in `src/explainability.py` must be upgraded to load `models/pure_one_way_v2/` and attribute reconstruction errors across the 26 pure forward dimensions.

### Next Steps (Stage 3):
1. Execute full operational test evaluation across all 2,300,825 Tuesday–Friday flows.
2. Generate comprehensive confusion matrix, ROC-AUC, PR-AUC, and per-attack breakdown across all 14 attack classes.
3. Conduct side-by-side performance comparison against the legacy 32-dim champion model.
4. Update `src/explainability.py`, `src/simulate_stream.py`, and `dashboard/server.py` to support v2.0 in production.

---

**AEGIS Phase 2 Stage 2 is COMPLETE and fully verified.**
