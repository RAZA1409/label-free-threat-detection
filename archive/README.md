# Research Archive

This directory preserves the exploratory, diagnostic, and experimental scripts developed during the iterative research lifecycle for **SIH 2026 (Problem ID: SIH26145)**.

For the active production pipeline, refer to the [`src/`](../src/) directory.

---

## Directory Structure

### 1. `diagnostics/`
One-off schema validation and data integrity sanity checks used during initial dataset ingestion:
- `audit_timestamps.py`: Audited start/end timestamp ranges across all raw CIC-IDS2017 files.
- `check_schema.py` & `check_schema_details.py`: Inspected raw CSV headers and data types.
- `check_webattacks_csv.py`: Investigated trailing blank rows in the Thursday WebAttacks capture.
- `dataset_inspection.py`: Initial flow count and column sanity audit.
- `inspect_labelled_flows.py`: Examined 5-tuple identifier formats in `GeneratedLabelledFlows`.
- `inspect_negative_features.py`: Audited `-1` sentinels in TCP window and negative duration values.
- `validate_generated_flows.py`: Verified multi-file timestamp parsing formats.
- `verify_prepared_dataset.py`: Verified `cleaned_flows.parquet` output integrity.

### 2. `experiments/`
Exploratory model iterations prior to converging on the Deep One-Way Autoencoder:
- `analyze_feature_families.py`: Preliminary feature grouping tests.
- `evaluate_pca_thresholds.py`: Threshold sensitivity sweeps on full-dimension PCA.
- `train_30_feature_pca.py` & `train_compact_pca_detector.py`: Heuristic feature-pruning trials.
- `train_combined_detector.py`: Early Isolation Forest + PCA rank combination.
- `train_isolation_forest_log.py` & `train_isolation_forest_robust.py`: Feature scaling variations on tree detectors.
- `train_kmeans_detector.py`: MiniBatchKMeans clustering experiment.
- `train_lof.py` & `train_pca_lof_hybrid.py`: Local Outlier Factor experiments (curse of dimensionality exploration).
- `train_pca_detector.py` & `train_pca_feature_detector.py`: Initial bidirectional 78-feature PCA baselines.
- `train_reduced_pca_detector.py`: 39-feature bidirectional PCA experiment.
- `train_targeted_isolation_forest.py`: 5-feature targeted tree experiment.
- `train_transformed_pca_detector.py`: Early feature transformation prototype.

### 3. `analysis/`
Terminal reporting and error-distribution scripts superseded by the unified XAI engine (`src/explainability.py`), the model leaderboard (`src/create_model_report.py`), and the interactive dashboard:
- `analyze_attack_detection.py`, `analyze_attack_scores.py`, `analyze_attack_feature_distributions.py`: Per-attack error breakdowns for older models.
- `analyze_benign_feature_distributions.py` & `analyze_calibration_thresholds.py`: Benign distribution audits.
- `analyze_feature_redundancy.py`: Spearman correlation matrix analysis.
- `analyze_features.py`: Feature variance and summary statistics.
- `analyze_oneway_attack_features.py`: Forward-feature distribution comparison.
- `analyze_pca_missed_attacks.py` & `analyze_reduced_pca_missed_attacks.py`: Residual error audits on missed attacks.
- `analyze_targeted_if_scores.py` & `analyze_thresholds.py`: Quantile distribution checks.
- `compare_all_model_attacks.py`, `compare_attack_detection.py`, `compare_models.py`, `compare_pca_feature_detector.py`: Ad-hoc comparison scripts.
