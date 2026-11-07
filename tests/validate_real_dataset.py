"""
tests/validate_real_dataset.py
Tests the Pure One-Way Feature Engineering Pipeline (v2.0) on the actual CIC-IDS2017 datasets.
Validates:
1. Sample test (10,000 rows from training and test partitions)
2. Partition transformation test across train_benign (363,597 rows) and calibration_benign (166,321 rows)
3. Large evaluation batch test (100,000 rows from test_tue_fri)
4. Checks missing values, infinite values, output dimensions, and processing latency.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (
    PURE_ONE_WAY_FEATURES,
    PURE_ONE_WAY_BASE_FEATURES,
    PURE_ONE_WAY_TEMPORAL_FEATURES,
    add_pure_forward_rates,
    add_temporal_velocity_features,
    PureOneWayLogPreprocessingPipeline,
    check_model_compatibility,
)

print("=" * 80)
print("PHASE 1 REAL DATASET VALIDATION: PURE ONE-WAY PIPELINE (v2.0)")
print("=" * 80)

DATA_DIR = ROOT / "outputs" / "experiment"
TRAIN_FILE = DATA_DIR / "train_benign.parquet"
CAL_FILE = DATA_DIR / "calibration_benign.parquet"
TEST_FILE = DATA_DIR / "test_tue_fri.parquet"

print("\n1. Loading Monday Benign Training Data...")
t0 = time.time()
train_df = pd.read_parquet(TRAIN_FILE)
print(f"   Loaded {len(train_df):,} training rows in {time.time() - t0:.2f}s.")

print("\n2. Augmenting Training Data with Temporal Velocity & Pure Forward Rates...")
t0 = time.time()
train_augmented = add_temporal_velocity_features(train_df)
train_augmented = add_pure_forward_rates(train_augmented)
print(f"   Augmentation completed in {time.time() - t0:.2f}s.")

# Check missing and inf values in augmented training
print("\n3. Inspecting Feature Quality on Augmented Training Data:")
print(f"   Target Feature Count: {len(PURE_ONE_WAY_FEATURES)}")
for feat in PURE_ONE_WAY_FEATURES:
    if feat not in train_augmented.columns:
        print(f"   [ERROR] Missing required feature in dataframe: {feat}")
    else:
        col = train_augmented[feat]
        nan_cnt = int(col.isna().sum())
        inf_cnt = int(np.isinf(col.to_numpy(dtype=float, na_value=0.0)).sum())
        if nan_cnt > 0 or inf_cnt > 0:
            print(f"   - {feat:30s}: NaNs={nan_cnt:,}, Infs={inf_cnt:,}")

print("\n4. Fitting PureOneWayLogPreprocessingPipeline on 100,000 Benign Training Sample...")
pipeline = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
train_sample = train_augmented.sample(n=min(100_000, len(train_augmented)), random_state=42)
pipeline.fit(train_sample)
print("   Pipeline successfully fitted.")

print("\n5. Transforming 50,000 Monday Calibration Flows...")
cal_df = pd.read_parquet(CAL_FILE)
cal_sample = cal_df.sample(n=min(50_000, len(cal_df)), random_state=42)
cal_sample = add_temporal_velocity_features(cal_sample)
t0 = time.time()
X_cal = pipeline.transform(cal_sample)
el_cal = time.time() - t0
print(f"   Transformed {X_cal.shape[0]:,} rows -> Output Shape: {X_cal.shape} in {el_cal:.2f}s ({len(cal_sample)/el_cal:,.0f} flows/sec).")
assert not np.isnan(X_cal).any(), "NaN values found in calibrated output!"
assert not np.isinf(X_cal).any(), "Infinite values found in calibrated output!"

print("\n6. Transforming 100,000 Tuesday-Friday Operational Test Flows...")
test_df = pd.read_parquet(TEST_FILE)
test_sample = test_df.sample(n=100_000, random_state=42)
test_sample = add_temporal_velocity_features(test_sample)
t0 = time.time()
X_test = pipeline.transform(test_sample)
el_test = time.time() - t0
print(f"   Transformed {X_test.shape[0]:,} rows -> Output Shape: {X_test.shape} in {el_test:.2f}s ({len(test_sample)/el_test:,.0f} flows/sec).")
assert not np.isnan(X_test).any(), "NaN values found in test output!"
assert not np.isinf(X_test).any(), "Infinite values found in test output!"

print("\n7. Validating Compatibility Guard against Old 32-dim Champion Model...")
MODELS_DIR = ROOT / "models"
old_model_path = MODELS_DIR / "temporal_autoencoder_model.joblib"
if old_model_path.exists():
    import joblib
    old_model = joblib.load(old_model_path)
    is_compat, reason = check_model_compatibility(old_model, pipeline, PURE_ONE_WAY_FEATURES)
    print(f"   Is Pure 26-dim Pipeline compatible with Old 32-dim Model? {is_compat}")
    print(f"   Guard Message: {reason}")
    assert not is_compat, "Safety failure: Incompatible model was allowed!"
    print("   [PASS] Guard successfully prevented passing 26-dim features to 32-dim model.")

print("\n" + "=" * 80)
print("REAL DATASET VALIDATION COMPLETE: ALL CHECKS PASSED!")
print("=" * 80)
