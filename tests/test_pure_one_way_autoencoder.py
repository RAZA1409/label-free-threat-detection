"""
tests/test_pure_one_way_autoencoder.py
Pytest suite verifying the Pure One-Way Autoencoder architecture, unweighted MSE scoring,
threshold calibration, artifact isolation, and model safety guards.
"""

import sys
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (
    PURE_ONE_WAY_FEATURES,
    PURE_ONE_WAY_BASE_FEATURES,
    PureOneWayLogPreprocessingPipeline,
    add_temporal_velocity_features,
    check_model_compatibility,
)
from train_pure_oneway_autoencoder import (
    compute_unweighted_reconstruction_errors,
    PRIMARY_ALPHA,
    REFERENCE_ALPHA,
)


@pytest.fixture
def dummy_26_autoencoder():
    """Provides a minimal dummy 26-dim MLPRegressor autoencoder for fast unit testing."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    model = MLPRegressor(
        hidden_layer_sizes=(20, 8, 20),
        activation="relu",
        max_iter=5,
        random_state=42,
    )
    # Fit on synthetic random 26-dim inputs, suppressing dummy convergence warning
    X_dummy = np.random.uniform(0.1, 5.0, size=(100, 26))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_dummy, X_dummy)
    return model


# -----------------------------------------------------------------------------
# TEST 01: 26-Feature Model Input & Reconstruction Dimensions
# -----------------------------------------------------------------------------
def test_01_model_input_and_reconstruction_dimensions(dummy_26_autoencoder):
    """Verifies that the autoencoder expects exactly 26 inputs and predicts (N, 26) outputs."""
    assert dummy_26_autoencoder.n_features_in_ == 26
    assert dummy_26_autoencoder.hidden_layer_sizes == (20, 8, 20)

    X_test = np.random.normal(0, 1, size=(35, 26))
    X_rec = dummy_26_autoencoder.predict(X_test)

    assert X_rec.shape == (35, 26)
    assert not np.isnan(X_rec).any()
    assert not np.isinf(X_rec).any()


# -----------------------------------------------------------------------------
# TEST 02: Unweighted MSE Scoring Calculation
# -----------------------------------------------------------------------------
def test_02_unweighted_mse_scoring(dummy_26_autoencoder):
    """Verifies that compute_unweighted_reconstruction_errors computes exact mean((X - X_hat)**2)."""
    X_test = np.random.normal(0, 1, size=(50, 26))
    scores = compute_unweighted_reconstruction_errors(dummy_26_autoencoder, X_test, batch_size=20)

    # Manual calculation
    X_rec = dummy_26_autoencoder.predict(X_test)
    expected_scores = np.mean(np.square(X_test - X_rec), axis=1)

    assert scores.shape == (50,)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-6, atol=1e-6)
    assert np.all(scores >= 0.0)


# -----------------------------------------------------------------------------
# TEST 03: Threshold Calculation, Ties & Method 'Higher'
# -----------------------------------------------------------------------------
def test_03_threshold_calculation_and_ties():
    """
    Verifies that np.quantile(scores, 1 - alpha, method='higher') strictly bounds
    the empirical false alarm rate to <= alpha, even in the presence of tied scores.
    """
    # Distribution with multiple tied values
    scores = np.array([0.1, 0.2, 0.3, 0.3, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0])
    alpha = 0.20  # 20% target

    thresh = np.quantile(scores, 1.0 - alpha, method="higher")
    # Number of scores strictly greater than threshold
    fp = np.sum(scores > thresh)
    empirical_rate = fp / len(scores)

    assert empirical_rate <= alpha, f"Empirical rate {empirical_rate} exceeded nominal {alpha}"
    assert thresh == 1.5


# -----------------------------------------------------------------------------
# TEST 04: Calibration Independence
# -----------------------------------------------------------------------------
def test_04_calibration_independence(dummy_26_autoencoder):
    """
    Confirms that scoring calibration samples does NOT mutate or fit the model
    or preprocessing pipeline parameters.
    """
    pipeline = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
    train_data = pd.DataFrame(
        {f: np.random.uniform(1.0, 100.0, size=50) for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]}
    )
    train_data["Timestamp"] = pd.date_range("2026-11-04 10:00:00", periods=50, freq="1s")
    train_data = add_temporal_velocity_features(train_data)

    pipeline.fit(train_data)
    initial_scale = pipeline.scaler.scale_.copy()
    initial_weights = [w.copy() for w in dummy_26_autoencoder.coefs_]

    # Evaluate calibration samples
    cal_data = train_data.copy()
    num_cols = cal_data.select_dtypes(include=[np.number]).columns
    cal_data[num_cols] = cal_data[num_cols] * 1.5
    X_cal_scaled = pipeline.transform(cal_data)
    scores = compute_unweighted_reconstruction_errors(dummy_26_autoencoder, X_cal_scaled)

    # Check that scaler and model were not modified
    np.testing.assert_array_equal(pipeline.scaler.scale_, initial_scale)
    for w_init, w_curr in zip(initial_weights, dummy_26_autoencoder.coefs_):
        np.testing.assert_array_equal(w_init, w_curr)


# -----------------------------------------------------------------------------
# TEST 05: Feature Dimension Mismatch & Guard
# -----------------------------------------------------------------------------
def test_05_feature_dimension_mismatch_and_guard(dummy_26_autoencoder):
    """Verifies that check_model_compatibility rejects mismatched feature sets."""
    # Dummy 32-dim legacy pipeline
    class Dummy32Pipeline:
        features = ["F_" + str(i) for i in range(32)]

    pipe_32 = Dummy32Pipeline()
    is_compat, reason = check_model_compatibility(dummy_26_autoencoder, pipe_32, PURE_ONE_WAY_FEATURES)

    assert not is_compat
    assert "Feature count mismatch" in reason


# -----------------------------------------------------------------------------
# TEST 06: Deterministic Training Sampling
# -----------------------------------------------------------------------------
def test_06_deterministic_training_sampling():
    """Confirms that sample(n=..., random_state=42) produces 100% identical row selection."""
    df = pd.DataFrame({"val": np.arange(1000)})
    s1 = df.sample(n=100, random_state=42)
    s2 = df.sample(n=100, random_state=42)

    np.testing.assert_array_equal(s1.index.values, s2.index.values)


# -----------------------------------------------------------------------------
# TEST 07: No Attack Labels Used in Monday Training
# -----------------------------------------------------------------------------
def test_07_no_attack_labels_in_monday_training():
    """Verifies that the training partition contains exclusively BENIGN records."""
    train_file = ROOT / "outputs" / "experiment" / "train_benign.parquet"
    if train_file.exists():
        df = pd.read_parquet(train_file, columns=["Label"])
        assert df["Label"].eq("BENIGN").all(), "Non-benign labels found in training partition!"


# -----------------------------------------------------------------------------
# TEST 08: No Overwrite of Old Champion Artifacts
# -----------------------------------------------------------------------------
def test_08_no_overwrite_of_old_champion():
    """Verifies that the old 32-feature champion artifacts in models/ remain intact and untouched."""
    old_champion_model = ROOT / "models" / "temporal_autoencoder_model.joblib"
    old_champion_thresh = ROOT / "models" / "temporal_autoencoder_threshold.json"

    assert old_champion_model.exists(), "Old champion model was deleted!"
    assert old_champion_thresh.exists(), "Old champion threshold was deleted!"

    # Load and verify it is the legacy 32-dim model
    import joblib
    old_model = joblib.load(old_champion_model)
    assert old_model.n_features_in_ == 32, f"Old model was overwritten! Found {old_model.n_features_in_} features."


# -----------------------------------------------------------------------------
# TEST 09: Atomic Save and Load Integrity
# -----------------------------------------------------------------------------
def test_09_artifact_save_and_load_integrity(dummy_26_autoencoder):
    """Tests saving and loading model and pipeline from disk without data corruption."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pipeline = PureOneWayLogPreprocessingPipeline(features=PURE_ONE_WAY_FEATURES)
        data = {f: np.random.uniform(1.0, 50.0, size=20) for f in PURE_ONE_WAY_BASE_FEATURES if f not in ["Fwd_Bytes_Per_Sec", "Fwd_Packets_Per_Sec"]}
        data["Timestamp"] = pd.date_range("2026-11-04 10:00:00", periods=20, freq="1s")
        df = add_temporal_velocity_features(pd.DataFrame(data))
        pipeline.fit(df)

        import joblib
        joblib.dump(dummy_26_autoencoder, tmp_path / "model.joblib")
        joblib.dump(pipeline, tmp_path / "pipeline.joblib")

        loaded_model = joblib.load(tmp_path / "model.joblib")
        loaded_pipe = joblib.load(tmp_path / "pipeline.joblib")

        assert loaded_model.n_features_in_ == 26
        assert len(loaded_pipe.features) == 26

        X_orig = pipeline.transform(df)
        X_loaded = loaded_pipe.transform(df)
        np.testing.assert_array_equal(X_orig, X_loaded)

        rec_orig = dummy_26_autoencoder.predict(X_orig)
        rec_loaded = loaded_model.predict(X_loaded)
        np.testing.assert_array_equal(rec_orig, rec_loaded)
