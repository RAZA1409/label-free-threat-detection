"""
src/model_adapters.py
AEGIS Phase 3: Dual-Model Dashboard Integration Adapters

Provides isolated, self-contained adapters for:
1. LegacyModelAdapter: Legacy 32-feature champion autoencoder (Production Champion)
2. PureOneWayV2Adapter: Pure One-Way 26-feature autoencoder (Experimental / Research Candidate)

Guarantees:
- Strict feature dimensionality validation (rejects cross-model feature misuse)
- Independent artifact deserialization with graceful degradation
- Distinct scoring formulas (residual-weighted MSE vs. unweighted MSE)
- Zero filesystem path leakage to public API responses
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
V2_MODELS_DIR = MODELS_DIR / "pure_one_way_v2"
EVAL_DIR = ROOT_DIR / "outputs" / "evaluation"
RESULTS_DIR = ROOT_DIR / "reports" / "phase_2_operational_results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Pickle compatibility aliases for legacy and pure one-way preprocessing pipelines
from features import (
    LogPreprocessingPipeline,
    PureOneWayLogPreprocessingPipeline,
    PURE_ONE_WAY_FEATURES,
)
sys.modules['__main__'].LogPreprocessingPipeline = LogPreprocessingPipeline
sys.modules['__main__'].PureOneWayLogPreprocessingPipeline = PureOneWayLogPreprocessingPipeline


class LegacyModelAdapter:
    """
    Adapter for the Legacy 32-feature Temporal Autoencoder (Active Production Champion).
    Requires 32 input dimensions and inverse-variance residual weighting.
    """

    MODEL_NAME: str = "Legacy Temporal Autoencoder (v1)"
    MODEL_VERSION: str = "1.0-champion"
    STATUS: str = "PRODUCTION CHAMPION"
    EXPECTED_DIM: int = 32
    DECISION_RULE: str = "score >= threshold"

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.model_name: str = self.MODEL_NAME
        self.model_version: str = self.MODEL_VERSION
        self.status: str = self.STATUS
        self.expected_dim: int = self.EXPECTED_DIM
        self.decision_rule: str = self.DECISION_RULE
        self.model = None
        self.pipeline = None
        self.weights = None
        self.threshold: float = 0.19088475755009496
        self.target_fpr: float = 0.002
        self.feature_names: List[str] = []
        self.is_available: bool = False
        self.load_error: Optional[str] = None

        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """Loads legacy model, pipeline, residual weights, and threshold config."""
        model_path = self.models_dir / "temporal_autoencoder_model.joblib"
        pipe_path = self.models_dir / "temporal_autoencoder_pipeline.joblib"
        weights_path = self.models_dir / "temporal_autoencoder_weights.joblib"
        thresh_path = self.models_dir / "temporal_autoencoder_threshold.json"

        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Legacy model file missing: {model_path.name}")
            if not pipe_path.exists():
                raise FileNotFoundError(f"Legacy pipeline file missing: {pipe_path.name}")

            self.model = joblib.load(model_path)
            self.pipeline = joblib.load(pipe_path)
            self.feature_names = list(self.pipeline.features)

            if len(self.feature_names) != self.EXPECTED_DIM:
                raise ValueError(
                    f"Unexpected legacy feature count: {len(self.feature_names)} (expected {self.EXPECTED_DIM})"
                )

            if weights_path.exists():
                self.weights = joblib.load(weights_path)
            else:
                self.weights = np.ones(self.EXPECTED_DIM, dtype=float)

            if thresh_path.exists():
                with open(thresh_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.threshold = float(cfg.get("threshold", self.threshold))
                    self.target_fpr = float(cfg.get("target_fpr", self.target_fpr))

            self.is_available = True
            self.load_error = None
            return True
        except Exception as e:
            self.is_available = False
            self.load_error = str(e)
            return False

    def validate_input(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Ensures incoming data strictly matches 32 features and rejects cross-model data."""
        if not self.is_available:
            raise RuntimeError(f"LegacyModelAdapter is unavailable: {self.load_error}")

        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 26:
                raise ValueError(
                    "Dimensionality mismatch: LegacyModelAdapter expects 32 features, but received 26. "
                    "Passing Pure One-Way v2 features into the Legacy v1 pipeline is strictly prohibited."
                )
            missing = [f for f in self.feature_names if f not in data.columns]
            if missing:
                raise ValueError(f"LegacyModelAdapter missing required features: {missing[:5]}...")
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                n_features = len(data)
            else:
                n_features = data.shape[1]
            if n_features != self.EXPECTED_DIM:
                raise ValueError(
                    f"Dimensionality mismatch: LegacyModelAdapter expects {self.EXPECTED_DIM} features, "
                    f"but received {n_features}."
                )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a DataFrame using the legacy 32-feature pipeline."""
        self.validate_input(df)
        return self.pipeline.transform(df)

    def score(self, scaled_data: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        """Computes inverse-variance weighted MSE reconstruction loss."""
        self.validate_input(scaled_data)
        errors = []
        n_samples = len(scaled_data)
        for start in range(0, n_samples, batch_size):
            batch = scaled_data[start : start + batch_size]
            reconstructed = self.model.predict(batch)
            diff_sq = np.square(batch - reconstructed) * self.weights
            errors.append(np.mean(diff_sq, axis=1))
        return np.concatenate(errors) if errors else np.empty(0, dtype=float)

    def predict(
        self, scaled_data: np.ndarray, threshold: Optional[float] = None, batch_size: int = 50_000
    ) -> np.ndarray:
        """Returns binary predictions (1 = Anomaly, 0 = Normal) using decision rule score >= threshold."""
        thresh = threshold if threshold is not None else self.threshold
        scores = self.score(scaled_data, batch_size=batch_size)
        return (scores >= thresh).astype(int)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns safe public metadata without exposing local disk paths."""
        metrics = self.get_evaluation_metrics()
        return {
            "model_id": "legacy_champion",
            "model_name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "status": self.STATUS,
            "is_available": self.is_available,
            "load_error": self.load_error,
            "input_dimensions": self.EXPECTED_DIM,
            "decision_rule": self.DECISION_RULE,
            "threshold": self.threshold,
            "target_calibration_fpr": self.target_fpr,
            "loss_metric": "Inverse-Variance Weighted MSE",
            "features": self.feature_names,
            "metrics": metrics,
        }

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Loads canonical evaluation metrics for legacy champion."""
        eval_path = EVAL_DIR / "temporal_autoencoder_metrics.json"
        if eval_path.exists():
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "detection_rate_pct": 93.08,
            "false_positive_rate_pct": 12.48,
            "precision_pct": 70.46,
            "f1_score": 0.8020,
            "average_precision": 0.8783,
            "roc_auc": 0.9538,
            "tp": 519031,
            "fp": 217598,
            "tn": 1525581,
            "fn": 38615,
        }


class PureOneWayV2Adapter:
    """
    Adapter for the Pure One-Way 26-feature Autoencoder (Experimental / Research Candidate).
    Requires exactly 26 pure forward-only features and unweighted MSE scoring.
    """

    MODEL_NAME: str = "Pure One-Way Autoencoder (v2)"
    MODEL_VERSION: str = "2.0-pure-oneway"
    STATUS: str = "EXPERIMENTAL / RESEARCH CANDIDATE"
    EXPECTED_DIM: int = 26
    DECISION_RULE: str = "score > threshold"

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else V2_MODELS_DIR
        self.model_name: str = self.MODEL_NAME
        self.model_version: str = self.MODEL_VERSION
        self.status: str = self.STATUS
        self.expected_dim: int = self.EXPECTED_DIM
        self.decision_rule: str = self.DECISION_RULE
        self.model = None
        self.pipeline = None
        self.primary_threshold: float = 0.314036
        self.primary_target_fpr: float = 0.002
        self.reference_threshold: float = 0.099018
        self.reference_target_fpr: float = 0.010
        self.feature_names: List[str] = list(PURE_ONE_WAY_FEATURES)
        self.is_available: bool = False
        self.load_error: Optional[str] = None

        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """Loads v2 model, pipeline, manifest, and threshold configuration."""
        model_path = self.artifacts_dir / "model.joblib"
        pipe_path = self.artifacts_dir / "pipeline.joblib"
        manifest_path = self.artifacts_dir / "feature_manifest.json"
        thresh_path = self.artifacts_dir / "threshold_config.json"

        try:
            if not model_path.exists():
                raise FileNotFoundError(f"v2 model file missing: {model_path.name}")
            if not pipe_path.exists():
                raise FileNotFoundError(f"v2 pipeline file missing: {pipe_path.name}")

            self.model = joblib.load(model_path)
            self.pipeline = joblib.load(pipe_path)

            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    if "features" in manifest and manifest["features"]:
                        self.feature_names = manifest["features"]

            if len(self.feature_names) != self.EXPECTED_DIM:
                raise ValueError(
                    f"Unexpected v2 feature count: {len(self.feature_names)} (expected {self.EXPECTED_DIM})"
                )

            if thresh_path.exists():
                with open(thresh_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f).get("calibration", {})
                    self.primary_threshold = float(cfg.get("primary_threshold", self.primary_threshold))
                    self.primary_target_fpr = float(cfg.get("primary_alpha", self.primary_target_fpr))
                    self.reference_threshold = float(cfg.get("reference_threshold", self.reference_threshold))
                    self.reference_target_fpr = float(cfg.get("reference_alpha", self.reference_target_fpr))

            self.is_available = True
            self.load_error = None
            return True
        except Exception as e:
            self.is_available = False
            self.load_error = str(e)
            return False

    def validate_input(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Ensures incoming data strictly matches 26 features and rejects cross-model data."""
        if not self.is_available:
            raise RuntimeError(f"PureOneWayV2Adapter is unavailable: {self.load_error}")

        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 32:
                raise ValueError(
                    "Dimensionality mismatch: PureOneWayV2Adapter expects 26 features, but received 32. "
                    "Passing Legacy v1 features into the Pure One-Way v2 pipeline is strictly prohibited."
                )
            missing = [f for f in self.feature_names if f not in data.columns]
            if missing:
                raise ValueError(f"PureOneWayV2Adapter missing required features: {missing[:5]}...")
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                n_features = len(data)
            else:
                n_features = data.shape[1]
            if n_features != self.EXPECTED_DIM:
                raise ValueError(
                    f"Dimensionality mismatch: PureOneWayV2Adapter expects {self.EXPECTED_DIM} features, "
                    f"but received {n_features}."
                )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a DataFrame using the pure one-way 26-feature pipeline."""
        self.validate_input(df)
        return self.pipeline.transform(df)

    def score(self, scaled_data: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        """Computes unweighted MSE reconstruction loss."""
        self.validate_input(scaled_data)
        errors = []
        n_samples = len(scaled_data)
        for start in range(0, n_samples, batch_size):
            batch = scaled_data[start : start + batch_size]
            reconstructed = self.model.predict(batch)
            diff_sq = np.square(batch - reconstructed)
            errors.append(np.mean(diff_sq, axis=1))
        return np.concatenate(errors) if errors else np.empty(0, dtype=float)

    def predict(
        self,
        scaled_data: np.ndarray,
        threshold: Optional[float] = None,
        operating_point: str = "primary",
        batch_size: int = 50_000,
    ) -> np.ndarray:
        """Returns binary predictions (1 = Anomaly, 0 = Normal) using decision rule score > threshold."""
        if threshold is not None:
            thresh = threshold
        elif operating_point == "reference":
            thresh = self.reference_threshold
        else:
            thresh = self.primary_threshold

        scores = self.score(scaled_data, batch_size=batch_size)
        return (scores > thresh).astype(int)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns safe public metadata without exposing local disk paths."""
        metrics_primary = self.get_evaluation_metrics(operating_point="primary")
        metrics_ref = self.get_evaluation_metrics(operating_point="reference")

        return {
            "model_id": "pure_one_way_v2",
            "model_name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "status": self.STATUS,
            "is_available": self.is_available,
            "load_error": self.load_error,
            "input_dimensions": self.EXPECTED_DIM,
            "decision_rule": self.DECISION_RULE,
            "primary_threshold": self.primary_threshold,
            "primary_target_calibration_fpr": self.primary_target_fpr,
            "reference_threshold": self.reference_threshold,
            "reference_target_calibration_fpr": self.reference_target_fpr,
            "loss_metric": "Unweighted Mean Squared Error (MSE)",
            "features": self.feature_names,
            "metrics": {
                "primary": metrics_primary,
                "reference": metrics_ref,
            },
            "operating_point_note": (
                "The primary v2 operating point reduces false positives compared with v1 but sacrifices recall. "
                "The reference operating point nearly matches v1 recall but has a higher FPR."
            ),
            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
            "compliance_note": "Feature-level compliance only; distinct from validation on a physical one-way capture device.",
            "drift_warning": (
                "Operational FPR on Tuesday–Friday exceeds the Monday calibration target. "
                "The cause has not yet been established."
            ),
        }

    def get_evaluation_metrics(self, operating_point: str = "primary") -> Dict[str, Any]:
        """Loads canonical evaluation metrics for v2 from operational results."""
        results_file = RESULTS_DIR / "overall_metrics.json"
        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    all_metrics = json.load(f).get("models", {})
                    key = "pure_one_way_v2_primary" if operating_point == "primary" else "pure_one_way_v2_reference"
                    if key in all_metrics:
                        m = all_metrics[key]
                        return {
                            "detection_rate_pct": m["rates"]["detection_rate_pct"],
                            "false_positive_rate_pct": m["rates"]["false_positive_rate_pct"],
                            "precision_pct": m["classification_metrics"]["precision_pct"],
                            "f1_score": m["classification_metrics"]["f1_score"],
                            "average_precision": m["continuous_curves"]["pr_auc"],
                            "roc_auc": m["continuous_curves"]["roc_auc"],
                            "tp": m["confusion_matrix"]["tp"],
                            "fp": m["confusion_matrix"]["fp"],
                            "tn": m["confusion_matrix"]["tn"],
                            "fn": m["confusion_matrix"]["fn"],
                        }
            except Exception:
                pass

        if operating_point == "reference":
            return {
                "detection_rate_pct": 93.01,
                "false_positive_rate_pct": 17.35,
                "precision_pct": 63.16,
                "f1_score": 0.7523,
                "average_precision": 0.8687,
                "roc_auc": 0.9478,
                "tp": 518647,
                "fp": 302461,
                "tn": 1440718,
                "fn": 38999,
            }
        else:
            return {
                "detection_rate_pct": 85.38,
                "false_positive_rate_pct": 10.70,
                "precision_pct": 71.86,
                "f1_score": 0.7804,
                "average_precision": 0.8687,
                "roc_auc": 0.9478,
                "tp": 476125,
                "fp": 186489,
                "tn": 1556690,
                "fn": 81521,
            }
