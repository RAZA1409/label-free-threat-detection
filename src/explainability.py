"""
src/explainability.py
Explainable AI (XAI) and Root-Cause Attribution Engine for One-Way Threat Detection.

Decomposes the reconstruction error into individual feature residual contributions:
    Attribution_j = w_j * (x_j - x_hat_j)^2
Translates mathematical residuals into actionable SOC forensic incident reports.
"""

import sys
import os
import subprocess
from pathlib import Path

# Automatically re-execute inside project .venv if invoked with system python
_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = _ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    _res = subprocess.run([str(_VENV_PYTHON)] + sys.argv)
    sys.exit(_res.returncode)

from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    ONE_WAY_FEATURES,
    TEMPORAL_FEATURES,
    AUGMENTED_ONE_WAY_FEATURES,
    clean_feature_frame,
    LogPreprocessingPipeline,
)
# Pickle compatibility alias
sys.modules['__main__'].LogPreprocessingPipeline = LogPreprocessingPipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Human-readable feature descriptions for SOC analysts (All 32 v1 Features Covered)
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Temporal streaming context features (3)
    "Rolling_Flow_Velocity_10s": "Rolling 10s forward flow arrival velocity (burst / DDoS flood indicator)",
    "Rolling_Bytes_Velocity_10s": "Rolling 10s forward payload byte volume (volumetric flood indicator)",
    "Rolling_Single_Pkt_Ratio_10s": "Rolling 10s single-packet flow ratio (reconnaissance / sweep indicator)",
    # Forward Flow Features (20)
    "Total Fwd Packets": "Volume of packets transmitted in forward direction",
    "Total Length of Fwd Packets": "Total forward payload volume (bytes)",
    "act_data_pkt_fwd": "Count of packets containing actual data payload",
    "Subflow Fwd Packets": "Subflow forward packet volume",
    "Subflow Fwd Bytes": "Subflow forward byte volume",
    "Fwd Header Length": "Forward TCP/IP header length",
    "min_seg_size_forward": "Minimum segment size advertised",
    "Avg Fwd Segment Size": "Average forward segment size",
    "Fwd Packet Length Max": "Maximum forward packet length observed",
    "Fwd Packet Length Min": "Minimum forward packet length observed",
    "Fwd Packet Length Mean": "Mean forward packet length",
    "Fwd Packet Length Std": "Variance in forward packet lengths",
    "Flow Duration": "Connection hold duration",
    "Fwd IAT Total": "Total inter-packet arrival time",
    "Fwd IAT Mean": "Mean inter-arrival time between forward packets",
    "Fwd IAT Std": "Standard deviation of forward inter-packet arrival times",
    "Fwd IAT Max": "Maximum pause between packets (slow-rate hold indicator)",
    "Fwd IAT Min": "Minimum pause between forward packets (burst pacing indicator)",
    "Fwd Packets/s": "Forward packet transmission rate (burst/flooding indicator)",
    "Flow Bytes/s": "Forward byte throughput rate",
    # TCP Flags & State (9)
    "Init_Win_bytes_forward": "Initial TCP window size (tool fingerprint/scan flag)",
    "Fwd PSH Flags": "Forward TCP Push flag (immediate data delivery request)",
    "Fwd URG Flags": "Forward TCP Urgent flag (urgent data pointer set)",
    "SYN Flag Count": "TCP SYN connection attempt flag",
    "FIN Flag Count": "TCP Finish flag (connection teardown)",
    "RST Flag Count": "TCP Reset flag (abrupt connection termination)",
    "ACK Flag Count": "TCP Acknowledgment flag",
    "URG Flag Count": "TCP Urgent flag count (anomalous pointer indicator)",
    "PSH Flag Count": "TCP Push flag (data urgency)",
    # Pure forward rate extensions (fallback aliases)
    "Fwd_Bytes_Per_Sec": "Pure forward byte throughput rate",
    "Fwd_Packets_Per_Sec": "Pure forward packet transmission rate",
}


class ThreatExplainer:
    """
    Explains why a given flow was flagged as an anomaly using feature reconstruction-error attribution.
    Decomposes the autoencoder mean squared error (MSE) into per-feature weighted residuals:
        Attribution_j = w_j * (x_j - x_hat_j)^2
    """

    def __init__(self, model_type: str = "temporal"):
        self.model_type = model_type
        if model_type == "temporal":
            self.pipeline_path = MODELS_DIR / "temporal_autoencoder_pipeline.joblib"
            self.model_path = MODELS_DIR / "temporal_autoencoder_model.joblib"
            self.weights_path = MODELS_DIR / "temporal_autoencoder_weights.joblib"
            self.threshold_path = MODELS_DIR / "temporal_autoencoder_threshold.json"
        else:
            self.pipeline_path = MODELS_DIR / "oneway_log_pipeline.joblib"
            self.model_path = MODELS_DIR / "oneway_autoencoder_model.joblib"
            self.weights_path = MODELS_DIR / "oneway_autoencoder_weights.joblib"
            self.threshold_path = MODELS_DIR / "oneway_autoencoder_threshold.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Autoencoder model not found at {self.model_path}.")

        self.pipeline = joblib.load(self.pipeline_path)
        self.model = joblib.load(self.model_path)
        self.feature_names = list(self.pipeline.features)
        self.weights = joblib.load(self.weights_path) if self.weights_path.exists() else np.ones(len(self.feature_names))

        import json
        with open(self.threshold_path, encoding="utf-8") as f:
            thresh_data = json.load(f)
            self.threshold = float(thresh_data["threshold"])

        self.is_available = True
        self.load_error = None

    def explain_flow(self, flow_dict: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Takes a single flow's raw feature dictionary, computes its anomaly score,
        verdict, and returns the top_n most anomalous feature reconstruction-error attributions.
        Handles None, NaN, and infinity values safely without crashing.
        """
        # Defensive input sanitization:
        # Build sanitized feature mapping where missing, None, NaN, or non-numeric values are converted to NaN
        # for pipeline imputation, preventing TypeError or unhandled transform faults.
        clean_row = {}
        for col in self.feature_names:
            raw = flow_dict.get(col) if flow_dict is not None else None
            if raw is None:
                clean_row[col] = np.nan
            else:
                try:
                    num_val = float(raw)
                    if np.isnan(num_val) or np.isinf(num_val):
                        clean_row[col] = np.nan
                    else:
                        clean_row[col] = num_val
                except (ValueError, TypeError):
                    clean_row[col] = np.nan

        df = pd.DataFrame([clean_row])

        # Transform through the fitted preprocessing pipeline (clean, log1p, impute, scale)
        X_scaled = self.pipeline.transform(df)

        # Autoencoder reconstruction
        X_rec = self.model.predict(X_scaled)

        # Element-wise squared weighted residual
        diff_sq = np.square(X_scaled[0] - X_rec[0]) * self.weights
        diff_sq = np.nan_to_num(diff_sq, nan=0.0, posinf=0.0, neginf=0.0)

        total_score = float(np.mean(diff_sq))
        is_anomaly = bool(total_score >= self.threshold)

        # Calculate percentage contribution per feature
        total_residual_sum = float(np.sum(diff_sq))
        if total_residual_sum > 0:
            pct_contributions = (diff_sq / total_residual_sum) * 100.0
        else:
            pct_contributions = np.zeros_like(diff_sq)

        # Rank features by residual contribution
        ranked_indices = np.argsort(diff_sq)[::-1]

        top_features = []
        for idx in ranked_indices[:top_n]:
            feat_name = self.feature_names[idx]
            raw_orig = flow_dict.get(feat_name) if flow_dict is not None else None
            try:
                if raw_orig is None or np.isnan(float(raw_orig)) or np.isinf(float(raw_orig)):
                    safe_raw = 0.0
                else:
                    safe_raw = round(float(raw_orig), 4)
            except (ValueError, TypeError):
                safe_raw = 0.0

            pct = float(pct_contributions[idx])
            desc = FEATURE_DESCRIPTIONS.get(feat_name, feat_name)

            top_features.append({
                "feature": feat_name,
                "description": desc,
                "raw_value": safe_raw,
                "residual_contribution_pct": round(pct, 2),
                "attribution_score": round(float(diff_sq[idx]), 4),
            })

        # Generate human-readable analyst narrative
        if is_anomaly:
            primary_cause = top_features[0]["description"]
            secondary_cause = top_features[1]["description"] if len(top_features) > 1 else ""
            summary = (
                f"ALERT: Flow flagged as anomalous (Score {total_score:.4f} exceeds threshold {self.threshold:.4f}). "
                f"Primary deviation: {primary_cause} ({top_features[0]['residual_contribution_pct']}% of residual). "
            )
            if secondary_cause:
                summary += f"Secondary factor: {secondary_cause} ({top_features[1]['residual_contribution_pct']}%)."
        else:
            summary = (
                f"NORMAL: Flow conforms to baseline normal traffic profile "
                f"(Score {total_score:.4f} is within threshold {self.threshold:.4f})."
            )

        return {
            "anomaly_score": round(total_score, 6),
            "threshold": round(self.threshold, 6),
            "is_anomaly": is_anomaly,
            "verdict": "ANOMALOUS (THREAT DETECTED)" if is_anomaly else "BENIGN (NORMAL TRAFFIC)",
            "severity": "HIGH" if total_score > self.threshold * 2 else ("MEDIUM" if is_anomaly else "LOW"),
            "analyst_narrative": summary,
            "narrative": summary,
            "top_attributions": top_features,
            "attribution_method": "Feature Reconstruction-Error Attribution",
        }


def generate_soc_narrative(
    flow_id: str = "#FL-000000",
    ground_truth: str = "BENIGN",
    is_anomaly: bool = False,
    anomaly_score: float = 0.0,
    threshold: float = 0.190885,
    top_attributions: list = None,
) -> str:
    """Generate human-readable SOC analyst narrative from flow attribution."""
    if top_attributions is None:
        top_attributions = []

    if is_anomaly:
        primary = top_attributions[0] if top_attributions else {"feature": "Unknown", "residual_contribution_pct": 0.0}
        prim_desc = primary.get("description", primary.get("feature", "Forward traffic deviation"))
        prim_pct = primary.get("residual_contribution_pct", 0.0)
        gt_context = f" ({ground_truth})" if ground_truth and ground_truth != "UNKNOWN" else ""
        narrative = (
            f"ALERT: Flow {flow_id}{gt_context} flagged as anomalous (Score {anomaly_score:.4f} exceeds threshold {threshold:.4f}). "
            f"Primary deviation: {prim_desc} ({prim_pct}% of residual)."
        )
        if len(top_attributions) > 1:
            sec = top_attributions[1]
            sec_desc = sec.get("description", sec.get("feature", "Secondary deviation"))
            sec_pct = sec.get("residual_contribution_pct", 0.0)
            narrative += f" Secondary factor: {sec_desc} ({sec_pct}%)."
        return narrative
    else:
        return (
            f"NORMAL: Flow {flow_id} conforms to baseline normal traffic profile "
            f"(Score {anomaly_score:.4f} is within threshold {threshold:.4f})."
        )


def demo():
    print("Testing ThreatExplainer with sample flows...")
    explainer = ThreatExplainer()
    print(f"Loaded explainer with threshold: {explainer.threshold:.6f}")

    # Test sample PortScan-like flow
    portscan_flow = {
        "Total Fwd Packets": 1,
        "Total Length of Fwd Packets": 0,
        "act_data_pkt_fwd": 0,
        "Subflow Fwd Packets": 1,
        "Subflow Fwd Bytes": 0,
        "Fwd Header Length": 40,
        "min_seg_size_forward": 40,
        "Avg Fwd Segment Size": 0,
        "Fwd Packet Length Max": 0,
        "Fwd Packet Length Min": 0,
        "Fwd Packet Length Mean": 0,
        "Fwd Packet Length Std": 0,
        "Flow Duration": 47,
        "Fwd IAT Total": 0,
        "Fwd IAT Mean": 0,
        "Fwd IAT Std": 0,
        "Fwd IAT Max": 0,
        "Fwd IAT Min": 0,
        "Fwd Packets/s": 21276.59,
        "Flow Bytes/s": 0,
        "Init_Win_bytes_forward": 29200,
        "Fwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "SYN Flag Count": 1,
        "FIN Flag Count": 0,
        "RST Flag Count": 0,
        "ACK Flag Count": 0,
        "URG Flag Count": 0,
        "PSH Flag Count": 0,
    }

    result = explainer.explain_flow(portscan_flow)
    print("\n--- PortScan Sample Result ---")
    print(f"Verdict:   {result['verdict']}")
    print(f"Score:     {result['anomaly_score']} (Thresh: {result['threshold']})")
    print(f"Narrative: {result['analyst_narrative']}")
    print("Top Feature Drivers:")
    for feat in result["top_attributions"]:
        print(f"  • {feat['feature']:<25} -> {feat['residual_contribution_pct']}% | {feat['description']}")


if __name__ == "__main__":
    demo()
