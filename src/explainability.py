"""
src/explainability.py
Explainable AI (XAI) and Root-Cause Attribution Engine for One-Way Threat Detection.

Decomposes the reconstruction error into individual feature residual contributions:
    Attribution_j = w_j * (x_j - x_hat_j)^2
Translates mathematical residuals into actionable SOC forensic incident reports.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd

import sys
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

# Human-readable feature descriptions for SOC analysts
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Temporal streaming context features
    "Rolling_Flow_Velocity_10s": "Rolling 10s forward flow arrival velocity (burst / DDoS flood indicator)",
    "Rolling_Bytes_Velocity_10s": "Rolling 10s forward payload byte volume (volumetric flood indicator)",
    "Rolling_Single_Pkt_Ratio_10s": "Rolling 10s single-packet flow ratio (reconnaissance / sweep indicator)",
    # Forward Flow Features
    "Fwd Packets/s": "Forward packet transmission rate (burst/flooding indicator)",
    "Flow Duration": "Connection hold duration",
    "Fwd IAT Mean": "Mean inter-arrival time between forward packets",
    "Fwd IAT Max": "Maximum pause between packets (slow-rate hold indicator)",
    "Fwd IAT Total": "Total inter-packet arrival time",
    "Init_Win_bytes_forward": "Initial TCP window size (tool fingerprint/scan flag)",
    "Total Fwd Packets": "Volume of packets transmitted in forward direction",
    "Total Length of Fwd Packets": "Total forward payload volume (bytes)",
    "Avg Fwd Segment Size": "Average forward segment size",
    "Fwd Packet Length Max": "Maximum forward packet length observed",
    "Fwd Packet Length Mean": "Mean forward packet length",
    "Fwd Packet Length Std": "Variance in forward packet lengths",
    "Flow Bytes/s": "Forward byte throughput rate",
    "Fwd Header Length": "Forward TCP/IP header length",
    "min_seg_size_forward": "Minimum segment size advertised",
    "SYN Flag Count": "TCP SYN connection attempt flag",
    "PSH Flag Count": "TCP Push flag (data urgency)",
    "ACK Flag Count": "TCP Acknowledgment flag",
    "FIN Flag Count": "TCP Finish flag (connection teardown)",
    "RST Flag Count": "TCP Reset flag (abrupt connection termination)",
    "act_data_pkt_fwd": "Count of packets containing actual data payload",
    "Subflow Fwd Packets": "Subflow forward packet volume",
    "Subflow Fwd Bytes": "Subflow forward byte volume",
}


class ThreatExplainer:
    """Explains why a given flow was flagged as an anomaly using feature residual attribution."""

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

    def explain_flow(self, flow_dict: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Takes a single flow's raw feature dictionary, computes its anomaly score,
        verdict, and returns the top_n most anomalous feature attributions.
        """
        # Convert single dict to DataFrame
        df = pd.DataFrame([flow_dict])
        
        # Ensure all required features are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        # Transform through the pipeline
        X_scaled = self.pipeline.transform(df)

        # Autoencoder reconstruction
        X_rec = self.model.predict(X_scaled)
        
        # Element-wise squared weighted residual
        diff_sq = np.square(X_scaled[0] - X_rec[0]) * self.weights
        total_score = float(np.mean(diff_sq))
        is_anomaly = bool(total_score >= self.threshold)

        # Calculate percentage contribution per feature
        total_residual_sum = np.sum(diff_sq)
        if total_residual_sum > 0:
            pct_contributions = (diff_sq / total_residual_sum) * 100.0
        else:
            pct_contributions = np.zeros_like(diff_sq)

        # Rank features by contribution
        ranked_indices = np.argsort(diff_sq)[::-1]

        top_features = []
        for idx in ranked_indices[:top_n]:
            feat_name = self.feature_names[idx]
            raw_val = float(flow_dict.get(feat_name, 0.0))
            pct = float(pct_contributions[idx])
            desc = FEATURE_DESCRIPTIONS.get(feat_name, feat_name)

            top_features.append({
                "feature": feat_name,
                "description": desc,
                "raw_value": round(raw_val, 4),
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
            "top_attributions": top_features,
        }


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
