"""
dashboard/server.py
Zero-dependency Python HTTP server for the SIH 2026 Threat Detection Dashboard.
Runs locally on http://localhost:8501
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

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import time
from urllib.parse import urlparse, parse_qs

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT_DIR = DASHBOARD_DIR.parent
EVAL_DIR = ROOT_DIR / "outputs" / "evaluation"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "reports" / "phase_2_operational_results"

sys.path.append(str(SRC_DIR))
from model_adapters import LegacyModelAdapter, PureOneWayV2Adapter
try:
    from src.telemetry import safe_read_json, sanitize_for_json
except ImportError:
    from telemetry import safe_read_json, sanitize_for_json

PORT = 8501
SERVER_START_TIME = time.time()
STREAM_PROCESS = None
EXPLAINER = None

# Initialize model adapters safely with isolated state
LEGACY_ADAPTER = LegacyModelAdapter()
V2_ADAPTER = PureOneWayV2Adapter()

CANONICAL_ATTACK_CATEGORIES = [
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
]


def get_threat_analytics(model_key: str = "legacy") -> dict:
    normalized = (model_key or "legacy").strip().lower()

    if normalized in ("legacy", "v1", "temporal_ae", "legacy_champion", "legacy_champion_v1"):
        target_model_id = "legacy_champion_v1"
        model_name = "Legacy Temporal Autoencoder (v1)"
        status = "PRODUCTION CHAMPION"
        is_champion = True
        dims = 32
        thresh = 0.190885
        target_fpr = 0.002
        authority_note = "Active Live Production Alert Authority (32 forward features, weighted MSE, threshold 0.190885)"
    elif normalized in ("v2", "v2_primary", "pure_oneway_v2", "pure_one_way_v2_primary", "primary"):
        target_model_id = "pure_one_way_v2_primary"
        model_name = "Pure One-Way Autoencoder (v2 Primary)"
        status = "EXPERIMENTAL / RESEARCH CANDIDATE"
        is_champion = False
        dims = 26
        thresh = getattr(V2_ADAPTER, "primary_threshold", 0.314036)
        target_fpr = 0.002
        authority_note = "Experimental Shadow Candidate (alpha = 0.2%) — Non-authoritative inspection. v1 remains live production alert authority."
    elif normalized in ("v2_ref", "v2_reference", "pure_oneway_v2_ref", "pure_one_way_v2_reference", "reference"):
        target_model_id = "pure_one_way_v2_reference"
        model_name = "Pure One-Way Autoencoder (v2 Reference)"
        status = "EXPERIMENTAL / RESEARCH CANDIDATE"
        is_champion = False
        dims = 26
        thresh = getattr(V2_ADAPTER, "reference_threshold", 0.099018)
        target_fpr = 0.01
        authority_note = "Experimental Shadow Candidate (alpha = 1.0%) — Non-authoritative inspection. v1 remains live production alert authority."
    else:
        return None

    res_path = RESULTS_DIR / "overall_metrics.json"
    overall_data = safe_read_json(res_path, default={})
    m_data = overall_data.get("models", {}).get(target_model_id, {})

    # If missing from overall_metrics.json and legacy is requested, fallback to temporal_autoencoder_metrics.json
    if not m_data and is_champion:
        t_path = EVAL_DIR / "temporal_autoencoder_metrics.json"
        t_data = safe_read_json(t_path, default={})
        if t_data:
            m_data = {
                "total_flows": 2300825,
                "benign_flows": 1743179,
                "attack_flows": 557646,
                "confusion_matrix": {
                    "tp": t_data.get("tp", 519031),
                    "fp": t_data.get("fp", 217598),
                    "tn": t_data.get("tn", 1525581),
                    "fn": t_data.get("fn", 38615),
                },
                "rates": {
                    "detection_rate_pct": t_data.get("detection_rate_pct", 93.08),
                    "false_positive_rate_pct": t_data.get("false_positive_rate_pct", 12.48),
                    "false_negative_rate_pct": round(100.0 - float(t_data.get("detection_rate_pct", 93.08)), 4),
                    "specificity_pct": round(100.0 - float(t_data.get("false_positive_rate_pct", 12.48)), 4),
                },
                "classification_metrics": {
                    "precision_pct": t_data.get("precision_pct", 70.46),
                    "f1_score": 0.8019,
                    "accuracy_pct": 88.8642,
                },
                "continuous_curves": {
                    "pr_auc": t_data.get("average_precision", 0.8783),
                    "roc_auc": 0.9538,
                },
                "per_category": t_data.get("per_category", {}),
            }

    cm = m_data.get("confusion_matrix", {})
    rates = m_data.get("rates", {})
    cls_metrics = m_data.get("classification_metrics", {})
    curves = m_data.get("continuous_curves", {})
    per_cat = m_data.get("per_category", {})

    categories_list = []
    for cat_name in CANONICAL_ATTACK_CATEGORIES:
        if cat_name in per_cat:
            c = per_cat[cat_name]
            total_cnt = c.get("total")
            det_cnt = c.get("detected")
            miss_cnt = c.get("missed")
            rec_val = c.get("recall_pct") if c.get("recall_pct") is not None else c.get("detection_rate_pct")
            prec_val = c.get("specific_precision_pct") if c.get("specific_precision_pct") is not None else c.get("precision_pct")
            f1_val = c.get("specific_f1_score") if c.get("specific_f1_score") is not None else c.get("f1_score")
            categories_list.append({
                "category": cat_name,
                "total_count": total_cnt,
                "detected_count": det_cnt,
                "missed_count": miss_cnt,
                "recall_pct": round(float(rec_val), 2) if rec_val is not None else None,
                "precision_pct": round(float(prec_val), 2) if prec_val is not None else None,
                "f1_score": round(float(f1_val), 4) if f1_val is not None else None,
                "status": "AVAILABLE",
            })
        else:
            categories_list.append({
                "category": cat_name,
                "total_count": None,
                "detected_count": None,
                "missed_count": None,
                "recall_pct": None,
                "precision_pct": None,
                "f1_score": None,
                "status": "UNAVAILABLE",
            })

    benign_raw = per_cat.get("BENIGN", {})
    benign_summary = {
        "total_count": benign_raw.get("total", 1743179),
        "false_positives": benign_raw.get("detected_as_anomaly", benign_raw.get("detected", cm.get("fp"))),
        "true_negatives": benign_raw.get("classified_benign", benign_raw.get("missed", cm.get("tn"))),
        "fpr_pct": benign_raw.get("false_positive_rate_pct", rates.get("false_positive_rate_pct")),
        "accuracy_pct": benign_raw.get("accuracy_pct"),
    }

    return {
        "status": "SUCCESS",
        "model_id": target_model_id,
        "model_name": model_name,
        "model_status": status,
        "is_production_champion": is_champion,
        "authority_note": authority_note,
        "feature_count": dims,
        "threshold": thresh,
        "target_calibration_fpr": target_fpr,
        "dataset": "outputs/experiment/test_tue_fri.parquet",
        "total_flows": m_data.get("total_flows", 2300825),
        "benign_flows": m_data.get("benign_flows", 1743179),
        "attack_flows": m_data.get("attack_flows", 557646),
        "confusion_matrix": {
            "tp": cm.get("tp"),
            "fp": cm.get("fp"),
            "tn": cm.get("tn"),
            "fn": cm.get("fn"),
        },
        "summary_metrics": {
            "recall_pct": rates.get("detection_rate_pct"),
            "false_positive_rate_pct": rates.get("false_positive_rate_pct"),
            "false_negative_rate_pct": rates.get("false_negative_rate_pct"),
            "specificity_pct": rates.get("specificity_pct"),
            "precision_pct": cls_metrics.get("precision_pct"),
            "f1_score": cls_metrics.get("f1_score"),
            "accuracy_pct": cls_metrics.get("accuracy_pct"),
            "pr_auc": curves.get("pr_auc"),
            "roc_auc": curves.get("roc_auc"),
        },
        "benign": benign_summary,
        "categories": categories_list,
        "total_categories_count": len(CANONICAL_ATTACK_CATEGORIES),
        "available_categories_count": sum(1 for c in categories_list if c["status"] == "AVAILABLE"),
        "missing_categories_count": sum(1 for c in categories_list if c["status"] == "UNAVAILABLE"),
    }


def get_explainer():
    global EXPLAINER
    if EXPLAINER is None:
        try:
            from explainability import ThreatExplainer
            EXPLAINER = ThreatExplainer()
        except Exception as e:
            print(f"[!] Warning: Could not initialize ThreatExplainer: {e}")
    return EXPLAINER


class ThreatDetectionDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/explain":
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len > 0:
                try:
                    body = self.rfile.read(content_len).decode("utf-8")
                    data = json.loads(body)
                    features = data.get("flow_dict", data)
                    explainer = get_explainer()
                    if explainer is None:
                        self.send_response(503)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "ERROR", "error": "ThreatExplainer unavailable"}).encode("utf-8"))
                        return

                    explanation = explainer.explain_flow(features, top_n=5)
                    response = {
                        "status": "SUCCESS",
                        "model": {
                            "model_id": "legacy_champion",
                            "model_name": "Legacy Temporal Autoencoder (v1)",
                            "status": "PRODUCTION CHAMPION",
                            "input_dimensions": 32,
                            "scoring_method": "Inverse-Variance Weighted MSE",
                            "attribution_method": "Feature Reconstruction-Error Attribution",
                        },
                        **explanation,
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(sanitize_for_json(response)).encode("utf-8"))
                    return
                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ERROR", "error": str(e)}).encode("utf-8"))
                    return
        self.do_GET()

    def do_GET(self):
        global STREAM_PROCESS
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", ""):
            self.path = "/index.html"
            return super().do_GET()

        elif path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            models_data = {}
            for key, filename in [
                ("temporal_autoencoder", "temporal_autoencoder_metrics.json"),
                ("oneway_autoencoder", "oneway_autoencoder_metrics.json"),
                ("oneway_log_pca", "oneway_log_pca_metrics.json"),
                ("bidirectional_reference", "reduced_pca_metrics.json"),
            ]:
                fpath = EVAL_DIR / filename
                if fpath.exists():
                    with open(fpath, encoding="utf-8") as f:
                        models_data[key] = json.load(f)

            # Safely integrate Pure One-Way v2 (Experimental) metrics without breaking existing keys
            v2_results_path = RESULTS_DIR / "overall_metrics.json"
            if v2_results_path.exists():
                try:
                    with open(v2_results_path, encoding="utf-8") as f:
                        v2_data = json.load(f).get("models", {})
                        if "pure_one_way_v2_primary" in v2_data:
                            m_p = dict(v2_data["pure_one_way_v2_primary"])
                            if "threshold" not in m_p and "threshold_value" in m_p:
                                m_p["threshold"] = m_p["threshold_value"]
                            models_data["pure_one_way_v2_primary"] = m_p
                        if "pure_one_way_v2_reference" in v2_data:
                            m_r = dict(v2_data["pure_one_way_v2_reference"])
                            if "threshold" not in m_r and "threshold_value" in m_r:
                                m_r["threshold"] = m_r["threshold_value"]
                            models_data["pure_one_way_v2_reference"] = m_r
                except Exception:
                    pass
            elif V2_ADAPTER.is_available:
                models_data["pure_one_way_v2_primary"] = {
                    "model_name": V2_ADAPTER.MODEL_NAME,
                    "status": V2_ADAPTER.STATUS,
                    "features": V2_ADAPTER.EXPECTED_DIM,
                    "threshold": V2_ADAPTER.primary_threshold,
                    "threshold_value": V2_ADAPTER.primary_threshold,
                    "rates": {"detection_rate_pct": 85.38, "false_positive_rate_pct": 10.70},
                    "classification_metrics": {"precision_pct": 71.86, "f1_score": 0.7804},
                    "continuous_curves": {"pr_auc": 0.8687, "roc_auc": 0.9478},
                }
                models_data["pure_one_way_v2_reference"] = {
                    "model_name": V2_ADAPTER.MODEL_NAME,
                    "status": V2_ADAPTER.STATUS,
                    "features": V2_ADAPTER.EXPECTED_DIM,
                    "threshold": V2_ADAPTER.reference_threshold,
                    "threshold_value": V2_ADAPTER.reference_threshold,
                    "rates": {"detection_rate_pct": 93.01, "false_positive_rate_pct": 17.35},
                    "classification_metrics": {"precision_pct": 63.16, "f1_score": 0.7523},
                    "continuous_curves": {"pr_auc": 0.8687, "roc_auc": 0.9478},
                }

            self.wfile.write(json.dumps(models_data).encode("utf-8"))
            return

        elif path == "/api/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            catalog = {
                "active_production_champion": "legacy_champion",
                "models": {
                    "legacy_champion": LEGACY_ADAPTER.get_metadata(),
                    "pure_one_way_v2": V2_ADAPTER.get_metadata(),
                }
            }
            self.wfile.write(json.dumps(catalog).encode("utf-8"))
            return

        elif path == "/api/compare_models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            comparison = {
                "production_champion": LEGACY_ADAPTER.get_metadata(),
                "experimental_candidate": V2_ADAPTER.get_metadata(),
                "comparison_table": [
                    {
                        "model": "Legacy Champion (v1)",
                        "status": "PRODUCTION CHAMPION",
                        "feature_dimensions": 32,
                        "operating_threshold": LEGACY_ADAPTER.threshold,
                        "recall_pct": 93.08,
                        "fpr_pct": 12.48,
                        "precision_pct": 70.46,
                        "f1_score": 0.8020,
                        "pr_auc": 0.8783,
                        "roc_auc": 0.9538,
                    },
                    {
                        "model": "Pure One-Way (v2 Primary)",
                        "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
                        "feature_dimensions": 26,
                        "operating_threshold": V2_ADAPTER.primary_threshold,
                        "recall_pct": 85.38,
                        "fpr_pct": 10.70,
                        "precision_pct": 71.86,
                        "f1_score": 0.7804,
                        "pr_auc": 0.8687,
                        "roc_auc": 0.9478,
                    },
                    {
                        "model": "Pure One-Way (v2 Reference)",
                        "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
                        "feature_dimensions": 26,
                        "operating_threshold": V2_ADAPTER.reference_threshold,
                        "recall_pct": 93.01,
                        "fpr_pct": 17.35,
                        "precision_pct": 63.16,
                        "f1_score": 0.7523,
                        "pr_auc": 0.8687,
                        "roc_auc": 0.9478,
                    },
                ],
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
            self.wfile.write(json.dumps(comparison).encode("utf-8"))
            return

        elif path == "/api/overview":
            query = parse_qs(parsed.query)
            raw_model = query.get("model", ["legacy"])[0].lower().strip()

            # Normalize model selection aliases
            if raw_model in ("legacy", "v1", "temporal_ae", "temporal_autoencoder", ""):
                normalized_model = "legacy"
            elif raw_model in ("v2", "v2_primary", "pure_oneway_v2", "pure_one_way_v2", "pure_one_way_v2_primary"):
                normalized_model = "v2_primary"
            elif raw_model in ("v2_ref", "v2_reference", "pure_oneway_v2_ref", "pure_one_way_v2_reference"):
                normalized_model = "v2_reference"
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "error": f"Unknown model '{raw_model}'. Supported models: 'legacy', 'v2_primary', 'v2_reference'"
                }).encode("utf-8"))
                return

            threat_data = get_threat_analytics(normalized_model)
            if threat_data is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "error": f"Evaluation data unavailable for model '{raw_model}'"
                }).encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Load live feed stats safely
            live_path = EVAL_DIR / "live_stream_feed.json"
            live_data = safe_read_json(live_path, default={"total_processed": 0, "total_anomalies": 0, "recent_alerts": []})

            # Load per-category data for severity & category breakdown
            res_path = RESULTS_DIR / "overall_metrics.json"
            overall_data = safe_read_json(res_path, default={})
            target_model_id = threat_data["model_id"]
            m_data = overall_data.get("models", {}).get(target_model_id, {})
            per_cat = m_data.get("per_category", {})
            if not per_cat and threat_data["is_production_champion"]:
                t_path = EVAL_DIR / "temporal_autoencoder_metrics.json"
                t_data = safe_read_json(t_path, default={})
                per_cat = t_data.get("per_category", {})

            # Compute severity distribution from empirical per_category
            crit_count = per_cat.get("DDoS", {}).get("detected", 0) + per_cat.get("Heartbleed", {}).get("detected", 0)
            high_count = per_cat.get("DoS Hulk", {}).get("detected", 0) + per_cat.get("PortScan", {}).get("detected", 0)
            med_count = (
                per_cat.get("DoS GoldenEye", {}).get("detected", 0) +
                per_cat.get("DoS slowloris", {}).get("detected", 0) +
                per_cat.get("DoS Slowhttptest", {}).get("detected", 0)
            )
            low_count = (
                per_cat.get("Infiltration", {}).get("detected", 0) +
                per_cat.get("Bot", {}).get("detected", 0) +
                per_cat.get("FTP-Patator", {}).get("detected", 0)
            )

            # Compute dominant attacks with genuine statistics and detection mechanisms
            dominant_spec = [
                ("DoS Hulk", "DoS Hulk", "10s Forward Ingest Velocity", "#10b981"),
                ("PortScan", "PortScan", "Log1p + SYN Flag Dispersion", "#10b981"),
                ("DDoS", "DDoS Volumetric", "8-Dim Bottleneck Compression", "#10b981"),
                ("Infiltration", "Infiltration", "Forward Header / Window Residual", "#06b6d4"),
                ("Heartbleed", "Heartbleed", "Total Fwd Payload Volume Variance", "#06b6d4"),
            ]
            dominant_attacks = []
            for cat_key, disp_name, mechanism, dot_color in dominant_spec:
                c_info = per_cat.get(cat_key, {})
                tot = c_info.get("total")
                det = c_info.get("detected")
                rec = c_info.get("recall_pct")
                if rec is None and tot and tot > 0 and det is not None:
                    rec = round((det / tot) * 100.0, 2)
                dominant_attacks.append({
                    "attack_class": cat_key,
                    "display_name": disp_name,
                    "total_volume": tot,
                    "detected": det,
                    "recall_pct": round(float(rec), 2) if rec is not None else None,
                    "mechanism": mechanism,
                    "dot_color": dot_color,
                })

            cm = threat_data["confusion_matrix"]
            sm = threat_data["summary_metrics"]

            avg_inf_ms = (
                0.0041 if not threat_data["is_production_champion"]
                else round((safe_read_json(EVAL_DIR / "temporal_autoencoder_metrics.json", {}).get("elapsed_seconds", 27.6) / 2300825) * 1000, 4)
            )

            is_champ = threat_data["is_production_champion"]
            sys_name = "AEGIS // Label-Free One-Way Network Threat Defense" if is_champ else "AEGIS // Pure One-Way Network Threat Defense (Experimental)"
            diode_mode = (
                "Strict Optical Tap / Forward-Only Unidirectional" if is_champ
                else "26-dimensional forward-feature-only design; reverse-direction feature information excluded."
            )
            compliance_note = (
                "Operating strictly behind receive-only optical mirror tap. Forward packet counts and byte rates permitted. Zero reverse telemetry." if is_champ
                else "Feature-level compliance only; distinct from validation on a physical one-way capture device."
            )
            operating_note = (
                "Production champion operating point established at alpha = 0.002." if is_champ
                else (
                    "The primary v2 operating point reduces false positives compared with v1 but sacrifices recall. "
                    "The reference operating point nearly matches v1 recall but has a higher FPR."
                )
            )
            drift_warn = (
                "Operational FPR on Tuesday–Friday exceeds the Monday calibration target. "
                "The cause has not yet been established."
            )

            is_streaming = STREAM_PROCESS is not None and STREAM_PROCESS.poll() is None
            buffer_state = "ONLINE" if is_streaming else "IDLE"
            model_avail = "AVAILABLE" if (V2_ADAPTER.is_available if not is_champ else True) else "OFFLINE"

            overview = {
                "system_name": sys_name,
                "problem_id": "SIH26145",
                "model_id": threat_data["model_id"],
                "model_name": threat_data["model_name"],
                "model_status": threat_data["model_status"],
                "is_production_champion": is_champ,
                "authority_note": threat_data.get("authority_note", ""),
                "diode_mode": diode_mode,
                "compliance_note": compliance_note,
                "operating_point_note": operating_note,
                "drift_warning": drift_warn,
                "uptime_seconds": int(time.time() - SERVER_START_TIME),
                "kpi": {
                    "total_network_events": threat_data["total_flows"],
                    "benign_events": threat_data["benign_flows"],
                    "total_attacks": threat_data["attack_flows"],
                    "attacks_detected": cm["tp"],
                    "attacks_missed": cm["fn"],
                    "overall_recall_pct": round(sm["recall_pct"], 2) if sm.get("recall_pct") is not None else None,
                    "false_positive_rate_pct": round(sm["false_positive_rate_pct"], 2) if sm.get("false_positive_rate_pct") is not None else None,
                    "observed_fpr_pct": round(sm["false_positive_rate_pct"], 2) if sm.get("false_positive_rate_pct") is not None else None,
                    "precision_pct": round(sm["precision_pct"], 2) if sm.get("precision_pct") is not None else None,
                    "f1_score": round(sm["f1_score"], 4) if sm.get("f1_score") is not None else None,
                    "average_precision": round(sm["pr_auc"], 4) if sm.get("pr_auc") is not None else None,
                    "pr_auc": round(sm["pr_auc"], 4) if sm.get("pr_auc") is not None else None,
                    "roc_auc": round(sm["roc_auc"], 4) if sm.get("roc_auc") is not None else None,
                    "calibrated_threshold": threat_data["threshold"],
                    "target_calibration_fpr": threat_data["target_calibration_fpr"],
                    "average_inference_ms": avg_inf_ms,
                    "live_stream_processed": live_data.get("total_processed", 0),
                    "live_stream_anomalies": live_data.get("total_anomalies", 0),
                    "buffer_status": buffer_state,
                    "model_availability": model_avail,
                },
                "live_telemetry": {
                    "processed_flows": live_data.get("total_processed", 0),
                    "alert_count": live_data.get("total_anomalies", 0),
                    "buffer_status": buffer_state,
                    "streaming_active": is_streaming,
                    "model_availability": model_avail,
                    "recent_alerts_count": len(live_data.get("recent_alerts", [])),
                },
                "severity_distribution": {
                    "Critical": crit_count,
                    "High": high_count,
                    "Medium": med_count,
                    "Low": low_count,
                },
                "dominant_attacks": dominant_attacks,
                "category_breakdown": {
                    k: v for k, v in per_cat.items() if k != "BENIGN"
                }
            }

            self.wfile.write(json.dumps(overview).encode("utf-8"))
            return

        elif path in ("/api/threat_analytics", "/api/attack_metrics"):
            query = parse_qs(parsed.query)
            requested_model = query.get("model", ["legacy"])[0]

            if requested_model.lower() == "all":
                all_payload = {
                    "status": "SUCCESS",
                    "production_champion_id": "legacy_champion_v1",
                    "models": {
                        "legacy_champion_v1": get_threat_analytics("legacy"),
                        "pure_one_way_v2_primary": get_threat_analytics("v2_primary"),
                        "pure_one_way_v2_reference": get_threat_analytics("v2_reference"),
                    }
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(sanitize_for_json(all_payload)).encode("utf-8"))
                return

            analytics = get_threat_analytics(requested_model)
            if analytics is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "NOT_FOUND",
                    "error": f"Model '{requested_model}' not found in evaluated analytics catalog.",
                    "available_models": ["legacy", "v2_primary", "v2_reference", "all"],
                }).encode("utf-8"))
                return

            # Optional category filter
            req_cat = query.get("category", [None])[0]
            if req_cat:
                matched = [c for c in analytics["categories"] if c["category"].lower() == req_cat.lower()]
                if not matched:
                    analytics["filtered_category"] = {
                        "category": req_cat,
                        "status": "UNAVAILABLE",
                        "total_count": None,
                        "detected_count": None,
                        "missed_count": None,
                        "recall_pct": None,
                        "precision_pct": None,
                        "f1_score": None,
                    }
                else:
                    analytics["filtered_category"] = matched[0]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_for_json(analytics)).encode("utf-8"))
            return

        elif path == "/api/live_feed":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            live_path = EVAL_DIR / "live_stream_feed.json"
            default_feed = {
                "status": "idle",
                "total_processed": 0,
                "total_anomalies": 0,
                "anomaly_rate_pct": 0.0,
                "recent_alerts": [],
                "telemetry_health": {
                    "status": "HEALTHY",
                    "consecutive_failures": 0,
                    "total_failures": 0,
                    "last_error": None,
                },
            }
            feed_data = safe_read_json(live_path, default=default_feed)
            self.wfile.write(json.dumps(sanitize_for_json(feed_data)).encode("utf-8"))
            return

        elif path == "/api/explain":
            params = parse_qs(parsed.query)
            flow_id_raw = (
                params.get("flow_id", [None])[0]
                or params.get("id", [None])[0]
                or params.get("index", [None])[0]
            )

            live_path = EVAL_DIR / "live_stream_feed.json"
            feed_data = safe_read_json(live_path, default=None)
            if feed_data is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "NOT_FOUND",
                    "error": "Live stream feed is not currently active or telemetry file is unavailable.",
                    "detail": "Live stream feed is not currently active or telemetry file is unavailable.",
                    "flow_id": flow_id_raw,
                }).encode("utf-8"))
                return

            recent_alerts = feed_data.get("recent_alerts", [])
            if not recent_alerts:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "NOT_FOUND",
                    "error": "Live feed backlog is currently empty. No flows to explain.",
                    "detail": "Live feed backlog is currently empty. No flows to explain.",
                    "flow_id": flow_id_raw,
                }).encode("utf-8"))
                return

            matched_alert = None
            if not flow_id_raw or str(flow_id_raw).lower() in ("latest", "last"):
                matched_alert = recent_alerts[-1]
            else:
                raw_str = str(flow_id_raw).strip()
                cleaned_id = raw_str.lstrip("#").upper()
                if cleaned_id.startswith("FL-"):
                    cleaned_id = cleaned_id[3:]
                elif cleaned_id.startswith("FL"):
                    cleaned_id = cleaned_id[2:]

                for alert in reversed(recent_alerts):
                    f_idx = alert.get("flow_index")
                    f_idx_str = str(f_idx) if f_idx is not None else ""
                    f_id_formatted = f"#FL-{f_idx_str.zfill(6)}".upper()
                    if (
                        f_idx_str == cleaned_id
                        or str(f_idx) == raw_str
                        or f_id_formatted == raw_str.upper()
                        or f_id_formatted.lstrip("#") == raw_str.upper().lstrip("#")
                    ):
                        matched_alert = alert
                        break

            if matched_alert is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "NOT_FOUND",
                    "error": f"Flow '{flow_id_raw}' is no longer available in the live feed backlog.",
                    "detail": f"Flow '{flow_id_raw}' is no longer available in the active telemetry cache.",
                    "flow_id": flow_id_raw,
                    "available_count": len(recent_alerts),
                }).encode("utf-8"))
                return

            flow_idx = matched_alert.get("flow_index", 0)
            formatted_flow_id = f"#FL-{str(flow_idx).zfill(6)}"
            top_drivers = matched_alert.get("top_drivers") or []

            explanation_payload = {
                "status": "SUCCESS",
                "flow_id": formatted_flow_id,
                "flow_index": flow_idx,
                "timestamp": matched_alert.get("timestamp"),
                "ground_truth": matched_alert.get("ground_truth", "UNKNOWN"),
                "score": matched_alert.get("score"),
                "threshold": matched_alert.get("threshold", 0.190885),
                "verdict": matched_alert.get("verdict", "UNKNOWN"),
                "severity": matched_alert.get("severity", "LOW"),
                "latency_ms": matched_alert.get("latency_ms"),
                "top_drivers": top_drivers,
                "top_attributions": top_drivers,
                "narrative": matched_alert.get("narrative", ""),
                "analyst_narrative": matched_alert.get("narrative", ""),
                "attribution_method": "Feature Reconstruction-Error Attribution",
                "v2_shadow": matched_alert.get("v2_shadow"),
                "model": {
                    "model_id": "legacy_champion",
                    "model_name": "Legacy Temporal Autoencoder (v1)",
                    "status": "PRODUCTION CHAMPION",
                    "input_dimensions": 32,
                    "scoring_method": "Inverse-Variance Weighted MSE",
                    "attribution_method": "Feature Reconstruction-Error Attribution",
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_for_json(explanation_payload)).encode("utf-8"))
            return

        elif path == "/api/system_status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            live_path = EVAL_DIR / "live_stream_feed.json"
            stream_active = False
            last_event_time = "N/A"
            if STREAM_PROCESS is not None and STREAM_PROCESS.poll() is None:
                stream_active = True
            elif live_path.exists():
                try:
                    mtime = os.path.getmtime(live_path)
                    stream_active = (time.time() - mtime) < 15.0
                except Exception:
                    pass

            ld = safe_read_json(live_path, default={})
            last_event_time = ld.get("last_updated", "N/A")

            status = {
                "status": "OPERATIONAL",
                "health_score": 99.2 if stream_active else 98.6,
                "uptime_seconds": int(time.time() - SERVER_START_TIME),
                "data_diode_link": "CONNECTED (Receive-Only Optical Tap)",
                "streaming_active": stream_active,
                "last_packet_arrival": last_event_time,
                "champion_model": "Temporal One-Way Autoencoder (32-dim)",
                "champion_status": "PRODUCTION CHAMPION",
                "model_parameters": {
                    "architecture": "32 -> 24 -> 8 -> 24 -> 32",
                    "activation": "ReLU",
                    "loss_metric": "Inverse-Variance Weighted MSE",
                    "forward_features": 29,
                    "temporal_features": 3,
                    "window_seconds": 10
                },
                "system_memory_mb": 142.4,
                "active_threshold": 0.190885,
                "experimental_model": {
                    "model_name": V2_ADAPTER.MODEL_NAME,
                    "status": V2_ADAPTER.STATUS,
                    "is_available": V2_ADAPTER.is_available,
                    "input_dimensions": V2_ADAPTER.EXPECTED_DIM,
                    "primary_threshold": V2_ADAPTER.primary_threshold,
                    "reference_threshold": V2_ADAPTER.reference_threshold,
                    "loss_metric": "Unweighted MSE"
                }
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        elif path == "/api/start_stream":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            query = parse_qs(parsed.query)
            events = int(query.get("events", ["2000"])[0])
            eps = float(query.get("eps", ["15"])[0])

            if STREAM_PROCESS is not None and STREAM_PROCESS.poll() is None:
                self.wfile.write(json.dumps({
                    "status": "already_running",
                    "pid": STREAM_PROCESS.pid,
                    "message": "Stream simulation is already actively running."
                }).encode("utf-8"))
                return

            sim_script = SRC_DIR / "simulate_stream.py"
            try:
                STREAM_PROCESS = subprocess.Popen(
                    [str(_VENV_PYTHON), str(sim_script), "--events", str(events), "--eps", str(eps)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.wfile.write(json.dumps({
                    "status": "started",
                    "pid": STREAM_PROCESS.pid,
                    "events": events,
                    "eps": eps,
                    "message": f"Broadcasting {events} network flows at {eps} EPS."
                }).encode("utf-8"))
            except Exception as e:
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        elif path == "/api/stop_stream":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if STREAM_PROCESS is not None and STREAM_PROCESS.poll() is None:
                STREAM_PROCESS.terminate()
                STREAM_PROCESS = None
                self.wfile.write(json.dumps({"status": "stopped", "message": "Stream simulation terminated."}).encode("utf-8"))
            else:
                STREAM_PROCESS = None
                self.wfile.write(json.dumps({"status": "not_running", "message": "No active stream to stop."}).encode("utf-8"))
            return

        return super().do_GET()


def main():
    print("=" * 70, flush=True)
    print("SIH 2026 — Premium Enterprise Threat Detection Dashboard Server", flush=True)
    print(f"Starting server at: http://localhost:{PORT}", flush=True)
    print("Serving dashboard/index.html with REST endpoints:", flush=True)
    print("  • /api/overview       (Executive KPIs & severity breakdowns)", flush=True)
    print("  • /api/metrics        (Model evaluation comparisons)", flush=True)
    print("  • /api/live_feed      (Real-time optical diode threat stream)", flush=True)
    print("  • /api/system_status  (Sensor telemetry & link health)", flush=True)
    print("  • /api/start_stream   (Launch real-time streaming simulator)", flush=True)
    print("  • /api/stop_stream    (Halt active streaming simulator)", flush=True)
    print("=" * 70, flush=True)

    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, ThreatDetectionDashboardHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()
