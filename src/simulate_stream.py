"""
src/simulate_stream.py
AEGIS Phase 3 Stage 4B.2: Chronological Live Streaming Simulator & v2 Shadow Inference.

Simulates real-world one-way packet capture / flow arrival:
- Plays back real flows from the Tuesday-Friday test set in strict chronological order.
- Dynamically derives causal rolling velocities and pure forward rates via StreamingFeatureEngine.
- Evaluates each incoming flow in real-time using the Champion Deep Autoencoder (v1).
- Runs isolated, observational Pure One-Way Autoencoder (v2) shadow inference.
- Emits backward-compatible SOC alerts and telemetry to outputs/evaluation/live_stream_feed.json.
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

import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    ONE_WAY_FEATURES,
    AUGMENTED_ONE_WAY_FEATURES,
    PURE_ONE_WAY_FEATURES,
    PURE_LOG_FEATURES,
    NON_NEGATIVE_PHYSICAL_FEATURES,
    TCP_WINDOW_FEATURES,
    sanitize_labels,
    add_temporal_velocity_features,
)
from explainability import ThreatExplainer
from model_adapters import LegacyModelAdapter, PureOneWayV2Adapter
from streaming_features import (
    StreamingFeatureEngine,
    StreamingCausalContext,
    prepare_chronological_stream,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "outputs" / "experiment" / "test_tue_fri.parquet"
LIVE_FEED_FILE = ROOT / "outputs" / "evaluation" / "live_stream_feed.json"
LIVE_FEED_FILE.parent.mkdir(parents=True, exist_ok=True)


def score_v2_shadow(
    flow_record: Dict[str, Any],
    v2_adapter: Optional[PureOneWayV2Adapter],
    v2_scaler_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a single flow under Pure One-Way v2 in isolated shadow mode.

    Guarantees:
    - Never raises an unhandled exception to caller.
    - Never modifies production v1 state.
    - If causal context is degraded by buffer overflow, scoring is bypassed and
      status is set to DEGRADED_CONTEXT_OVERFLOW with scores/verdicts set to None.
    - Exactly 26 forward-only features are used.
    - Top drivers is set to None for this stage.
    """
    is_valid = bool(flow_record.get("_is_valid", True))
    context_state = str(flow_record.get("_context_state", "HEALTHY"))
    overflow_events = int(flow_record.get("_overflow_events", 0))

    p_thresh = float(v2_adapter.primary_threshold) if v2_adapter else 0.314036
    r_thresh = float(v2_adapter.reference_threshold) if v2_adapter else 0.099018

    # 1. Check context degradation
    if not is_valid or context_state == "DEGRADED":
        return {
            "status": "DEGRADED_CONTEXT_OVERFLOW",
            "score": None,
            "primary_threshold": p_thresh,
            "primary_is_anomaly": None,
            "primary_verdict": None,
            "reference_threshold": r_thresh,
            "reference_is_anomaly": None,
            "reference_verdict": None,
            "inference_latency_ms": None,
            "latency_ms": None,
            "context_validity": False,
            "context_state": "DEGRADED",
            "overflow_event_count": overflow_events,
            "top_drivers": None,
            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
            "primary": {
                "threshold": p_thresh,
                "is_anomaly": None,
                "verdict": None,
            },
            "reference": {
                "threshold": r_thresh,
                "is_anomaly": None,
                "verdict": None,
            },
        }

    # 2. Check model availability
    if v2_adapter is None or not getattr(v2_adapter, "is_available", False):
        err_msg = getattr(v2_adapter, "load_error", "v2 adapter not initialized")
        return {
            "status": "UNAVAILABLE",
            "error": "MODEL_UNAVAILABLE",
            "error_message": str(err_msg),
            "score": None,
            "primary_threshold": p_thresh,
            "primary_is_anomaly": None,
            "primary_verdict": None,
            "reference_threshold": r_thresh,
            "reference_is_anomaly": None,
            "reference_verdict": None,
            "inference_latency_ms": None,
            "latency_ms": None,
            "context_validity": is_valid,
            "context_state": context_state,
            "overflow_event_count": overflow_events,
            "top_drivers": None,
            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
            "primary": {
                "threshold": p_thresh,
                "is_anomaly": None,
                "verdict": None,
            },
            "reference": {
                "threshold": r_thresh,
                "is_anomaly": None,
                "verdict": None,
            },
        }

    # 3. Perform v2 shadow inference inside isolated try-except block
    try:
        t0_v2 = time.perf_counter()

        # Validate and extract exactly 26 features in canonical order
        v2_raw = {}
        for feat in PURE_ONE_WAY_FEATURES:
            if feat not in flow_record:
                raise KeyError(f"Missing required pure one-way feature: {feat}")
            val = flow_record[feat]
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = np.nan
            v2_raw[feat] = val

        if len(v2_raw) != 26:
            raise ValueError(f"v2 feature count mismatch: got {len(v2_raw)}, expected 26")

        if v2_scaler_params is not None:
            raw_arr = np.array([v2_raw[f] for f in PURE_ONE_WAY_FEATURES], dtype=np.float64)
            for i, f in enumerate(PURE_ONE_WAY_FEATURES):
                if (f in NON_NEGATIVE_PHYSICAL_FEATURES or f in TCP_WINDOW_FEATURES) and raw_arr[i] < 0.0:
                    raw_arr[i] = np.nan
                if np.isinf(raw_arr[i]):
                    raw_arr[i] = np.nan
            for idx in v2_scaler_params["log_indices"]:
                if not np.isnan(raw_arr[idx]) and raw_arr[idx] >= 0.0:
                    raw_arr[idx] = np.log1p(raw_arr[idx])
            nan_m = np.isnan(raw_arr)
            raw_arr[nan_m] = v2_scaler_params["medians"][nan_m]
            scaled = ((raw_arr - v2_scaler_params["means"]) / v2_scaler_params["scales"]).reshape(1, 26)
        else:
            df_26 = pd.DataFrame([v2_raw])
            scaled = v2_adapter.transform(df_26)

        reconstructed = v2_adapter.model.predict(scaled)
        diff_sq = np.square(scaled[0] - reconstructed[0])
        v2_score = float(np.mean(diff_sq))
        t1_v2 = time.perf_counter()
        v2_latency_ms = (t1_v2 - t0_v2) * 1000.0

        p_anomaly = bool(v2_score > p_thresh)
        r_anomaly = bool(v2_score > r_thresh)
        p_verdict = "ALERT" if p_anomaly else "PASS"
        r_verdict = "ALERT" if r_anomaly else "PASS"

        return {
            "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
            "score": round(v2_score, 6),
            "primary_threshold": p_thresh,
            "primary_is_anomaly": p_anomaly,
            "primary_verdict": p_verdict,
            "reference_threshold": r_thresh,
            "reference_is_anomaly": r_anomaly,
            "reference_verdict": r_verdict,
            "inference_latency_ms": round(v2_latency_ms, 4),
            "latency_ms": round(v2_latency_ms, 4),
            "context_validity": True,
            "context_state": "HEALTHY",
            "overflow_event_count": overflow_events,
            "top_drivers": None,
            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
            "primary": {
                "threshold": p_thresh,
                "is_anomaly": p_anomaly,
                "verdict": p_verdict,
            },
            "reference": {
                "threshold": r_thresh,
                "is_anomaly": r_anomaly,
                "verdict": r_verdict,
            },
        }

    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "error": type(e).__name__,
            "error_message": str(e),
            "score": None,
            "primary_threshold": p_thresh,
            "primary_is_anomaly": None,
            "primary_verdict": None,
            "reference_threshold": r_thresh,
            "reference_is_anomaly": None,
            "reference_verdict": None,
            "inference_latency_ms": None,
            "latency_ms": None,
            "context_validity": is_valid,
            "context_state": context_state,
            "overflow_event_count": overflow_events,
            "top_drivers": None,
            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
            "primary": {
                "threshold": p_thresh,
                "is_anomaly": None,
                "verdict": None,
            },
            "reference": {
                "threshold": r_thresh,
                "is_anomaly": None,
                "verdict": None,
            },
        }


def run_stream_simulation(
    speed_eps: float = 20.0,
    max_events: int = 700,
    inject_attacks: bool = True,
    chunk_size: int = 50,
    v1_only: bool = False,
    max_buffer_capacity: int = 10_000,
    test_file: Optional[Path] = None,
    live_feed_file: Optional[Path] = None,
    write_feed: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Simulates a streaming network tap / data diode ingest with chronological playback
    and dual-model shadow inference.

    speed_eps: Events per second to simulate (<= 0 for unthrottled maximum throughput).
    max_events: Total events to simulate.
    inject_attacks: Sample realistic mix of benign and attack bursts.
    chunk_size: Streaming chunk size for multi-batch continuity.
    v1_only: Disable v2 shadow mode and execute Champion v1 alone.
    max_buffer_capacity: Configurable safety ceiling on causal sliding context buffer.
    """
    feed_path = Path(live_feed_file) if live_feed_file else LIVE_FEED_FILE
    parquet_path = Path(test_file) if test_file else TEST_FILE

    if verbose:
        print("=" * 75)
        print("SIH 2026 — REAL-TIME ONE-WAY THREAT DETECTION STREAM SIMULATOR (STAGE 4B.2)")
        rate_str = f"{speed_eps} flows/sec" if speed_eps > 0 else "UNTHROTTLED"
        print(f"Target playback rate: {rate_str} | Chunk size: {chunk_size}")
        print("=" * 75)

    # 1. Initialize Champion v1 Explainer & Autoencoder
    if verbose:
        print("1. Initializing Champion Threat Explainer (v1 32-dim)...")
    explainer = ThreatExplainer()
    if verbose:
        print(f"   Champion v1 threshold: {explainer.threshold:.6f}")

    # 2. Initialize v2 Shadow Adapter & Scaler Params
    v2_adapter = None
    v2_scaler_params = None
    if not v1_only:
        if verbose:
            print("2. Initializing Pure One-Way v2 Shadow Adapter (26-dim)...")
        v2_adapter = PureOneWayV2Adapter()
        if v2_adapter.is_available:
            pipe = v2_adapter.pipeline
            v2_scaler_params = {
                "medians": np.asarray(pipe.imputer.statistics_, dtype=np.float64),
                "means": np.asarray(pipe.scaler.mean_, dtype=np.float64),
                "scales": np.asarray(pipe.scaler.scale_, dtype=np.float64),
                "log_indices": [i for i, f in enumerate(v2_adapter.feature_names) if f in PURE_LOG_FEATURES],
            }
            if verbose:
                print(f"   v2 primary threshold:   {v2_adapter.primary_threshold:.6f}")
                print(f"   v2 reference threshold: {v2_adapter.reference_threshold:.6f}")
        else:
            if verbose:
                print(f"   [!] PureOneWayV2Adapter unavailable: {v2_adapter.load_error}")

    # 3. Initialize Streaming Feature Engine
    v2_pipe = v2_adapter.pipeline if (v2_adapter and v2_adapter.is_available) else None
    engine = StreamingFeatureEngine(
        pipeline=v2_pipe,
        window_seconds=10,
        max_buffer_capacity=max_buffer_capacity,
    )

    # 4. Prepare Chronological Stream
    if verbose:
        print(f"3. Loading and sorting chronological stream from {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    df["Clean_Label"] = sanitize_labels(df["Label"])

    stream_pool = prepare_chronological_stream(
        df,
        max_events=max_events,
        inject_attacks=inject_attacks,
        attack_ratio=0.25,
        random_state=42,
    )

    total_pool_len = len(stream_pool)
    if verbose:
        print(f"   Chronological stream ready: {total_pool_len} flows queued.\n")
        print(f"{'TIME':<10} | {'GROUND TRUTH':<18} | {'CHAMPION (v1)':<22} | {'SHADOW (v2)':<22} | {'TOP FEATURE'}")
        print("-" * 92)

    delay = 1.0 / max(speed_eps, 1.0) if speed_eps > 0 else 0.0
    recent_alerts: List[Dict[str, Any]] = []
    total_processed = 0
    total_anomalies = 0

    v2_primary_anomalies = 0
    v2_reference_anomalies = 0
    v2_degraded_count = 0
    v2_unavailable_count = 0
    v2_both_alert = 0
    v2_both_pass = 0
    v2_latencies: List[float] = []

    def _process_single_flow(flow_row: Dict[str, Any]) -> bool:
        nonlocal total_processed, total_anomalies
        nonlocal v2_primary_anomalies, v2_reference_anomalies
        nonlocal v2_degraded_count, v2_unavailable_count
        nonlocal v2_both_alert, v2_both_pass

        # v1 inference & attribution (preserves exact 32-dim pipeline)
        v1_features = {f: flow_row.get(f, 0.0) for f in explainer.feature_names}
        t0_v1 = time.perf_counter()
        explanation = explainer.explain_flow(v1_features)
        v1_latency_ms = (time.perf_counter() - t0_v1) * 1000.0

        is_alert = explanation["is_anomaly"]
        score = explanation["anomaly_score"]
        top_driver = explanation["top_attributions"][0]["feature"] if explanation["top_attributions"] else "None"
        driver_pct = explanation["top_attributions"][0]["residual_contribution_pct"] if explanation["top_attributions"] else 0.0

        total_processed += 1
        if is_alert:
            total_anomalies += 1
            status_str = f"\033[91mALERT [{explanation['severity']}]\033[0m"
        else:
            status_str = "\033[92mPASS [NORMAL]\033[0m"

        # v2 shadow inference (isolated, non-blocking)
        v2_shadow = None
        if not v1_only and v2_adapter is not None:
            v2_shadow = score_v2_shadow(flow_row, v2_adapter, v2_scaler_params)
            v2_status = v2_shadow.get("status")
            if v2_status == "EXPERIMENTAL / RESEARCH CANDIDATE":
                if v2_shadow.get("latency_ms") is not None:
                    v2_latencies.append(v2_shadow["latency_ms"])
                if v2_shadow.get("primary_is_anomaly"):
                    v2_primary_anomalies += 1
                if v2_shadow.get("reference_is_anomaly"):
                    v2_reference_anomalies += 1
                if is_alert and v2_shadow.get("primary_is_anomaly"):
                    v2_both_alert += 1
                elif (not is_alert) and (not v2_shadow.get("primary_is_anomaly")):
                    v2_both_pass += 1
            elif v2_status == "DEGRADED_CONTEXT_OVERFLOW":
                v2_degraded_count += 1
            elif v2_status == "UNAVAILABLE":
                v2_unavailable_count += 1

        curr_time = time.strftime("%H:%M:%S")
        true_label = str(flow_row.get("Clean_Label", flow_row.get("Label", "UNKNOWN")))

        v2_info = ""
        if v2_shadow and v2_shadow.get("status") == "EXPERIMENTAL / RESEARCH CANDIDATE":
            v2_info = f"v2: {v2_shadow['score']:.4f} [{'ALERT' if v2_shadow['primary_is_anomaly'] else 'PASS'}]"
        elif v2_shadow:
            v2_info = f"v2: {v2_shadow.get('status')[:18]}"

        if verbose:
            print(f"{curr_time:<10} | {true_label:<18} | {score:<7.4f} {status_str:<22} | {v2_info:<22} | {top_driver} ({driver_pct:.1f}%)")

        alert_record = {
            "timestamp": curr_time,
            "flow_index": total_processed,
            "ground_truth": true_label,
            "score": score,
            "threshold": explanation["threshold"],
            "verdict": explanation["verdict"],
            "severity": explanation["severity"],
            "latency_ms": round(v1_latency_ms, 2),
            "top_drivers": explanation["top_attributions"][:3],
            "narrative": explanation["analyst_narrative"],
        }
        if v2_shadow is not None:
            alert_record["v2_shadow"] = v2_shadow

        recent_alerts.append(alert_record)
        if len(recent_alerts) > 150:
            recent_alerts.pop(0)

        # Write latest state to shared live feed for Dashboard consumption
        if write_feed:
            feed_state = {
                "last_updated": curr_time,
                "total_processed": total_processed,
                "total_anomalies": total_anomalies,
                "anomaly_rate_pct": round((total_anomalies / total_processed) * 100, 2) if total_processed > 0 else 0.0,
                "recent_alerts": recent_alerts,
            }
            if not v1_only and v2_adapter is not None:
                concordance = round(((v2_both_alert + v2_both_pass) / max(total_processed, 1)) * 100, 2)
                mean_v2_lat = round(float(np.mean(v2_latencies)), 3) if v2_latencies else 0.0
                feed_state["v2_shadow_summary"] = {
                    "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
                    "primary_anomalies": v2_primary_anomalies,
                    "primary_anomaly_rate_pct": round((v2_primary_anomalies / max(total_processed, 1)) * 100, 2),
                    "reference_anomalies": v2_reference_anomalies,
                    "reference_anomaly_rate_pct": round((v2_reference_anomalies / max(total_processed, 1)) * 100, 2),
                    "degraded_flows": v2_degraded_count,
                    "unavailable_flows": v2_unavailable_count,
                    "total_dropped_overflow_events": engine.total_dropped_overflow_events,
                    "concordance_rate_pct": concordance,
                    "mean_latency_ms": mean_v2_lat,
                }
            with open(feed_path, "w", encoding="utf-8") as f:
                json.dump(feed_state, f, indent=2)

        if max_events and total_processed >= max_events:
            return False

        if delay > 0:
            time.sleep(delay)
        return True

    # 5. Stream Replay Loop across Chunks with flush=False
    n_flows = len(stream_pool)
    actual_chunk_size = max(chunk_size, 1)
    chunks = [stream_pool.iloc[i : i + actual_chunk_size] for i in range(0, n_flows, actual_chunk_size)]

    stop_early = False
    try:
        for chunk_df in chunks:
            if stop_early:
                break
            augmented_chunk = engine.context.process_batch(
                chunk_df,
                causal_mode="simultaneous",
                flush=False,
            )
            for _, row in augmented_chunk.iterrows():
                flow_dict = row.to_dict()
                if not _process_single_flow(flow_dict):
                    stop_early = True
                    break
    finally:
        # 6. Shutdown flush of trailing pending timestamp group
        final_batch = engine.context.flush()
        if not stop_early and len(final_batch) > 0:
            for _, row in final_batch.iterrows():
                flow_dict = row.to_dict()
                if not _process_single_flow(flow_dict):
                    break

    if verbose:
        print("-" * 92)
        print(f"\nStream simulation complete. Processed {total_processed} flows, detected {total_anomalies} anomalies.")
        if not v1_only and v2_adapter is not None:
            print(f"v2 Shadow results: {v2_primary_anomalies} primary alerts, {v2_reference_anomalies} reference alerts.")
            print(f"Context degraded events: {v2_degraded_count} | Dropped overflow events: {engine.total_dropped_overflow_events}")

    summary_result = {
        "total_processed": total_processed,
        "total_anomalies": total_anomalies,
        "anomaly_rate_pct": round((total_anomalies / total_processed) * 100, 2) if total_processed > 0 else 0.0,
        "v2_primary_anomalies": v2_primary_anomalies,
        "v2_reference_anomalies": v2_reference_anomalies,
        "v2_degraded_count": v2_degraded_count,
        "v2_unavailable_count": v2_unavailable_count,
        "total_dropped_overflow_events": engine.total_dropped_overflow_events,
        "recent_alerts": recent_alerts,
    }
    return summary_result


def main():
    parser = argparse.ArgumentParser(description="One-Way Network Traffic Streaming Simulator (AEGIS Phase 3)")
    parser.add_argument("--eps", type=float, default=20.0, help="Events per second (simulation speed, 0 for unthrottled)")
    parser.add_argument("--events", type=int, default=700, help="Total events to simulate (default: 700)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Streaming batch chunk size (default: 50)")
    parser.add_argument("--v1-only", action="store_true", help="Run in Legacy v1 Champion mode only (disable v2 shadow)")
    parser.add_argument("--max-capacity", type=int, default=10_000, help="Context buffer max capacity (default: 10,000)")
    parser.add_argument("--no-attacks", action="store_false", dest="inject_attacks", help="Disable synthetic attack injection")
    args = parser.parse_args()

    run_stream_simulation(
        speed_eps=args.eps,
        max_events=args.events,
        inject_attacks=args.inject_attacks,
        chunk_size=args.chunk_size,
        v1_only=args.v1_only,
        max_buffer_capacity=args.max_capacity,
    )


if __name__ == "__main__":
    main()
