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
import math
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
    FlowQuarantineManager,
    validate_and_sanitize_flow_record,
    prepare_chronological_stream,
)
try:
    from src.telemetry import (
        TelemetryManager,
        atomic_write_json,
        safe_read_json,
        sanitize_for_json,
    )
    from src.checkpoint import (
        CheckpointManager,
        RecoveryStatus,
        RecoveryResult,
        AlertIdempotencyTracker,
    )
except ImportError:
    from telemetry import (
        TelemetryManager,
        atomic_write_json,
        safe_read_json,
        sanitize_for_json,
    )
    from checkpoint import (
        CheckpointManager,
        RecoveryStatus,
        RecoveryResult,
        AlertIdempotencyTracker,
    )

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "outputs" / "experiment" / "test_tue_fri.parquet"
LIVE_FEED_FILE = ROOT / "outputs" / "evaluation" / "live_stream_feed.json"
LIVE_FEED_FILE.parent.mkdir(parents=True, exist_ok=True)


def _parse_valid_threshold(val: Any, name: str) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Safely validates that a threshold value is a real, non-negative, finite number.
    Rejects None, booleans, NaN, Inf, non-numeric strings, and negative values.

    Returns:
        (is_valid: bool, parsed_value: Optional[float], error_message: Optional[str])
    """
    if val is None:
        return False, None, f"{name} threshold is None"
    if isinstance(val, bool):
        return False, None, f"{name} threshold cannot be boolean ({val})"
    try:
        f_val = float(val)
    except (ValueError, TypeError) as e:
        return False, None, f"{name} threshold could not be converted to float: {e}"
    if math.isnan(f_val):
        return False, None, f"{name} threshold is NaN"
    if math.isinf(f_val):
        return False, None, f"{name} threshold is infinite ({f_val})"
    if f_val < 0.0:
        return False, None, f"{name} threshold is negative ({f_val})"
    return True, f_val, None


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
    try:
        is_valid = bool(flow_record.get("_is_valid", True))
        context_state = str(flow_record.get("_context_state", "HEALTHY"))
        overflow_events = int(flow_record.get("_overflow_events", 0))

        # 1. Check model availability first
        if v2_adapter is None:
            return {
                "status": "UNAVAILABLE",
                "error": "MODEL_UNAVAILABLE",
                "error_message": "v2 adapter not initialized",
                "score": None,
                "primary_threshold": 0.314036,
                "primary_is_anomaly": None,
                "primary_verdict": None,
                "reference_threshold": 0.099018,
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
                    "threshold": 0.314036,
                    "is_anomaly": None,
                    "verdict": None,
                },
                "reference": {
                    "threshold": 0.099018,
                    "is_anomaly": None,
                    "verdict": None,
                },
            }

        if not getattr(v2_adapter, "is_available", False):
            err_msg = getattr(v2_adapter, "load_error", "v2 adapter not available")
            p_val = getattr(v2_adapter, "primary_threshold", 0.314036)
            p_ok, p_thresh, _ = _parse_valid_threshold(p_val, "primary")
            if not p_ok:
                p_thresh = 0.314036
            r_val = getattr(v2_adapter, "reference_threshold", 0.099018)
            r_ok, r_thresh, _ = _parse_valid_threshold(r_val, "reference")
            if not r_ok:
                r_thresh = 0.099018
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

        # 2. Strict threshold extraction and validation for active adapter
        if not hasattr(v2_adapter, "primary_threshold"):
            return {
                "status": "UNAVAILABLE",
                "error": "INVALID_THRESHOLD",
                "error_message": "v2 adapter missing 'primary_threshold' attribute",
                "score": None,
                "primary_threshold": None,
                "primary_is_anomaly": None,
                "primary_verdict": None,
                "reference_threshold": None,
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
                    "threshold": None,
                    "is_anomaly": None,
                    "verdict": None,
                },
                "reference": {
                    "threshold": None,
                    "is_anomaly": None,
                    "verdict": None,
                },
            }

        p_raw = getattr(v2_adapter, "primary_threshold")
        p_ok, p_thresh, p_err = _parse_valid_threshold(p_raw, "primary")
        if not p_ok:
            return {
                "status": "UNAVAILABLE",
                "error": "INVALID_THRESHOLD",
                "error_message": p_err,
                "score": None,
                "primary_threshold": None,
                "primary_is_anomaly": None,
                "primary_verdict": None,
                "reference_threshold": None,
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
                    "threshold": None,
                    "is_anomaly": None,
                    "verdict": None,
                },
                "reference": {
                    "threshold": None,
                    "is_anomaly": None,
                    "verdict": None,
                },
            }

        if not hasattr(v2_adapter, "reference_threshold"):
            return {
                "status": "UNAVAILABLE",
                "error": "INVALID_THRESHOLD",
                "error_message": "v2 adapter missing 'reference_threshold' attribute",
                "score": None,
                "primary_threshold": p_thresh,
                "primary_is_anomaly": None,
                "primary_verdict": None,
                "reference_threshold": None,
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
                    "threshold": None,
                    "is_anomaly": None,
                    "verdict": None,
                },
            }

        r_raw = getattr(v2_adapter, "reference_threshold")
        r_ok, r_thresh, r_err = _parse_valid_threshold(r_raw, "reference")
        if not r_ok:
            return {
                "status": "UNAVAILABLE",
                "error": "INVALID_THRESHOLD",
                "error_message": r_err,
                "score": None,
                "primary_threshold": p_thresh,
                "primary_is_anomaly": None,
                "primary_verdict": None,
                "reference_threshold": None,
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
                    "threshold": None,
                    "is_anomaly": None,
                    "verdict": None,
                },
            }

        # 3. Check context degradation
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

        # 4. Perform v2 shadow inference inside isolated execution block
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
        safe_p = p_thresh if "p_thresh" in locals() and p_thresh is not None else None
        safe_r = r_thresh if "r_thresh" in locals() and r_thresh is not None else None
        return {
            "status": "UNAVAILABLE",
            "error": type(e).__name__,
            "error_message": str(e),
            "score": None,
            "primary_threshold": safe_p,
            "primary_is_anomaly": None,
            "primary_verdict": None,
            "reference_threshold": safe_r,
            "reference_is_anomaly": None,
            "reference_verdict": None,
            "inference_latency_ms": None,
            "latency_ms": None,
            "context_validity": flow_record.get("_is_valid", True) if isinstance(flow_record, dict) else False,
            "context_state": flow_record.get("_context_state", "UNKNOWN") if isinstance(flow_record, dict) else "UNKNOWN",
            "overflow_event_count": flow_record.get("_overflow_events", 0) if isinstance(flow_record, dict) else 0,
            "top_drivers": None,
            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
            "primary": {
                "threshold": safe_p,
                "is_anomaly": None,
                "verdict": None,
            },
            "reference": {
                "threshold": safe_r,
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
    telemetry_throttle_sec: float = 0.25,
    telemetry_throttle_flows: int = 20,
    checkpoint_file: Optional[Union[str, Path]] = None,
    auto_restore: bool = True,
    checkpoint_interval_events: Optional[int] = None,
    save_checkpoint_on_shutdown: bool = True,
    stop_after_events: Optional[int] = None,
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
    telemetry_throttle_sec: Bounded interval between non-critical telemetry file writes.
    telemetry_throttle_flows: Flow count threshold between non-critical telemetry file writes.
    checkpoint_file: Path to versioned checkpoint file for persistence and recovery.
    auto_restore: Whether to automatically restore from checkpoint_file if it exists.
    checkpoint_interval_events: Save checkpoint every N flows (None for shutdown only).
    save_checkpoint_on_shutdown: Save final checkpoint upon stream completion/shutdown.
    stop_after_events: Optional early termination limit to test restart recovery midway.
    """
    feed_path = Path(live_feed_file) if live_feed_file else LIVE_FEED_FILE
    parquet_path = Path(test_file) if test_file else TEST_FILE

    telemetry_mgr = TelemetryManager(
        feed_path=feed_path if write_feed else None,
        enabled=write_feed,
        throttle_interval_sec=telemetry_throttle_sec,
        throttle_event_count=telemetry_throttle_flows,
    )

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
            pipe = getattr(v2_adapter, "pipeline", None)
            if pipe is not None and hasattr(pipe, "imputer") and hasattr(pipe, "scaler"):
                v2_scaler_params = {
                    "medians": np.asarray(pipe.imputer.statistics_, dtype=np.float64),
                    "means": np.asarray(pipe.scaler.mean_, dtype=np.float64),
                    "scales": np.asarray(pipe.scaler.scale_, dtype=np.float64),
                    "log_indices": [i for i, f in enumerate(v2_adapter.feature_names) if f in PURE_LOG_FEATURES],
                }
            if verbose:
                p_raw = getattr(v2_adapter, "primary_threshold", None)
                r_raw = getattr(v2_adapter, "reference_threshold", None)
                p_disp = f"{p_raw:.6f}" if isinstance(p_raw, (int, float)) and not isinstance(p_raw, bool) else "None"
                r_disp = f"{r_raw:.6f}" if isinstance(r_raw, (int, float)) and not isinstance(r_raw, bool) else "None"
                print(f"   v2 primary threshold:   {p_disp}")
                print(f"   v2 reference threshold: {r_disp}")
        else:
            if verbose:
                print(f"   [!] PureOneWayV2Adapter unavailable: {v2_adapter.load_error}")

    # 3. Prepare Chronological Stream
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

    # Derive trusted replay horizon from complete stream pool before chunking
    trusted_horizon = stream_pool.attrs.get("trusted_horizon")
    if trusted_horizon is None and len(stream_pool) > 0:
        trusted_horizon = stream_pool["Timestamp"].max() + pd.Timedelta(seconds=10.0)

    # 4. Initialize Streaming Feature Engine with trusted horizon
    v2_pipe = v2_adapter.pipeline if (v2_adapter and v2_adapter.is_available) else None
    quarantine_manager = FlowQuarantineManager(max_diagnostic_samples=100)
    engine = StreamingFeatureEngine(
        pipeline=v2_pipe,
        window_seconds=10,
        max_buffer_capacity=max_buffer_capacity,
        clock_mode="event_time",
        max_timestamp_horizon=trusted_horizon,
        quarantine_manager=quarantine_manager,
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

    checkpoint_mgr: Optional[CheckpointManager] = None
    recovery_result: Optional[RecoveryResult] = None
    if checkpoint_file:
        checkpoint_mgr = CheckpointManager(
            checkpoint_path=checkpoint_file,
            max_buffer_capacity=max_buffer_capacity,
        )
        if auto_restore:
            recovery_result = checkpoint_mgr.restore_checkpoint(
                context=engine.context,
                telemetry_mgr=telemetry_mgr,
            )
            if recovery_result.status == RecoveryStatus.SUCCESS:
                if verbose:
                    print(f"   [+] Restored checkpoint: {recovery_result.restored_buffer_entries} buffer entries, {recovery_result.restored_alerts_count} alerts.")
                total_processed = recovery_result.restored_flow_index
                total_anomalies = recovery_result.restored_anomalies
                v2_metrics = recovery_result.restored_v2_metrics
                v2_primary_anomalies = v2_metrics.get("primary_anomalies", 0)
                v2_reference_anomalies = v2_metrics.get("reference_anomalies", 0)
                v2_degraded_count = v2_metrics.get("degraded_count", 0)
                v2_unavailable_count = v2_metrics.get("unavailable_count", 0)
                v2_both_alert = v2_metrics.get("both_alert", 0)
                v2_both_pass = v2_metrics.get("both_pass", 0)

                if recovery_result.restored_alerts:
                    for al in recovery_result.restored_alerts:
                        if al not in recent_alerts:
                            recent_alerts.append(al)
                    if len(recent_alerts) > 150:
                        recent_alerts = recent_alerts[-150:]
                elif telemetry_mgr is not None and hasattr(telemetry_mgr, "_backlog_alerts"):
                    for al in telemetry_mgr._backlog_alerts:
                        if al not in recent_alerts:
                            recent_alerts.append(al)
                            if len(recent_alerts) > 150:
                                recent_alerts.pop(0)

                # Resume stream pool from total_processed if applicable
                if total_processed > 0 and total_processed < len(stream_pool):
                    stream_pool = stream_pool.iloc[total_processed:].copy()
                    if verbose:
                        print(f"   [+] Resuming chronological stream from index {total_processed} ({len(stream_pool)} flows remaining).")
            elif recovery_result.status == RecoveryStatus.SKIPPED:
                if verbose:
                    print("   [*] No checkpoint found. Starting clean stream.")
            else:
                if verbose:
                    print(f"   [!] Checkpoint rejected ({recovery_result.status.value}): {recovery_result.error_message}. Starting clean stream.")

    def _build_feed_state(curr_timestamp: Optional[str] = None) -> Dict[str, Any]:
        ts = curr_timestamp or time.strftime("%H:%M:%S")
        st = {
            "last_updated": ts,
            "total_processed": total_processed,
            "total_anomalies": total_anomalies,
            "anomaly_rate_pct": round((total_anomalies / total_processed) * 100, 2) if total_processed > 0 else 0.0,
            "quarantine_summary": quarantine_manager.get_metrics(),
            "recent_alerts": recent_alerts,
        }
        if not v1_only and v2_adapter is not None:
            concordance = round(((v2_both_alert + v2_both_pass) / max(total_processed, 1)) * 100, 2)
            mean_v2_lat = round(float(np.mean(v2_latencies)), 3) if v2_latencies else 0.0
            st["v2_shadow_summary"] = {
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
        return st

    def _process_single_flow(flow_row: Dict[str, Any]) -> bool:
        nonlocal total_processed, total_anomalies
        nonlocal v2_primary_anomalies, v2_reference_anomalies
        nonlocal v2_degraded_count, v2_unavailable_count
        nonlocal v2_both_alert, v2_both_pass

        # v1 inference & attribution (preserves exact 32-dim pipeline)
        v1_features = {f: flow_row.get(f, 0.0) for f in explainer.feature_names}
        t0_v1 = time.perf_counter()
        v1_error = None
        try:
            explanation = explainer.explain_flow(v1_features)
            v1_latency_ms = (time.perf_counter() - t0_v1) * 1000.0
            is_alert = explanation["is_anomaly"]
            score = explanation["anomaly_score"]
            threshold = explanation["threshold"]
            verdict = explanation["verdict"]
            severity = explanation["severity"]
            top_drivers = explanation["top_attributions"][:3]
            top_driver = explanation["top_attributions"][0]["feature"] if explanation["top_attributions"] else "None"
            driver_pct = explanation["top_attributions"][0]["residual_contribution_pct"] if explanation["top_attributions"] else 0.0
            narrative = explanation["analyst_narrative"]
        except Exception as e:
            v1_latency_ms = (time.perf_counter() - t0_v1) * 1000.0
            v1_error = f"{type(e).__name__}: {str(e)}"
            is_alert = False
            score = None
            threshold = getattr(explainer, "threshold", 0.190885)
            verdict = "UNAVAILABLE (PROCESSING_ERROR)"
            severity = "UNAVAILABLE"
            top_drivers = None
            top_driver = "None"
            driver_pct = 0.0
            narrative = f"Champion v1 inference failure: {v1_error}"

        total_processed += 1
        if is_alert:
            total_anomalies += 1
            status_str = f"\033[91mALERT [{severity}]\033[0m"
        elif v1_error is not None:
            status_str = "\033[93mUNAVAILABLE (PROCESSING_ERROR)\033[0m"
        else:
            status_str = "\033[92mPASS [NORMAL]\033[0m"

        # v2 shadow inference (isolated, non-blocking)
        v2_shadow = None
        if not v1_only and v2_adapter is not None:
            try:
                v2_shadow = score_v2_shadow(flow_row, v2_adapter, v2_scaler_params)
            except Exception as e:
                v2_shadow = {
                    "status": "UNAVAILABLE",
                    "error": f"CALLER_FAULT_{type(e).__name__}",
                    "error_message": str(e),
                    "score": None,
                    "primary_threshold": None,
                    "primary_is_anomaly": None,
                    "primary_verdict": None,
                    "reference_threshold": None,
                    "reference_is_anomaly": None,
                    "reference_verdict": None,
                    "inference_latency_ms": None,
                    "latency_ms": None,
                    "context_validity": bool(flow_row.get("_is_valid", True)),
                    "context_state": str(flow_row.get("_context_state", "HEALTHY")),
                    "overflow_event_count": int(flow_row.get("_overflow_events", 0)),
                    "top_drivers": None,
                    "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
                    "primary": {"threshold": None, "is_anomaly": None, "verdict": None},
                    "reference": {"threshold": None, "is_anomaly": None, "verdict": None},
                }
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
            score_disp = f"{score:<7.4f}" if score is not None else "None   "
            print(f"{curr_time:<10} | {true_label:<18} | {score_disp} {status_str:<22} | {v2_info:<22} | {top_driver} ({driver_pct:.1f}%)")

        alert_record = {
            "timestamp": curr_time,
            "flow_index": total_processed,
            "ground_truth": true_label,
            "score": score,
            "threshold": threshold,
            "verdict": verdict,
            "severity": severity,
            "latency_ms": round(v1_latency_ms, 2),
            "top_drivers": top_drivers,
            "narrative": narrative,
        }
        if v1_error is not None:
            alert_record["v1_error"] = v1_error
        if v2_shadow is not None:
            alert_record["v2_shadow"] = v2_shadow

        recent_alerts.append(alert_record)
        if len(recent_alerts) > 150:
            recent_alerts.pop(0)

        # Write latest state to shared live feed for Dashboard consumption
        telemetry_mgr.record_event(is_critical=is_alert)
        if write_feed:
            feed_state = _build_feed_state(curr_time)
            telemetry_mgr.update_and_write(
                feed_state,
                force=is_alert,
                recent_alert=alert_record if is_alert else None,
            )

        if checkpoint_mgr is not None and is_alert:
            checkpoint_mgr.idempotency_tracker.register(f"{total_processed}:{curr_time}")

        if (
            checkpoint_mgr is not None
            and checkpoint_interval_events is not None
            and checkpoint_interval_events > 0
            and total_processed % checkpoint_interval_events == 0
        ):
            checkpoint_mgr.save_checkpoint(
                context=engine.context,
                telemetry_mgr=telemetry_mgr,
                extra_state={
                    "total_processed": total_processed,
                    "total_anomalies": total_anomalies,
                    "recent_alerts": recent_alerts,
                    "v2_primary_anomalies": v2_primary_anomalies,
                    "v2_reference_anomalies": v2_reference_anomalies,
                    "v2_degraded_count": v2_degraded_count,
                    "v2_unavailable_count": v2_unavailable_count,
                    "v2_both_alert": v2_both_alert,
                    "v2_both_pass": v2_both_pass,
                },
            )

        effective_limit = stop_after_events if stop_after_events is not None else max_events
        if effective_limit and total_processed >= effective_limit:
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

            valid_rows = []
            for _, row in chunk_df.iterrows():
                row_dict = row.to_dict()
                is_val, reason, sanitized_row = validate_and_sanitize_flow_record(
                    row_dict,
                    clock_mode="event_time",
                    max_timestamp_horizon=trusted_horizon,
                )
                if is_val:
                    valid_rows.append(sanitized_row)
                else:
                    # Malformed record quarantined: never enters causal context
                    quarantine_manager.quarantine(row_dict, reason)
                    total_processed += 1
                    curr_time = time.strftime("%H:%M:%S")
                    true_label = str(row_dict.get("Clean_Label", row_dict.get("Label", "MALFORMED")))

                    if verbose:
                        print(f"{curr_time:<10} | {true_label:<18} | {'N/A':<7} \033[93mQUARANTINED [{reason[:12]}]\033[0m | {'v2: UNAVAILABLE':<22} | None")

                    q_alert = {
                        "timestamp": curr_time,
                        "flow_index": total_processed,
                        "ground_truth": true_label,
                        "score": None,
                        "threshold": getattr(explainer, "threshold", 0.190885),
                        "verdict": "QUARANTINED",
                        "severity": "UNAVAILABLE",
                        "latency_ms": 0.0,
                        "top_drivers": None,
                        "narrative": f"Flow quarantined before causal context: {reason}",
                        "quarantine_reason": reason,
                    }
                    if not v1_only and v2_adapter is not None:
                        q_p_raw = getattr(v2_adapter, "primary_threshold", None)
                        q_r_raw = getattr(v2_adapter, "reference_threshold", None)
                        q_p_safe = q_p_raw if isinstance(q_p_raw, (int, float)) and not isinstance(q_p_raw, bool) else 0.314036
                        q_r_safe = q_r_raw if isinstance(q_r_raw, (int, float)) and not isinstance(q_r_raw, bool) else 0.099018
                        q_alert["v2_shadow"] = {
                            "status": "UNAVAILABLE",
                            "error": "QUARANTINED_INPUT",
                            "error_message": f"Input quarantined: {reason}",
                            "score": None,
                            "primary_threshold": q_p_safe,
                            "primary_is_anomaly": None,
                            "primary_verdict": None,
                            "reference_threshold": q_r_safe,
                            "reference_is_anomaly": None,
                            "reference_verdict": None,
                            "inference_latency_ms": None,
                            "latency_ms": None,
                            "context_validity": False,
                            "context_state": "QUARANTINED",
                            "overflow_event_count": engine.total_dropped_overflow_events,
                            "top_drivers": None,
                            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
                            "primary": {
                                "threshold": q_p_safe,
                                "is_anomaly": None,
                                "verdict": None,
                            },
                            "reference": {
                                "threshold": q_r_safe,
                                "is_anomaly": None,
                                "verdict": None,
                            },
                        }
                    recent_alerts.append(q_alert)
                    if len(recent_alerts) > 150:
                        recent_alerts.pop(0)

                    telemetry_mgr.record_event(is_critical=True)
                    if write_feed:
                        feed_state = _build_feed_state(curr_time)
                        telemetry_mgr.update_and_write(
                            feed_state,
                            force=True,
                            recent_alert=q_alert,
                        )

                    if checkpoint_mgr is not None:
                        checkpoint_mgr.idempotency_tracker.register(f"{total_processed}:{curr_time}")

                    effective_limit = stop_after_events if stop_after_events is not None else max_events
                    if effective_limit and total_processed >= effective_limit:
                        stop_early = True
                        break

            if not valid_rows:
                continue

            valid_chunk_df = pd.DataFrame(valid_rows)
            augmented_chunk = engine.context.process_batch(
                valid_chunk_df,
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
        if len(final_batch) > 0:
            for _, row in final_batch.iterrows():
                flow_dict = row.to_dict()
                if not _process_single_flow(flow_dict):
                    break

        # 7. Final telemetry flush to ensure complete and accurate final state on disk
        if write_feed:
            final_feed_state = _build_feed_state()
            telemetry_mgr.flush(final_feed_state)

        # 8. Final checkpoint save on graceful shutdown
        if checkpoint_mgr is not None and save_checkpoint_on_shutdown:
            checkpoint_mgr.save_checkpoint(
                context=engine.context,
                telemetry_mgr=telemetry_mgr,
                extra_state={
                    "total_processed": total_processed,
                    "total_anomalies": total_anomalies,
                    "recent_alerts": recent_alerts,
                    "v2_primary_anomalies": v2_primary_anomalies,
                    "v2_reference_anomalies": v2_reference_anomalies,
                    "v2_degraded_count": v2_degraded_count,
                    "v2_unavailable_count": v2_unavailable_count,
                    "v2_both_alert": v2_both_alert,
                    "v2_both_pass": v2_both_pass,
                },
            )

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
        "quarantine_summary": quarantine_manager.get_metrics(),
        "recent_alerts": recent_alerts,
        "telemetry_health": telemetry_mgr.get_health(),
        "recovery_result": recovery_result.to_dict() if recovery_result is not None else None,
        "checkpoint_saved": bool(checkpoint_mgr.total_checkpoints_saved > 0) if checkpoint_mgr is not None else False,
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
    parser.add_argument("--telemetry-throttle-sec", type=float, default=0.25, help="Telemetry disk write throttle interval in seconds (default: 0.25)")
    parser.add_argument("--telemetry-throttle-flows", type=int, default=20, help="Telemetry disk write flow batch size (default: 20)")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Path to checkpoint file for persistence and recovery")
    parser.add_argument("--no-auto-restore", action="store_false", dest="auto_restore", help="Disable automatic recovery from existing checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Save checkpoint every N flows")
    args = parser.parse_args()

    run_stream_simulation(
        speed_eps=args.eps,
        max_events=args.events,
        inject_attacks=args.inject_attacks,
        chunk_size=args.chunk_size,
        v1_only=args.v1_only,
        max_buffer_capacity=args.max_capacity,
        telemetry_throttle_sec=args.telemetry_throttle_sec,
        telemetry_throttle_flows=args.telemetry_throttle_flows,
        checkpoint_file=args.checkpoint_file,
        auto_restore=args.auto_restore,
        checkpoint_interval_events=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
