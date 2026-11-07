"""
AEGIS Checkpointing & Restart Recovery Module (Phase 3 Stage 4B.3D)

Provides versioned, atomic checkpoint persistence, schema validation,
durable critical-alert backlog preservation, replay idempotency, and
safe restart recovery for the AEGIS streaming threat detection pipeline.

Core Invariants:
1. Atomic Persistence: Checkpoint updates execute via temporary file swap
   and os.replace() to guarantee partial writes never overwrite valid checkpoints.
2. Safe Restoration: Strict schema, version, capacity, dimension, and timestamp
   validation gates prevent corrupted state from entering live inference.
3. Durable Alert Recovery: Critical alerts retained during outages survive
   process termination without duplication or chronological disorder.
4. Graceful Shutdown: Clean flush of pending groups, telemetry, and final checkpoint.
5. Invariant Preservation: v1 32-dim champion authority, v2 26-dim shadow,
   causal (t-10s, t] window, and 10,000-entry capacity ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
import pandas as pd

try:
    from src.telemetry import (
        atomic_write_json,
        safe_read_json,
        sanitize_for_json,
        TelemetryManager,
    )
except ImportError:
    from telemetry import (
        atomic_write_json,
        safe_read_json,
        sanitize_for_json,
        TelemetryManager,
    )

logger = logging.getLogger("aegis.checkpoint")

CURRENT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}
DEFAULT_MAX_BUFFER_CAPACITY = 10_000
DEFAULT_V1_FEATURE_DIM = 32
DEFAULT_V2_FEATURE_DIM = 26


# -----------------------------------------------------------------------------
# 1. Recovery Status & Result
# -----------------------------------------------------------------------------

class RecoveryStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    CORRUPTED_OR_MALFORMED = "CORRUPTED_OR_MALFORMED"
    INCOMPATIBLE_VERSION = "INCOMPATIBLE_VERSION"
    CAPACITY_VIOLATION = "CAPACITY_VIOLATION"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    INVALID_TIMESTAMPS = "INVALID_TIMESTAMPS"
    FAILED = "FAILED"


@dataclass
class RecoveryResult:
    """Represents the outcome of a checkpoint recovery operation."""
    status: RecoveryStatus
    checkpoint_path: Optional[Path] = None
    schema_version: Optional[str] = None
    restored_timestamp: Optional[pd.Timestamp] = None
    restored_buffer_entries: int = 0
    restored_pending_entries: int = 0
    restored_alerts_count: int = 0
    restored_flow_index: int = 0
    restored_anomalies: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    restored_v2_metrics: Dict[str, int] = field(default_factory=dict)
    restored_alerts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "schema_version": self.schema_version,
            "restored_timestamp": self.restored_timestamp.isoformat() if self.restored_timestamp else None,
            "restored_buffer_entries": self.restored_buffer_entries,
            "restored_pending_entries": self.restored_pending_entries,
            "restored_alerts_count": self.restored_alerts_count,
            "restored_flow_index": self.restored_flow_index,
            "restored_anomalies": self.restored_anomalies,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "restored_v2_metrics": self.restored_v2_metrics,
        }


# -----------------------------------------------------------------------------
# 2. Alert Idempotency Tracker
# -----------------------------------------------------------------------------

class AlertIdempotencyTracker:
    """
    Maintains a bounded set of committed alert identifiers to prevent
    duplicate alert emission or counting when resuming from checkpoints.
    """

    def __init__(self, max_capacity: int = 5_000):
        self.max_capacity = max(100, int(max_capacity))
        self._seen_ids: Set[str] = set()
        self._id_order: List[str] = []

    def register(self, alert_id: Union[str, int]) -> bool:
        """
        Registers an alert ID. Returns True if newly added, False if duplicate.
        """
        key = str(alert_id)
        if key in self._seen_ids:
            return False
        self._seen_ids.add(key)
        self._id_order.append(key)
        if len(self._id_order) > self.max_capacity:
            oldest = self._id_order.pop(0)
            self._seen_ids.discard(oldest)
        return True

    def is_seen(self, alert_id: Union[str, int]) -> bool:
        """Checks whether an alert ID has already been recorded."""
        return str(alert_id) in self._seen_ids

    def export_ids(self) -> List[str]:
        """Exports the active list of seen alert IDs in FIFO order."""
        return list(self._id_order)

    def load_ids(self, ids: List[Union[str, int]]) -> None:
        """Restores seen alert IDs from a checkpoint payload."""
        self._seen_ids.clear()
        self._id_order.clear()
        for item in ids[-self.max_capacity:]:
            key = str(item)
            self._seen_ids.add(key)
            self._id_order.append(key)


# -----------------------------------------------------------------------------
# 3. Checkpoint Manager
# -----------------------------------------------------------------------------

class CheckpointManager:
    """
    Coordinates atomic creation, validation, and safe restoration of AEGIS
    streaming checkpoints.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        schema_version: str = CURRENT_SCHEMA_VERSION,
        max_buffer_capacity: int = DEFAULT_MAX_BUFFER_CAPACITY,
        v1_feature_dim: int = DEFAULT_V1_FEATURE_DIM,
        v2_feature_dim: int = DEFAULT_V2_FEATURE_DIM,
        max_durable_alerts: int = 150,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.schema_version = str(schema_version)
        self.max_buffer_capacity = int(max_buffer_capacity)
        self.v1_feature_dim = int(v1_feature_dim)
        self.v2_feature_dim = int(v2_feature_dim)
        self.max_durable_alerts = int(max_durable_alerts)

        self.idempotency_tracker = AlertIdempotencyTracker(max_capacity=5_000)
        self.last_saved_time_iso: Optional[str] = None
        self.total_checkpoints_saved: int = 0
        self.total_checkpoints_failed: int = 0
        self.total_dropped_durable_alerts: int = 0

    def create_payload(
        self,
        context: Any,
        telemetry_mgr: Optional[TelemetryManager] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constructs a complete, versioned checkpoint payload from live components.
        """
        extra = extra_state or {}
        actual_ctx = getattr(context, "context", context)

        # 1. Causal buffer state
        raw_buffer = getattr(actual_ctx, "buffer", [])
        serialized_buffer = [
            [
                item[0].isoformat() if hasattr(item[0], "isoformat") else str(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
            ]
            for item in raw_buffer
        ]

        # 2. Pending simultaneous group state
        raw_pending = getattr(actual_ctx, "pending_group", [])
        serialized_pending = [
            [
                int(item[0]),
                item[1].isoformat() if hasattr(item[1], "isoformat") else str(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                dict(item[5]) if hasattr(item[5], "items") else {},
            ]
            for item in raw_pending
        ]

        latest_ts = getattr(actual_ctx, "latest_timestamp", None)
        latest_ts_iso = latest_ts.isoformat() if (latest_ts is not None and hasattr(latest_ts, "isoformat")) else None

        last_overflow = getattr(actual_ctx, "last_overflow_timestamp", None)
        last_overflow_iso = last_overflow.isoformat() if (last_overflow is not None and hasattr(last_overflow, "isoformat")) else None

        # 3. Durable alerts
        durable_alerts = []
        if telemetry_mgr is not None and hasattr(telemetry_mgr, "_backlog_alerts"):
            durable_alerts = list(telemetry_mgr._backlog_alerts)
        elif "recent_alerts" in extra:
            durable_alerts = list(extra["recent_alerts"])

        # Enforce bounded capacity on durable backlog
        if len(durable_alerts) > self.max_durable_alerts:
            excess = len(durable_alerts) - self.max_durable_alerts
            self.total_dropped_durable_alerts += excess
            durable_alerts = durable_alerts[-self.max_durable_alerts:]

        # 4. Assemble payload
        v2_shadow_metrics = {
            "primary_anomalies": int(extra.get("v2_primary_anomalies", 0)),
            "reference_anomalies": int(extra.get("v2_reference_anomalies", 0)),
            "degraded_count": int(extra.get("v2_degraded_count", 0)),
            "unavailable_count": int(extra.get("v2_unavailable_count", 0)),
            "both_alert": int(extra.get("v2_both_alert", 0)),
            "both_pass": int(extra.get("v2_both_pass", 0)),
        }

        payload = {
            "schema_version": self.schema_version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine_metadata": {
                "v1_feature_dim": self.v1_feature_dim,
                "v2_feature_dim": self.v2_feature_dim,
                "window_seconds": getattr(context, "window_seconds", 10),
                "max_buffer_capacity": self.max_buffer_capacity,
                "causal_mode": "simultaneous",
            },
            "causal_context": {
                "latest_timestamp": latest_ts_iso,
                "total_processed": getattr(context, "total_processed", 0),
                "total_late_events": getattr(context, "total_late_events", 0),
                "is_degraded": bool(getattr(context, "is_degraded", False)),
                "last_overflow_timestamp": last_overflow_iso,
                "total_dropped_overflow_events": getattr(context, "total_dropped_overflow_events", 0),
                "rolling_buffer": serialized_buffer,
                "pending_group": serialized_pending,
            },
            "alert_state": {
                "total_anomalies": extra.get("total_anomalies", 0),
                "durable_alert_backlog": durable_alerts,
                "committed_alert_ids": self.idempotency_tracker.export_ids(),
                "total_dropped_durable_alerts": self.total_dropped_durable_alerts,
                "v2_shadow_state": v2_shadow_metrics,
            },
            "stream_progress": {
                "flow_index": extra.get("total_processed", getattr(context, "total_processed", 0)),
                "last_processed_timestamp": latest_ts_iso,
            },
        }
        return payload

    def save_checkpoint(
        self,
        context: Any,
        telemetry_mgr: Optional[TelemetryManager] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Atomically saves live state to checkpoint_path.
        Guarantees that a failed or interrupted save never damages a valid checkpoint.
        """
        if not self.checkpoint_path:
            return False

        try:
            payload = self.create_payload(context, telemetry_mgr=telemetry_mgr, extra_state=extra_state)
            success = atomic_write_json(
                self.checkpoint_path,
                payload,
                allow_nan=False,
                indent=2,
                raise_on_error=True,
            )
            if success:
                self.last_saved_time_iso = payload["created_at"]
                self.total_checkpoints_saved += 1
                return True
            else:
                self.total_checkpoints_failed += 1
                return False
        except Exception as e:
            self.total_checkpoints_failed += 1
            logger.warning("Failed to save checkpoint to %s: %s: %s", self.checkpoint_path, type(e).__name__, e)
            return False

    def validate_payload(self, payload: Any) -> Tuple[bool, RecoveryStatus, Optional[str]]:
        """
        Validates checkpoint schema, version, dimensions, buffer capacity, and timestamps.
        Returns (is_valid, status, error_message).
        """
        if not isinstance(payload, dict):
            return False, RecoveryStatus.CORRUPTED_OR_MALFORMED, "Checkpoint payload is not a valid dictionary"

        # 1. Version validation
        version = payload.get("schema_version")
        if not version or version not in SUPPORTED_SCHEMA_VERSIONS:
            return (
                False,
                RecoveryStatus.INCOMPATIBLE_VERSION,
                f"Unsupported checkpoint schema version '{version}'. Supported: {SUPPORTED_SCHEMA_VERSIONS}",
            )

        # 2. Structural fields validation
        for section in ["engine_metadata", "causal_context", "alert_state"]:
            if section not in payload or not isinstance(payload[section], dict):
                return False, RecoveryStatus.CORRUPTED_OR_MALFORMED, f"Missing required section '{section}'"

        meta = payload["engine_metadata"]
        ctx = payload["causal_context"]

        # 3. Dimension validation
        v1_dim = meta.get("v1_feature_dim")
        v2_dim = meta.get("v2_feature_dim")
        if v1_dim != self.v1_feature_dim or v2_dim != self.v2_feature_dim:
            return (
                False,
                RecoveryStatus.DIMENSION_MISMATCH,
                f"Feature dimension mismatch: expected v1={self.v1_feature_dim}/v2={self.v2_feature_dim}, got v1={v1_dim}/v2={v2_dim}",
            )

        # 4. Capacity validation
        buf = ctx.get("rolling_buffer", [])
        pending = ctx.get("pending_group", [])
        if not isinstance(buf, list) or not isinstance(pending, list):
            return False, RecoveryStatus.CORRUPTED_OR_MALFORMED, "Buffer or pending group is not a list"

        total_entries = len(buf) + len(pending)
        if total_entries > self.max_buffer_capacity:
            return (
                False,
                RecoveryStatus.CAPACITY_VIOLATION,
                f"Checkpoint total entries ({total_entries}) exceeds max_buffer_capacity ({self.max_buffer_capacity})",
            )

        # 5. Timestamp and chronological monotonicity validation
        parsed_timestamps: List[pd.Timestamp] = []
        tz_set: Set[Optional[str]] = set()

        for idx, entry in enumerate(buf):
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                return False, RecoveryStatus.CORRUPTED_OR_MALFORMED, f"Malformed buffer entry at index {idx}"
            raw_ts = entry[0]
            try:
                ts_pd = pd.to_datetime(raw_ts, errors="raise")
                if pd.isna(ts_pd):
                    return False, RecoveryStatus.INVALID_TIMESTAMPS, f"Null or NaT timestamp at buffer index {idx}"
                parsed_timestamps.append(ts_pd)
                tz_set.add(str(ts_pd.tz) if ts_pd.tz is not None else None)
            except Exception as e:
                return False, RecoveryStatus.INVALID_TIMESTAMPS, f"Unparseable timestamp '{raw_ts}' at index {idx}: {e}"

        # Enforce timezone consistency across all buffer timestamps
        if len(tz_set) > 1:
            return False, RecoveryStatus.INVALID_TIMESTAMPS, f"Inconsistent timezones detected in buffer timestamps: {tz_set}"

        # Enforce non-decreasing chronological ordering within rolling buffer
        for i in range(len(parsed_timestamps) - 1):
            if parsed_timestamps[i] > parsed_timestamps[i + 1]:
                return (
                    False,
                    RecoveryStatus.INVALID_TIMESTAMPS,
                    f"Non-chronological timestamp order in buffer: {parsed_timestamps[i]} > {parsed_timestamps[i+1]}",
                )

        # Validate pending group timestamps
        pending_tz_set: Set[Optional[str]] = set()
        for idx, entry in enumerate(pending):
            if not isinstance(entry, (list, tuple)) or len(entry) < 6:
                return False, RecoveryStatus.CORRUPTED_OR_MALFORMED, f"Malformed pending group entry at index {idx}"
            raw_ts = entry[1]
            try:
                ts_pd = pd.to_datetime(raw_ts, errors="raise")
                if pd.isna(ts_pd):
                    return False, RecoveryStatus.INVALID_TIMESTAMPS, f"Null/NaT timestamp in pending group at index {idx}"
                pending_tz_set.add(str(ts_pd.tz) if ts_pd.tz is not None else None)
            except Exception as e:
                return False, RecoveryStatus.INVALID_TIMESTAMPS, f"Unparseable pending timestamp '{raw_ts}' at index {idx}: {e}"

        if len(pending_tz_set) > 1:
            return False, RecoveryStatus.INVALID_TIMESTAMPS, f"Inconsistent timezones in pending group: {pending_tz_set}"

        return True, RecoveryStatus.SUCCESS, None

    def restore_checkpoint(
        self,
        context: Any,
        telemetry_mgr: Optional[TelemetryManager] = None,
    ) -> RecoveryResult:
        """
        Reads, validates, and safely restores state from checkpoint_path.

        If the file does not exist, returns RecoveryStatus.SKIPPED.
        If corrupted or invalid, rejects safely and leaves context in clean state.
        If valid, restores buffer, watermarks, alerts, and idempotency tracking.
        """
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return RecoveryResult(
                status=RecoveryStatus.SKIPPED,
                checkpoint_path=self.checkpoint_path,
                error_message="Checkpoint file does not exist",
            )

        payload = safe_read_json(self.checkpoint_path, default=None)
        if payload is None:
            return RecoveryResult(
                status=RecoveryStatus.CORRUPTED_OR_MALFORMED,
                checkpoint_path=self.checkpoint_path,
                error_message="Checkpoint file is empty, locked, or contains malformed JSON",
            )

        is_valid, status, err_msg = self.validate_payload(payload)
        if not is_valid:
            logger.warning("Checkpoint validation failed (%s): %s", status.value, err_msg)
            return RecoveryResult(
                status=status,
                checkpoint_path=self.checkpoint_path,
                schema_version=payload.get("schema_version") if isinstance(payload, dict) else None,
                error_message=err_msg,
            )

        try:
            # 1. Restore Causal Context
            ctx_data = payload["causal_context"]
            rolling_buffer_raw = ctx_data["rolling_buffer"]
            pending_raw = ctx_data["pending_group"]

            from collections import deque
            restored_buffer = deque()
            for entry in rolling_buffer_raw:
                ts_pd = pd.Timestamp(entry[0])
                restored_buffer.append((ts_pd, float(entry[1]), float(entry[2]), float(entry[3])))

            restored_pending = []
            for entry in pending_raw:
                row_idx = int(entry[0])
                ts_pd = pd.Timestamp(entry[1])
                p = float(entry[2])
                b = float(entry[3])
                s = float(entry[4])
                orig_dict = dict(entry[5])
                restored_pending.append((row_idx, ts_pd, p, b, s, orig_dict))

            # Apply state to context
            actual_ctx = getattr(context, "context", context)
            actual_ctx.buffer = restored_buffer
            actual_ctx.pending_group = restored_pending

            latest_ts_str = ctx_data.get("latest_timestamp")
            actual_ctx.latest_timestamp = pd.Timestamp(latest_ts_str) if latest_ts_str else None

            actual_ctx.total_processed = int(ctx_data.get("total_processed", 0))
            actual_ctx.total_late_events = int(ctx_data.get("total_late_events", 0))
            actual_ctx.is_degraded = bool(ctx_data.get("is_degraded", False))
            actual_ctx.total_dropped_overflow_events = int(ctx_data.get("total_dropped_overflow_events", 0))

            overflow_ts_str = ctx_data.get("last_overflow_timestamp")
            actual_ctx.last_overflow_timestamp = pd.Timestamp(overflow_ts_str) if overflow_ts_str else None

            # 2. Restore Alert & Idempotency State
            alert_data = payload["alert_state"]
            durable_alerts = alert_data.get("durable_alert_backlog", [])
            committed_ids = alert_data.get("committed_alert_ids", [])
            self.idempotency_tracker.load_ids(committed_ids)

            if telemetry_mgr is not None:
                telemetry_mgr._backlog_alerts = list(durable_alerts)
                # Register all restored alerts in idempotency tracker
                for al in durable_alerts:
                    al_id = al.get("flow_index", al.get("timestamp"))
                    if al_id is not None:
                        self.idempotency_tracker.register(al_id)

            v2_state = alert_data.get("v2_shadow_state", {})
            restored_v2_metrics = {
                "primary_anomalies": int(v2_state.get("primary_anomalies", 0)),
                "reference_anomalies": int(v2_state.get("reference_anomalies", 0)),
                "degraded_count": int(v2_state.get("degraded_count", 0)),
                "unavailable_count": int(v2_state.get("unavailable_count", 0)),
                "both_alert": int(v2_state.get("both_alert", 0)),
                "both_pass": int(v2_state.get("both_pass", 0)),
            }

            stream_prog = payload.get("stream_progress", {})
            restored_flow_index = int(stream_prog.get("flow_index", getattr(context, "total_processed", 0)))
            restored_anomalies = int(alert_data.get("total_anomalies", 0))

            return RecoveryResult(
                status=RecoveryStatus.SUCCESS,
                checkpoint_path=self.checkpoint_path,
                schema_version=payload["schema_version"],
                restored_timestamp=getattr(actual_ctx, "latest_timestamp", None),
                restored_buffer_entries=len(getattr(actual_ctx, "buffer", [])),
                restored_pending_entries=len(getattr(actual_ctx, "pending_group", [])),
                restored_alerts_count=len(durable_alerts),
                restored_flow_index=restored_flow_index,
                restored_anomalies=restored_anomalies,
                restored_v2_metrics=restored_v2_metrics,
                restored_alerts=list(durable_alerts),
                metrics={
                    "total_processed": getattr(actual_ctx, "total_processed", 0),
                    "is_degraded": getattr(actual_ctx, "is_degraded", False),
                    "total_dropped_overflow_events": getattr(actual_ctx, "total_dropped_overflow_events", 0),
                    "committed_alert_ids_count": len(self.idempotency_tracker.export_ids()),
                },
            )
        except Exception as e:
            logger.error("Unexpected exception during state restoration: %s", e, exc_info=True)
            # Guarantee: reset context to clean default on failed restoration!
            if hasattr(context, "reset"):
                context.reset()
            return RecoveryResult(
                status=RecoveryStatus.FAILED,
                checkpoint_path=self.checkpoint_path,
                schema_version=payload.get("schema_version") if isinstance(payload, dict) else None,
                error_message=f"Failed to apply validated state: {type(e).__name__}: {e}",
            )
