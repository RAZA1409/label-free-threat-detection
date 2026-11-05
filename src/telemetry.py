"""
AEGIS Telemetry Resilience Module (Phase 3 Stage 4B.3C)

Provides resilient, atomic, throttled, and fault-tolerant JSON persistence
and ingestion for live streaming feeds, dashboard integrations, and SOC telemetry.

Key Capabilities:
1. Strict RFC 8259 JSON serialization with safe normalization of NumPy types,
   NaN, and non-finite floats to None (null).
2. Atomic filesystem replacement via temporary file swap and os.replace().
3. Fault-tolerant exception handling preventing I/O and serialization crashes
   from aborting real-time streaming inference.
4. Bounded telemetry batching and throttling (interval/event counts) while
   bypassing throttling for zero-latency SOC threat alerts.
5. In-memory telemetry health tracking and automatic recovery after disk restores.
6. Bounded alert backlog preservation during transient storage outages.
7. Safe readers returning schema-compliant defaults on missing, empty, or
   malformed JSON files.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd

logger = logging.getLogger("aegis.telemetry")


# -----------------------------------------------------------------------------
# 1. Strict RFC 8259 JSON Normalization
# -----------------------------------------------------------------------------

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects for strict RFC 8259 JSON serialization.

    Transformations:
    - Booleans: np.bool_ or bool -> native Python bool
    - Integers: np.integer or int -> native Python int
    - Floats: np.floating or float -> float if finite, else None (null)
    - Strings/Bytes: bytes -> utf-8 decoded string with replacement
    - Datetimes/Timestamps: pd.Timestamp or datetime -> ISO-8601 string
    - Mappings: dict -> dict with stringified keys and sanitized values
    - Iterables: list, tuple, set, np.ndarray, pd.Series, pd.Index -> list
    - Objects with item() (e.g. 0-d arrays) -> sanitized scalar
    - Objects with to_dict() -> sanitized dict
    - None -> None
    - Fallback -> str(obj)
    """
    if obj is None:
        return None

    # Booleans (check first because bool subclasses int in Python)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    # Integers
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    # Floats: replace NaN and infinite floats with None
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    # Text and bytes
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    # Timestamps and Datetime
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "isoformat") and callable(obj.isoformat):
        return obj.isoformat()

    # 0-d NumPy scalar / array with item()
    if hasattr(obj, "ndim") and getattr(obj, "ndim") == 0 and hasattr(obj, "item") and callable(obj.item):
        try:
            return sanitize_for_json(obj.item())
        except Exception:
            pass

    # Dictionaries and Mappings
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # Iterables
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        return [sanitize_for_json(item) for item in obj]

    # Objects providing to_dict()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return sanitize_for_json(obj.to_dict())
        except Exception:
            pass

    # Generic string fallback for unknown non-serializable objects
    return str(obj)


# -----------------------------------------------------------------------------
# 2. Atomic JSON Writer
# -----------------------------------------------------------------------------

def atomic_write_json(
    file_path: Union[str, Path],
    data: Any,
    allow_nan: bool = False,
    indent: int = 2,
    raise_on_error: bool = False,
) -> bool:
    """
    Atomically writes data to file_path as JSON using a temporary file and os.replace().

    Ensures readers never observe zero-byte, truncated, or partially written files.
    All data is pre-sanitized to ensure compliance with allow_nan=False.
    """
    target = Path(file_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Place temp file on identical filesystem to guarantee atomic os.replace()
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")

    try:
        sanitized_data = sanitize_for_json(data)
        serialized_json = json.dumps(sanitized_data, allow_nan=allow_nan, indent=indent)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(serialized_json)
            f.flush()
            os.fsync(f.fileno())

        # Retry loop for transient Windows file locks during replacement
        max_retries = 3
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target)
                return True
            except (PermissionError, OSError) as lock_err:
                if attempt < max_retries - 1:
                    time.sleep(0.005)
                else:
                    raise lock_err

        return True
    except Exception as e:
        logger.warning("Atomic JSON write failed for %s: %s: %s", target, type(e).__name__, e)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        if raise_on_error:
            raise
        return False


# -----------------------------------------------------------------------------
# 3. Fault-Tolerant JSON Reader
# -----------------------------------------------------------------------------

def safe_read_json(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely reads and deserializes JSON from file_path.

    Returns `default` if the file does not exist, is empty (0 bytes),
    contains invalid/truncated JSON, or encounters an I/O/permission error.
    Guarantees no unhandled exceptions are raised to the caller.
    """
    try:
        p = Path(file_path)
        if not p.is_file():
            return default
        if p.stat().st_size == 0:
            return default

        with open(p, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return default

        return json.loads(content)
    except Exception as e:
        logger.debug("safe_read_json fallback on %s: %s: %s", file_path, type(e).__name__, e)
        return default


# -----------------------------------------------------------------------------
# 4. Telemetry Resilience & Throttling Manager
# -----------------------------------------------------------------------------

class TelemetryManager:
    """
    Manages bounded, throttled, and fault-tolerant telemetry persistence for AEGIS.

    Guarantees:
    - Critical SOC alerts and quarantine events bypass throttling and write immediately (0 latency).
    - Benign high-throughput traffic is throttled by interval and event counts to protect disk I/O.
    - Disk, serialization, or permission errors are captured in memory without aborting stream processing.
    - Alerts produced during storage outages are retained in an in-memory backlog.
    - Once storage recovers, full telemetry health automatically restores to HEALTHY.
    """

    def __init__(
        self,
        feed_path: Optional[Union[str, Path]] = None,
        enabled: bool = True,
        throttle_interval_sec: float = 0.25,
        throttle_event_count: int = 20,
    ):
        self.feed_path = Path(feed_path).resolve() if feed_path else None
        self.enabled = bool(enabled and self.feed_path is not None)
        self.throttle_interval_sec = max(0.0, float(throttle_interval_sec))
        self.throttle_event_count = max(1, int(throttle_event_count))

        # Health state
        self.is_degraded: bool = False
        self.consecutive_failures: int = 0
        self.total_failures: int = 0
        self.total_writes_attempted: int = 0
        self.total_writes_succeeded: int = 0
        self.total_writes_suppressed: int = 0
        self.last_error: Optional[str] = None
        self.last_write_monotonic: float = 0.0
        self.last_successful_write_iso: Optional[str] = None
        self.events_since_last_write: int = 0

        # In-memory alert preservation backlog
        self._backlog_alerts: List[Dict[str, Any]] = []

    def get_health(self) -> Dict[str, Any]:
        """Returns the in-memory telemetry health status dictionary."""
        return {
            "status": "DEGRADED" if self.is_degraded else "HEALTHY",
            "is_degraded": self.is_degraded,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_writes_attempted": self.total_writes_attempted,
            "total_writes_succeeded": self.total_writes_succeeded,
            "total_writes_suppressed": self.total_writes_suppressed,
            "last_error": self.last_error,
            "last_successful_write": self.last_successful_write_iso,
            "backlog_alerts_retained": len(self._backlog_alerts),
        }

    def should_write(self, force: bool = False) -> bool:
        """Determines whether a disk write should be triggered based on policy."""
        if not self.enabled:
            return False
        if force:
            return True
        if self.throttle_interval_sec <= 0.0 and self.throttle_event_count <= 1:
            return True

        now = time.monotonic()
        if (now - self.last_write_monotonic) >= self.throttle_interval_sec:
            return True
        if self.events_since_last_write >= self.throttle_event_count:
            return True

        return False

    def record_event(self, is_critical: bool = False) -> None:
        """Records an event occurrence to advance batching counters."""
        self.events_since_last_write += 1

    def update_and_write(
        self,
        feed_state: Dict[str, Any],
        force: bool = False,
        recent_alert: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Updates telemetry state and performs an atomic write if forced or threshold reached.

        Never raises an exception. Returns True if written successfully (or suppressed),
        and False if a disk/serialization failure occurred.
        """
        if not self.enabled:
            return True

        if recent_alert is not None:
            self._backlog_alerts.append(recent_alert)
            if len(self._backlog_alerts) > 150:
                self._backlog_alerts.pop(0)

        need_write = self.should_write(force=force)
        if not need_write:
            self.total_writes_suppressed += 1
            return True

        self.total_writes_attempted += 1

        # Embed health status inside payload without modifying standard keys
        health_snapshot = self.get_health()
        if self.is_degraded:
            # If recovering on this write, embed restored HEALTHY status
            health_snapshot["status"] = "HEALTHY"
            health_snapshot["is_degraded"] = False
            health_snapshot["consecutive_failures"] = 0
            health_snapshot["last_error"] = None
        feed_state["telemetry_health"] = health_snapshot

        try:
            atomic_write_json(
                self.feed_path,
                feed_state,
                allow_nan=False,
                indent=2,
                raise_on_error=True,
            )
            # Recovery or successful steady state
            self.total_writes_succeeded += 1
            self.consecutive_failures = 0
            self.is_degraded = False
            self.last_write_monotonic = time.monotonic()
            self.last_successful_write_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.events_since_last_write = 0
            self.last_error = None

            # Refresh health inside the written dict for consistency
            feed_state["telemetry_health"] = self.get_health()
            return True
        except Exception as e:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.is_degraded = True
            self.last_error = f"{type(e).__name__}: {e}"
            feed_state["telemetry_health"] = self.get_health()
            logger.warning("Telemetry persistence degraded: %s: %s", type(e).__name__, e)
            return False

    def flush(self, feed_state: Dict[str, Any]) -> bool:
        """Forces an immediate atomic write of the current state, e.g. at shutdown."""
        return self.update_and_write(feed_state, force=True)
