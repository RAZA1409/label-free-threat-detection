# AEGIS Phase 3 Stage 4B — Live Streaming Integration Architecture Audit

**Project:** AEGIS – Label-Free Threat Detection in One-Way Network Traffic
**SIH Problem ID:** SIH26145
**Current Branch:** `feat/phase-2-pure-one-way-model`
**Current HEAD:** `707417b`
**Compliance Phrasing:** 26-dimensional forward-feature-only design; reverse-direction feature information excluded.
**Auditor:** Principal ML Systems Architect, Streaming ML Engineer & Independent QA Auditor
**Audit Date:** 2026-11-04
**Stage 4B Audit Recommendation:** **CONDITIONAL GO — implementation authorized only with the above safeguards.**

---

## 1. Executive Summary & Audit Mandate

Following the successful implementation and independent verification of the **Causal Streaming Feature Engine** in Stage 4A (58/58 tests passed, $\Delta = 0.0$ bit-exact parity across all 26 dimensions), this Stage 4B integration audit evaluates the technical architecture, safety boundaries, data flow, and performance implications of integrating the Pure One-Way v2 model into the live streaming simulator (`src/simulate_stream.py`).

### Mandatory Architectural Boundaries
1. **Legacy Champion (v1) Invariance:** v1 remains the active, default production champion.
2. **Shadow-Mode Isolation:** v2 executes strictly as an experimental shadow observer. No v2 score, prediction, or failure may alter v1 alerting, decision thresholds, or pipeline execution.
3. **Strict Directional Hygiene:** v2 consumes strictly 26 forward-only features (`PURE_ONE_WAY_FEATURES`). Zero reverse-direction or bidirectional features may enter the v2 pipeline.
4. **Causal & Temporal Integrity:** Monotonic chronological flow replay must replace random sampling, enforcing the backward-looking $(t - 10\text{s}, t]$ window.
5. **Dashboard & API Backward Compatibility:** The schema of `outputs/evaluation/live_stream_feed.json` must preserve 100% of existing top-level keys consumed by `dashboard/server.py` and `dashboard/index.html`.

---

## 2. Existing Live-Stream Architecture Audit

### 2.1 Current End-to-End Ingestion Flow

```
                                CURRENT LIVE STREAMING FLOW

   Disk Dataset: outputs/experiment/test_tue_fri.parquet
           |
           v
   src/simulate_stream.py: run_stream_simulation()
     • Reads static parquet capture
     • Cleans labels via features.py:sanitize_labels()
     • Computes temporal velocities statically: features.py:add_temporal_velocity_features()
     • Shuffles sampled pool: pd.concat([benign, attack]).sample(frac=1.0)  <-- [HAZARD: Breaks Time]
           |
           | Iterates row-by-row (throttled by sleep(1/eps))
           v
   src/explainability.py: ThreatExplainer.explain_flow()
     • Converts row dict to 1-row DataFrame
     • Transforms via LogPreprocessingPipeline (32 features)
     • Computes weighted MSE reconstruction error via MLPRegressor
     • Decision: score >= 0.190885 -> is_anomaly
     • Generates top-5 residual attribution and analyst narrative
           |
           v
   outputs/evaluation/live_stream_feed.json
     • Writes JSON state (total_processed, total_anomalies, recent_alerts [150 FIFO])
           |
           +--------------------------------+
           |                                |
           v                                v
   dashboard/server.py              dashboard/index.html
     • GET /api/live_feed             • pollLiveFeed() (1.5s interval)
     • Streams JSON file to client    • Renders live radar table
                                      • Populates forensic drawer
```

### 2.2 Critical Vulnerabilities in Current Streaming Simulator
1. **Temporal Disruption:** `sample(frac=1.0)` randomly shuffles events across capture days, corrupting dynamic causal windowing.
2. **Missing v2 Features:** Streaming ingestion does not compute forward transmission rates (`Fwd_Bytes_Per_Sec`, `Fwd_Packets_Per_Sec`).
3. **Single-Row DataFrame Overhead:** `ThreatExplainer.explain_flow()` incurs a 24.5 ms latency overhead per flow constructing Pandas DataFrames, capping unthrottled single-thread throughput at ~40 flows/second.
4. **Zero Shadow Hooks:** The simulator has no interface for dual-model evaluation.

---

## 3. Dual-Model Shadow Integration Design

The proposed Stage 4B architecture integrates `StreamingFeatureEngine` alongside `ThreatExplainer`, isolating v2 in an independent shadow execution context:

```
                            PROPOSED STAGE 4B SHADOW ARCHITECTURE

                  Chronological Replay Buffer (test_tue_fri.parquet)
                                           |
                                           v
                       StreamingFeatureEngine / StreamingCausalContext
                         • Causal rolling window: (t - 10s, t]
                         • Max buffer capacity ceiling: max_buffer_capacity
                         • Computes: Fwd_Bytes_Per_Sec, Fwd_Packets_Per_Sec
                         • Computes: Rolling velocities (10s)
                                           |
                   +-----------------------+-----------------------+
                   |                                               |
                   v                                               v
        v1 Production Champion                           v2 Experimental Shadow
     (src/explainability.py)                         (src/model_adapters.py)
   -----------------------------                   -----------------------------
   • Input: 32 dimensions                          • Input: 26 dimensions
   • Preprocessing: LogPreprocessingPipeline       • Preprocessing: Fast NumPy (103 us)
   • Scoring: Weighted MSE                         • Scoring: Unweighted MSE
   • Threshold: 0.190885                           • Thresholds: 0.314036 / 0.099018
   • Decision: score >= threshold                  • Decision: score > threshold
   • Role: ACTIVE SOC ALERTS                       • Role: ISOLATED SHADOW EVALUATION
                   |                                               |
                   |                                 [try / except Fault Boundary]
                   v                                               v
        v1 Primary SOC Alert Payload                     v2 Shadow Verdict Object
                   \                                               /
                    \                                             /
                     +----------------------+--------------------+
                                            |
                                            v
                               Alert Payload Consolidator
                                            |
                                            v
                      outputs/evaluation/live_stream_feed.json
                               (Backward-Compatible)
```

### 3.1 Fault-Isolated Shadow Execution
All v2 operations are wrapped in an isolated exception block:
```python
v2_shadow = None
if v2_engine is not None and v2_adapter is not None and v2_adapter.is_available:
    try:
        t0_v2 = time.perf_counter()
        # Fast NumPy transform (<110 us)
        X_v2_scaled = v2_engine.transform_flow_fast(raw_flow_dict)
        # Unweighted MSE score
        X_v2_rec = v2_adapter.model.predict(X_v2_scaled)
        v2_diff_sq = np.square(X_v2_scaled[0] - X_v2_rec[0])
        v2_score = float(np.mean(v2_diff_sq))
        v2_latency_ms = (time.perf_counter() - t0_v2) * 1000.0

        v2_shadow = {
            "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
            "score": round(v2_score, 4),
            "primary": {
                "threshold": v2_adapter.primary_threshold,
                "is_anomaly": bool(v2_score > v2_adapter.primary_threshold),
                "verdict": "ALERT" if v2_score > v2_adapter.primary_threshold else "PASS"
            },
            "reference": {
                "threshold": v2_adapter.reference_threshold,
                "is_anomaly": bool(v2_score > v2_adapter.reference_threshold),
                "verdict": "ALERT" if v2_score > v2_adapter.reference_threshold else "PASS"
            },
            "latency_ms": round(v2_latency_ms, 3),
            "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
            "top_drivers": _extract_v2_top_drivers(v2_diff_sq, v2_engine.feature_names, top_n=3)
        }
    except Exception as e:
        v2_shadow = {
            "status": "UNAVAILABLE",
            "error": str(e)
        }
```

### 3.2 Backward-Compatible JSON Schema
`outputs/evaluation/live_stream_feed.json` retains every existing top-level key:
```json
{
  "last_updated": "14:22:05",
  "total_processed": 500,
  "total_anomalies": 62,
  "anomaly_rate_pct": 12.4,
  "v2_shadow_summary": {
    "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
    "primary_anomalies": 48,
    "primary_anomaly_rate_pct": 9.6,
    "reference_anomalies": 84,
    "reference_anomaly_rate_pct": 16.8,
    "concordance_rate_pct": 92.4,
    "mean_latency_ms": 0.125
  },
  "recent_alerts": [
    {
      "timestamp": "14:22:05",
      "flow_index": 500,
      "ground_truth": "DoS Hulk",
      "score": 0.8412,
      "threshold": 0.190885,
      "verdict": "ANOMALOUS (THREAT DETECTED)",
      "severity": "HIGH",
      "latency_ms": 24.5,
      "top_drivers": [
        {
          "feature": "Rolling_Flow_Velocity_10s",
          "residual_contribution_pct": 42.5
        }
      ],
      "narrative": "ALERT: Flow flagged as anomalous...",
      "v2_shadow": {
        "status": "EXPERIMENTAL / RESEARCH CANDIDATE",
        "score": 0.7915,
        "primary": { "threshold": 0.314036, "is_anomaly": true, "verdict": "ALERT" },
        "reference": { "threshold": 0.099018, "is_anomaly": true, "verdict": "ALERT" },
        "latency_ms": 0.112,
        "compliance_scope": "26-dimensional forward-feature-only design; reverse-direction feature information excluded.",
        "top_drivers": [
          { "feature": "Rolling_Flow_Velocity_10s", "residual_contribution_pct": 46.1 }
        ]
      }
    }
  ]
}
```

---

## 4. Proposed Files to Modify and Create

### 4.1 Files to Modify
1. `src/streaming_features.py`:
   - Add `max_buffer_capacity: int = 10_000` to `StreamingCausalContext.__init__()`.
   - Implement safe overflow eviction policy with telemetry tracking (`total_dropped_overflow_events`).
2. `src/simulate_stream.py`:
   - Replace random shuffling with chronological replay via `prepare_chronological_stream()`.
   - Instantiate `StreamingFeatureEngine` and `PureOneWayV2Adapter`.
   - Incorporate the fault-isolated v2 shadow inference block into the flow processing loop.
   - Output the dual-model consolidated telemetry payload to `live_stream_feed.json`.
3. `dashboard/index.html`: (Optional UI enhancement)
   - Add a subtle shadow v2 indicator badge in the Live Ingestion Radar table and slide-over drawer without changing existing v1 elements.

### 4.2 Files to Create
1. `tests/test_live_streaming_integration.py`:
   - End-to-end integration test suite verifying v1 backward compatibility, v2 shadow isolation, buffer capacity capping, and JSON schema compliance.
2. `reports/PHASE_3_STAGE_4B_LIVE_STREAMING_IMPLEMENTATION_REPORT.md`:
   - Implementation deliverable documenting empirical performance, concordance metrics, and operational stability.

---

## 5. Buffer Capacity & Overflow Policy Design

### 5.1 The Volumetric DDoS Problem
Stage 4A verification confirmed that `StreamingCausalContext` is time-bounded $(t - 10\text{s}, t]$, not count-bounded. Under a 5,000 EPS flood, the buffer naturally expands to >10,000 items. To prevent unbounded memory expansion during sustained multi-gigabit denial-of-service traffic, a deterministic capacity ceiling must be enforced.

### 5.2 Configurable Capacity & Safe Eviction Policy
```python
class StreamingCausalContext:
    def __init__(
        self,
        window_seconds: int = 10,
        max_buffer_capacity: int = 10_000,
        max_late_seconds: float = 10.0,
    ):
        self.window_seconds = window_seconds
        self.max_buffer_capacity = max_buffer_capacity
        self.total_dropped_overflow_events = 0
        ...

    def _enforce_capacity(self) -> None:
        """Enforces max_buffer_capacity ceiling by evicting the oldest in-window entry."""
        while len(self.buffer) > self.max_buffer_capacity:
            self.buffer.popleft()
            self.total_dropped_overflow_events += 1
```

### Guarantees of the Overflow Policy:
1. **Memory Bound:** Context buffer memory is strictly capped at $10,000 \times 64\text{ bytes} \approx 640\text{ KB}$.
2. **Causal Invariance:** Eviction removes the oldest entry from the left (tail), preserving the causal property that no future event ($t' > t$) is ever admitted.
3. **Audit Telemetry:** `total_dropped_overflow_events` is tracked and reported in `live_stream_feed.json`. The system never silently drops events.

---

## 6. Timestamp and Late-Arrival Safety

| Operational Scenario | Ingestion Mechanism | Safety & Causality Guarantee |
|---|---|---|
| **Simultaneous Events (Same $t$)** | Grouped in `pending_group` | Finalized together when timestamp advances; reproduces offline `transform("last")` bit-for-bit. |
| **Intermediate Stream Batches** | Processed with `flush=False` | Open timestamp group bridges chunk boundaries without premature emission. |
| **Stream Termination** | Calls `context.flush()` at EOF | Forces finalization of the trailing timestamp group before process exit. |
| **In-Window Late Arrival** | Inserted chronologically into deque | Evaluated causally; **never retroactively mutates** previously emitted alerts. |
| **Expired Late Arrival ($>10\text{s}$)** | Evaluated in single-event isolation | Does not append to buffer; strictly zero contamination of future streaming context. |
| **Event-Time vs Wall-Clock** | Sliding window tracks `Timestamp` | Simulation playback speed (`--eps`) throttles loop via `sleep(1/eps)` without affecting window math. |

---

## 7. Performance & Latency Benchmark Plan

Benchmarks must separately profile algorithmic latency vs. throttled simulation throughput:

```
+-----------------------------------------------------------------------------------------------+
| BENCHMARK MEASUREMENT METHODOLOGY (Stage 4B Plan)                                             |
+-----------------------------------+-----------------------------------+-----------------------+
| Metric                            | Target Measurement                | Repetitions / Method  |
+-----------------------------------+-----------------------------------+-----------------------+
| 1. Feature Generation (t_feat)    | StreamingFeatureEngine transform  | 10,000 reps, timer    |
| 2. v1 Inference (t_v1)            | ThreatExplainer.explain_flow()    | 500 reps, timer       |
| 3. v2 Shadow Inference (t_v2)     | MLPRegressor.predict() (26 dims)  | 10,000 reps, timer    |
| 4. Alert Payload Consolidation    | JSON dictionary assembly          | 5,000 reps, timer     |
| 5. Disk Write (Atomic I/O)        | File write to live_feed.json      | 500 reps, timer       |
| 6. Total Per-Flow Processing Time | t_feat + t_v1 + t_v2 + t_io       | End-to-end stopwatch  |
| 7. Burst Throughput Headroom      | Unthrottled streaming playback    | 5,000 flows, no sleep |
+-----------------------------------+-----------------------------------+-----------------------+
```

### Projected Latency Profile:
- `t_feat` (v2 fast features): ~0.048 ms
- `t_v2` (v2 inference + attribution): ~0.085 ms
- `t_v1` (v1 Pandas + inference + attribution): ~24.5 ms (existing bottleneck)
- `t_io` (JSON serialization): ~0.35 ms
- **Total per-flow processing:** ~25.0 ms [ESTIMATE] (supports up to 40 EPS unthrottled single-threaded).
- Note: This 25.0 ms/flow latency is strictly an engineering projection governed by v1's single-row Pandas DataFrame construction in `ThreatExplainer`. Actual end-to-end latency will be empirically measured post-integration, reporting p50, p95, and p99 distributions alongside unthrottled throughput. Playback rate should remain configured at $\le 20$ EPS.

---

## 8. Automated Test Specifications for Stage 4B

The following tests will be implemented in `tests/test_live_streaming_integration.py`:

1. **`test_live_01_v1_payload_backward_compatibility`:**
   Verify that all existing v1 keys (`score`, `threshold`, `verdict`, `severity`, `top_drivers`, `narrative`) are present, correctly typed, and numerically unchanged when v2 shadow mode is active.
2. **`test_live_02_v2_shadow_isolation_on_v2_failure`:**
   Inject artificial exceptions into v2 (e.g., corrupt weights, missing files); verify that v1 continues emitting valid alerts without interruption and sets `v2_shadow.status = "UNAVAILABLE"`.
3. **`test_live_03_strict_26_dimension_v2_input_guard`:**
   Verify that v2 receives strictly 26 features and rejects any 32-dimensional legacy DataFrame with `ValueError`.
4. **`test_live_04_buffer_capacity_ceiling_enforcement`:**
   Stream 15,000 flows in a tight simulated burst; verify that `buffer_length` never exceeds `max_buffer_capacity` (10,000), `total_dropped_overflow_events` equals 5,000, and no NaN/Inf vectors are emitted.
5. **`test_live_05_chronological_ordering_integrity`:**
   Verify that the stream simulator plays back events in strictly nondecreasing timestamp order ($t_{i+1} \ge t_i$) with stable tie-breaking for duplicate timestamps, preserving simultaneous timestamp groups across batch boundaries.
6. **`test_live_06_live_feed_json_schema_validation`:**
   Verify that `outputs/evaluation/live_stream_feed.json` strictly adheres to the schema expected by `dashboard/server.py` and `dashboard/index.html`.
7. **`test_live_07_dual_model_concordance_metrics`:**
   Verify that agreement/divergence counts (`both_alert`, `v1_only`, `v2_only`, `both_pass`) are accurately calculated and reported in `v2_shadow_summary`.

---

## 9. Risk & Mitigation Matrix

| Operational Risk | Likelihood | Impact | Architectural Mitigation |
|---|---|---|---|
| **v2 Exception Crashes Stream** | Low | Critical | Isolated `try...except` boundary around all v2 feature extraction and scoring. |
| **Dashboard JSON Parsing Error** | Very Low | High | Strict schema compliance; all legacy keys preserved at top-level; v2 isolated under nested `v2_shadow`. |
| **High Memory on Floods** | Medium | Medium | Configurable `max_buffer_capacity` (10,000 items) with oldest-entry eviction and telemetry counter. |
| **Premature Timestamp Closure** | Low | Low | Enforce `flush=False` on intermediate stream chunks; call `flush()` only upon stream termination. |
| **v1 Timing Degradation** | Low | Low | v2 fast NumPy transform takes only ~0.104 ms, adding negligible (<0.5%) overhead to the 25 ms v1 cycle. |

---

## 10. Independent Engineering Conditions Before Implementation

To ensure safe, robust, and mathematically sound integration of Pure One-Way v2 shadow inference into the live stream simulator, the following five engineering conditions must be satisfied prior to and during Stage 4B implementation:

### 10.1 Buffer Overflow Safety & Degraded Context Policy
- **Hazard Analysis:** When `max_buffer_capacity` evicts events that are still within the active $(t - 10\text{s}, t]$ causal window, the calculated rolling velocity metrics (`Rolling_Flow_Velocity_10s`, `Rolling_Bytes_Velocity_10s`, `Rolling_Single_Pkt_Ratio_10s`) will be derived from a truncated, incomplete window. Generating standard v2 anomaly scores on artificially suppressed velocity features would produce unreliable verdicts.
- **Mandatory Safeguards:**
  1. *No Silent Scoring:* The engine must **never silently score v2** using incomplete context.
  2. *Explicit Degraded State:* When an in-window entry is dropped due to capacity overflow, the context transitions to `is_degraded = True`. During this state:
     - `v2_shadow["status"]` is set to `"DEGRADED_CONTEXT_OVERFLOW"`.
     - `v2_shadow["primary"]["is_anomaly"]` and `v2_shadow["reference"]["is_anomaly"]` are set to `null` (or marked `DEGRADED (OVERFLOW)`).
     - Telemetry counter `total_dropped_overflow_events` is incremented and exposed in `v2_shadow_summary`.
  3. *Deterministic Recovery Policy:* Normal v2 scoring resumes **only after a full 10-second window has elapsed with zero dropped events** ($t_{\text{current}} > t_{\text{last\_overflow}} + 10\text{s}$), at which point `is_degraded` transitions back to `False`.

### 10.2 Nondecreasing Timestamp Ordering & Batch Group Integrity
- **Ordering Invariant:** Flow replay must enforce **weak monotonicity (nondecreasing timestamps)**:
  $$t_{i+1} \ge t_i$$
  Strictly increasing timestamps ($t_{i+1} > t_i$) must **not** be required because capture files naturally contain simultaneous events sharing identical timestamps (e.g. 1-minute truncation in CIC-IDS2017).
- **Secondary Sort Key:** Duplicate timestamps must be ordered deterministically using a stable tie-breaker: `sort_values(by=["Timestamp", "_orig_pos"], kind="stable")`.
- **Multi-Batch Boundary Preservation:** In multi-batch streaming, timestamp groups that span batch boundaries must be preserved using `flush=False` on intermediate chunks, finalising groups only when the timestamp strictly advances or when stream termination invokes `context.flush()`.

### 10.3 v1 Regression Protection & Alert Ordering Separation
- **Separation of Concerns:**
  1. *Alert Sequence Reordering:* In the legacy simulator, `stream_pool` was shuffled via `.sample(frac=1.0, random_state=42)`. In chronological mode, events arrive sorted by timestamp. The *sequence of emitted alerts* will naturally change (e.g., Friday flows arrive after Tuesday flows). This sequence shift is expected and does not constitute a model regression.
  2. *Deterministic Model Invariance:* For any given flow $i$ (indexed by original dataset position), its 32-dimensional preprocessed array, weighted anomaly score, calibrated threshold (0.190885), verdict, severity, and residual drivers must be **100% numerically identical** between the legacy and chronological pipelines.
- **Deterministic Verification Test:**
  Stage 4B tests must include a deterministic comparison running a fixed set of $N = 500$ flows through both the legacy v1 pipeline and the modified chronological simulator, asserting $\Delta_{\text{score}} < 10^{-9}$ and identical verdicts on aligned flow records.

### 10.4 v2 Explainability Isolation & Optional Rollout
- **No 32-Dimensional Leakage:** `ThreatExplainer` in `src/explainability.py` operates on 32 dimensions, decomposing residuals across bidirectional features (e.g., `Flow Duration`, `Flow Bytes/s`, `ACK Flag Count`). Reusing v1 explanations for v2 is strictly prohibited and would violate the Pure One-Way mandate.
- **Adapter Capability Check:** `PureOneWayV2Adapter` currently implements scoring (`score()`, `predict()`) but lacks a feature attribution method.
- **Implementation Policy:**
  In Stage 4B, v2 explainability must be:
  1. Calculated strictly from the 26 forward squared residuals $\text{diff\_sq}_k = (X_{\text{scaled}, k} - X_{\text{rec}, k})^2$ mapped to `PURE_ONE_WAY_FEATURES`, OR
  2. Made **strictly optional** (with `v2_shadow["top_drivers"] = null`) until a dedicated `PureOneWayExplainer` class is separately specified and validated.

### 10.5 Empirical Latency Verification & SLA Enforcement
- **Labelled Estimates:** The projected ~25.0 ms/flow latency must remain clearly documented as an engineering estimate dominated by legacy v1 single-row DataFrame construction.
- **Post-Integration Benchmark Requirements:**
  Following Stage 4B integration, actual end-to-end wall-clock latency must be empirically measured over $N \ge 2,000$ flows, reporting:
  - Median latency (p50)
  - 95th Percentile latency (p95)
  - 99th Percentile latency (p99)
  - Maximum unthrottled throughput (flows/second without simulation sleep)

---

## 11. Final Recommendation: CONDITIONAL GO

**Auditor Decision:** **CONDITIONAL GO — implementation authorized only with the above safeguards.**

### Summary of Mandatory Safeguards:
1. **Buffer Overflow Safeguard:** Enforce explicit degraded/unavailable state for v2 when `max_buffer_capacity` is exceeded, resuming scoring only after a full 10-second clean window.
2. **Timestamp Nondecreasing Invariant:** Enforce $t_{i+1} \ge t_i$ with stable secondary tie-breaking, preserving simultaneous groups across batch boundaries via `flush=False`.
3. **v1 Regression Protection:** Deterministically prove v1 per-flow scoring invariance, separating expected alert sequence reordering from model scoring behavior.
4. **v2 Explainability Hygiene:** Prohibit reuse of v1 32-dimensional explanations; make v2 top drivers optional until a 26-dimensional forward residual explainer is validated.
5. **Empirical Benchmark SLA:** Measure and document real post-integration p50, p95, p99 latencies and unthrottled throughput.
6. **Strict Shadow Role:** v2 verdicts must remain purely observational in `live_stream_feed.json` without affecting production SOC alerts.

---

## 12. Working Tree & Git Status

```powershell
$ git status --short
?? archive/backups/
?? reports/PHASE_3_STAGE_4B_LIVE_INTEGRATION_AUDIT.md
```

- Strict read-only audit: No application code or test files were modified.
- Current HEAD: `707417b` on branch `feat/phase-2-pure-one-way-model`.
- Working tree clean of any tracked modifications (`git diff --check` clean with 0 diffs).
- No changes staged, committed, pushed, or merged.
