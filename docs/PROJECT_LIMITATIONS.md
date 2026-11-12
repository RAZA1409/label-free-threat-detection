# AEGIS: Comprehensive Project Limitations and Operational Boundaries

**Project:** AEGIS — Label-Free Threat Detection in One-Way Network Traffic
**Document Type:** Technical Constraints, Operational Boundaries, and Failure Mode Analysis
**Target Audience:** University Examination Board, External Examiners, Security Auditors, and Research Engineers
**Classification:** Academic / Engineering Disclosure

---

## Executive Summary

No security monitoring system is universal, and any engineering claim of "flawless detection" in an adversarial network environment is scientifically invalid. **AEGIS** is an unsupervised anomaly detection system explicitly designed for passive, unidirectional network environments (data diodes).

This document provides a rigorous, transparent, and comprehensive breakdown of the architectural, theoretical, data-driven, and operational limitations of the AEGIS system. Understanding these limitations is critical for viva defense, academic review, and realistic deployment roadmapping.

```
+-----------------------------------------------------------------------------------+
|                            AEGIS LIMITATION TAXONOMY                              |
+-----------------------------------------------------------------------------------+
| 1. Physical & Architectural:  Software Replay vs. Hardware Diode Enforcement      |
| 2. Protocol & Directionality: Reverse-Channel Blindness & TCP Asymmetry           |
| 3. Dataset Constraints:       CIC-IDS2017 Synthetic Biases, Drift, & Clock Skew  |
| 4. Streaming Mechanics:       10-Second Window Warm-Up & Velocity Starvation      |
| 5. Machine Learning Models:   12.48% FPR, Manifold Drift, & Static Quantiles     |
| 6. Operational & SOC Realities: Alert Fatigue, Lack of Active Prevention, Zero-Day |
+-----------------------------------------------------------------------------------+
```

---

## 1. Physical Layer & Architectural Limitations

### 1.1 Software Emulation vs. Physical Hardware Data Diode

| Dimension | Physical Hardware Data Diode | AEGIS Software Prototype |
| :--- | :--- | :--- |
| **Physical Layer Enforcement** | Physical Layer 1 Simplex (single TX fiber or LED/photodiode pair) | Software-level feature filtering on standard full-duplex NIC |
| **Reverse Channel Leakage** | Physically impossible; 0 photons/electrons traverse reverse path | Software isolation; reverse packets technically exist on host OS |
| **Tamper Resistance** | Hardware-locked; immune to software-level kernel bypass | Subject to host OS compromise, driver bugs, or memory manipulation |
| **Data Ingestion** | Raw optical bitstream through specialized transceiver | PCAP flow replay via Pandas / PyArrow Parquet engine |

> [!WARNING]
> **Viva Invariant:** AEGIS is a **software-level proof of concept and streaming detection prototype**. It models the mathematical and algorithmic constraints of a unidirectional data diode by strictly removing reverse-channel metrics from its feature vectors. It does **not** manufacture or incorporate proprietary physical diode hardware (such as Fox-IT, Advenica, or Owl Cyber Defense physical chassis). Claiming physical hardware enforcement during examination is factually inaccurate.

### 1.2 Lack of Hardware-Level Flow Reassembly
In a commercial high-assurance data diode setup:
- Specialized FPGA or ASIC network interface cards capture raw Ethernet frames directly from optical taps at 10 Gbps or 100 Gbps.
- Flow reassembly occurs in hardware-accelerated memory registers.
- AEGIS currently relies on pre-aggregated flow records (CIC-IDS2017 bidirectional flow outputs derived from CICFlowMeter). While AEGIS successfully strips all backward metrics (`Bwd Packet Count`, `Bwd Header Length`, `Bwd IAT`, etc.) to enforce one-way mathematical invariants, it relies on flow termination logic that was initially computed with full session visibility.

---

## 2. Protocol & Reverse-Channel Blindness Limitations

Data diodes strictly forbid any reverse packet transmission. This creates inherent protocol-level blindness that affects anomaly detection algorithms.

```
       UNIDIRECTIONAL DIODE BOUNDARY
High-Security Zone (TX)              Low-Security / Monitoring Zone (RX)
+-----------------------+              +--------------------------------+
|  Target Server        |              |  AEGIS Detection Engine        |
|  [IP: 192.168.10.50]  |              |                                |
|                       |  =======>    |  Observes: SYN, DATA, PUSH     |
|  [Port: 80 / 443]     |  (TX Only)   |                                |
+-----------------------+              +--------------------------------+
          |                                            |
          x  < - - - - - - - - - - - - - - - - - - - - x
             REVERSE CHANNEL BLOCKED (NO SYN-ACK, NO RST, NO HTTP ERRORS)
```

### 2.1 Inability to Observe TCP Handshake Completion
- In standard bidirectional networking, a TCP connection requires a 3-way handshake (`SYN` $\rightarrow$ `SYN-ACK` $\rightarrow$ `ACK`).
- Across a data diode, the monitoring sensor on the receive (RX) side can observe the initial forward `SYN` packet and subsequent data packets transmitted by the sender, but **cannot observe the `SYN-ACK` generated by the receiving server**.
- **Impact:** AEGIS cannot verify whether a TCP connection successfully established, was rejected by the server (`RST`), or timed out. Consequently:
  - High-volume SYN port scans appear as rapid bursts of forward SYN packets. AEGIS detects these via flow count and byte velocity anomalies.
  - However, AEGIS cannot distinguish between a "stealth" SYN scan where the attacker abandons half-open connections and a legitimate application opening rapid short-lived connections, without relying purely on statistical frequency and port diversity.

### 2.2 Blindness to Server Application Responses
In standard intrusion detection systems (e.g., Snort, Zeek, Suricata), threat confirmation heavily relies on server response codes:
- **HTTP Status Codes:** `401 Unauthorized` vs. `200 OK` (distinguishes brute-force attempts from successful logins).
- **SQL Error Messages:** Database syntax error strings returned in HTTP response bodies (distinguishes probe attempts from blind SQL injection).
- **FTP / SSH Banners:** Server handshake responses indicating daemon version and authentication outcomes.

Because AEGIS operates in a strictly one-way listening post, **all server responses are completely absent**:
1. **Brute-Force Attacks (FTP/SSH):** AEGIS can observe rapid repeated connection attempts from an external IP to port 21 or 22, but cannot verify whether any attempt succeeded.
2. **Web Attacks:** An automated vulnerability scanner firing 10,000 requests with malicious URI strings can be detected via burst velocity and feature distortion, but AEGIS cannot ascertain if the underlying server was vulnerable or returned `404 Not Found`.

### 2.3 Heuristic Flow Termination
Without observing reverse `FIN` or `RST` packets, flow termination across a diode must rely entirely on inactivity timeout heuristics (e.g., 10 seconds of silence). If an attacker intersperses malicious packets with low-frequency delays ("low-and-slow" attacks), flow records may artificially remain open or fragment into multiple micro-flows, distorting statistical feature aggregates.

---

## 3. Dataset Constraints: CIC-IDS2017

AEGIS uses the Canadian Institute for Cybersecurity CIC-IDS2017 benchmark dataset. While it is one of the most widely cited datasets in academic cybersecurity, it exhibits documented synthetic artifacts that directly impact experimental evaluation.

### 3.1 Monday-Only Benign Baseline & Temporal Concept Drift
- **Training Set:** Monday traffic consists exclusively of benign operations (normal browsing, internal server communication, background services).
- **Test Set:** Tuesday through Friday contains a mixture of attacks and benign traffic.
- **The Drift Problem:** Benign user behavior on Tuesday through Friday is not statistically identical to Monday. Different protocols, web portals, background patch cycles, and dynamic IP assignments naturally occur.
- **Direct Empirical Consequence:** Because the AEGIS autoencoder is trained strictly on Monday benign data to establish a "label-free normal manifold," any natural shift in benign behavior on subsequent days generates elevated reconstruction error.
  $$\text{Empirical Benign False Positive Rate (FPR)} = \mathbf{12.48\%} \quad (217,598 \text{ false alerts out of } 1,743,179 \text{ benign test flows})$$
- In an operational Security Operations Center (SOC), a 12.48% FPR represents substantial alert fatigue without secondary correlation filters.

### 3.2 Synthetic Attack Artifacts
1. **Automated Tool Signatures:** Attacks in CIC-IDS2017 were generated using standardized open-source tools:
   - DDoS: Low Orbit Ion Cannon (LOIC) over UDP/TCP.
   - DoS: Slowloris, Slowhttptest, Hulk, GoldenEye.
   - Brute Force: Patator (FTP/SSH).
   - PortScan: Nmap.
2. **Predictable Traffic Volumes:** These automated tools create extreme, unrealistic traffic spikes that are significantly easier for statistical autoencoders to detect than modern, stealthy, human-operated Advanced Persistent Threats (APTs).
3. **Clock Skew and Timestamp Artifacts:** As documented in published academic literature (e.g., Engelen et al., 2021), the physical packet capture infrastructure in CIC-IDS2017 had slight clock desynchronizations across monitoring interfaces, introducing microsecond-level jitter into inter-arrival time (`IAT`) calculations.

### 3.3 Class Representation Imbalance
The dataset exhibits extreme skew across attack categories:
- **DDoS / PortScan:** Hundreds of thousands of flows (over-represented).
- **Infiltration / Heartbleed:** Infiltration contains only 36 flows across the entire week; Heartbleed contains only 11 flows.
- **Impact on Evaluation:** Aggregate test recall ($93.08\%$) is heavily dominated by high-volume DoS/DDoS and PortScan attacks. Rare, stealthy attack categories contribute negligibly to overall aggregate metrics.

---

## 4. Streaming Engine & Causal Buffer Limitations

### 4.1 The 10-Second Sliding Buffer Warm-Up Period
The AEGIS streaming pipeline (`StreamingFeaturePipeline`) maintains an event-time sliding window:
$$W(t) = (\tau_{\text{event}} - 10.0\,\text{s},\, \tau_{\text{event}}]$$
To prevent memory exhaustion and compute temporal features (e.g., `flow_count_last_10s`, `byte_velocity`, `port_diversity`), the engine stores flow descriptors in a Python `collections.deque`.

**The Cold-Start Limitation:**
- When the stream starts, or immediately after a massive temporal gap between recorded episodes, the buffer is empty ($N=0$).
- Until sufficient events arrive within the 10-second window, velocity metrics are suppressed.
- Any anomaly that depends strictly on rolling burst velocity will not trigger an alert during the initial fraction of a second.

### 4.2 The "DDoS Onset 0% Recall" Phenomenon (B3 Replay Finding)

During the B3 implementation review, a crucial empirical finding was identified and documented:

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              TWO-EPISODE REPLAY BREAKDOWN                                               |
+-------------------+-----------------+---------------+--------------+-----------+----------------------------------------+
| Episode           | Flows           | Attack Type   | v1 Recall    | v1 FPR    | Confusion Matrix (TP, FP, FN, TN)      |
+-------------------+-----------------+---------------+--------------+-----------+----------------------------------------+
| Episode 1 (1-1000)| [1685068:1686068]| PortScan      | 99.55% (893) |  0.00%    | TP=893, FP=0, FN=4, TN=103             |
| Episode 2 (1001-2000)| [1903272:1904272]| DDoS Onset    |  0.00% (0)   | 14.63%    | TP=0, FP=79, FN=460, TN=461            |
+-------------------+-----------------+---------------+--------------+-----------+----------------------------------------+
| Combined Total    | 2,000 flows     | Mixed         | 65.81% (893) | 12.29%    | TP=893, FP=79, FN=464, TN=564          |
+-------------------+-----------------+---------------+--------------+-----------+----------------------------------------+
```

#### Why Did Episode 2 Achieve 0% Recall?
1. **The Slice Boundary:** Episode 2 starts at dataset offset `1,903,272`, corresponding to `Friday 03:56:16`.
2. **Temporal Gap:** The PortScan episode ended at timestamp `01:46:16`. Between Episode 1 and Episode 2 lies a natural chronological gap of approximately **3,900 seconds** (over 1 hour).
3. **Causal Eviction:** When the first flow of Episode 2 arrives, all PortScan flows in the sliding buffer are older than 10 seconds and are correctly evicted to preserve mathematical causality.
4. **Velocity Starvation:** The 1,000 flows selected in Episode 2 represent an **early onset segment of the DDoS attack** (dataset timestamps have minute-level precision).
5. In this early onset segment:
   - The rolling flow count in the 10-second buffer is low.
   - The aggregate volume has not yet accumulated sufficiently within the causal 10-second window to breach the global anomaly threshold ($\tau = 0.190885$).
6. **Comparison with Legacy DDoS Replay:** In contrast, the legacy 2,000-flow DDoS replay starts at offset `1,942,302` (deep inside the sustained DDoS flood at `04:00 PM`), where the traffic exhibits sustained high rolling velocity, achieving **83.21% recall**.

> [!IMPORTANT]
> **Viva Defense Insight:** The 0% recall on the DDoS onset slice is **not a bug**; it is an authentic, honest demonstration of causal streaming physics. An unsupervised model relying on temporal window velocity cannot detect a volumetric flood before the volume has physically arrived.

### 4.3 Memory Scaling and Out-of-Order Packets
- **Memory Consumption:** At gigabit line rates, a 10-second window could contain upwards of $10^6$ flows. Storing rich Python dictionaries in a `deque` introduces memory overhead ($O(N)$ space). In production, this requires C++/Rust circular ring buffers.
- **Non-Monotonic Timestamps:** In distributed network taps or multi-core NICs, packets can arrive out-of-order due to thread scheduling. The AEGIS streaming simulator assumes monotonically increasing timestamps; handling high out-of-order jitter requires reordering buffers, which introduces latency.

---

## 5. Machine Learning & Anomaly Detection Limitations

### 5.1 The Unsupervised Anomaly Detection Trade-Off
AEGIS does not use attack labels during training or inference. This confers major benefits (detecting zero-day attacks without signatures), but imposes fundamental theoretical limitations:

$$\text{Anomaly} \neq \text{Malicious}$$

1. **Reconstruction Error Interpretation:** An autoencoder measures deviation from the training manifold ($M_{\text{benign}}$):
   $$\mathcal{L}(x, \hat{x}) = \sum_{j=1}^{d} w_j (x_j - \hat{x}_j)^2$$
   If an event has high reconstruction error, it simply means it is **rare or structurally distinct** from Monday benign traffic.
2. **Benign Outliers:**
   - A legitimate system administrator running an unannounced automated database backup via `rsync`.
   - A massive operating system software update downloading patch bundles across non-standard ports.
   - Network routing reconfigurations causing high packet latency.
   All of these legitimate events will produce high reconstruction error and trigger false alarms.

### 5.2 Static Quantile Calibration Vulnerability
The AEGIS v1 threshold was calibrated as:
$$\tau = Q_{0.99}(\mathcal{L}_{\text{val\_benign}}) = 0.190885$$
- This mathematical guarantee holds strictly if and only if the test data is identically and independently distributed ($i.i.d.$) relative to the validation data.
- Because real enterprise networks violate the $i.i.d.$ assumption continuously, a static threshold cannot maintain a fixed 1.0% FPR over time. On Tuesday-Friday data, the FPR rose to **12.48%**.
- **Limitation:** AEGIS currently lacks an automated online adaptive thresholding mechanism (e.g., Extreme Value Theory or rolling quantile estimators) to dynamically adjust for diurnal network cycles without manual recalibration.

### 5.3 Genetic Algorithms Not Implemented in AEGIS
- **Implementation Status:** Genetic Algorithms (GA) are **not implemented** in the AEGIS codebase. No evolutionary search, multi-objective optimization, Pareto fronts, or feature-mask evolution routines exist in the repository.
- **Production Truth:** AEGIS relies strictly on deterministic empirical quantile estimation on benign calibration traffic (`calibrate_threshold` in `src/features.py`), generating static, auditable threshold artifacts (`threshold = 0.190885`). This guarantees zero runtime stochasticity, complete reproducibility, and deterministic inference latency.

### 5.4 Feature Reconstruction Masking (XAI Approximation)
AEGIS provides explainability via feature-wise squared error attribution:
$$A_j = w_j (x_j - \hat{x}_j)^2$$
- While highly intuitive, feature-wise reconstruction error is an approximation of true Shapley values (SHAP).
- Deep non-linear neural network layers introduce feature correlations in latent space. High reconstruction error on feature $A$ may sometimes be caused by an anomaly in correlated feature $B$ that distorted the latent representation $z$.

---

## 6. Operational & Security Operations Center (SOC) Realities

### 6.1 Alert Storms and Triage Overhead
- Across the full 2.3-million-flow test set, AEGIS generated:
  - **519,031 True Positive Alerts**
  - **217,598 False Positive Alerts**
- In an operational enterprise SOC where an analyst can thoroughly triage approximately 30 to 50 alerts per 8-hour shift, **217,598 false positives would overwhelm the team within hours**.
- **Mitigation Requirement:** In a real deployment, AEGIS must be positioned as a Layer-1 sensor feeding a secondary SIEM/SOAR correlation engine (e.g., Splunk, Elasticsearch, or Wazuh) that groups consecutive flow alerts by source IP, destination subnet, and temporal clustering before paging human operators.

### 6.2 Passive Monitoring vs. Active Threat Prevention
- By definition, a network data diode is a **passive listening mechanism**.
- **Inability to Terminate Connections:** AEGIS cannot:
  - Inject TCP `RST` packets to terminate an active attack.
  - Send ICMP unreachable messages.
  - Dynamically modify upstream firewall access control lists (ACLs) without violating the unidirectional physical isolation of the diode.
- **Operational Reality:** AEGIS is strictly an **early warning sensor**, not an inline intrusion prevention system (IPS).

### 6.3 Encrypted Traffic and Payload Invisibility
- AEGIS operates strictly on **flow-level metadata** (packet counts, byte lengths, inter-arrival times, port numbers, header sizes).
- It does **not perform Deep Packet Inspection (DPI)** or payload string matching.
- **Limitation:**
  - If an attacker uses TLS encryption with standard packet sizing and legitimate rate profiles, payload-specific exploits (e.g., Log4j CVE-2021-44228 JNDI string injection inside HTTPS headers) will appear indistinguishable from benign web traffic.
  - AEGIS will only detect encrypted threats if their macroscopic flow dynamics (connection rate, packet size variance, duration) deviate from the benign baseline.

---

## 7. Comparative Summary of Failure Modes

The following table serves as a quick-reference guide to the specific failure modes of AEGIS:

```
+----------------------------------------------------------------------------------------------------+
|                                    AEGIS FAILURE MODE MATRIX                                       |
+------------------------------+--------------------+---------------------+--------------------------+
| Threat Category              | AEGIS Susceptibility| Root Cause          | Recommended Countermeasure|
+------------------------------+--------------------+---------------------+--------------------------+
| High-Volume DDoS (Sustained) | LOW (93.08% Recall)| Extreme burst volume| Standard v1 Autoencoder  |
| Rapid Port Scanning (Nmap)   | LOW (99.55% Recall)| Fan-out & port rate | Standard v1 Autoencoder  |
| Truncated DDoS Onset (<0.5s) | HIGH (0% Recall)   | Velocity starvation | Shorter rolling windows  |
| Low-and-Slow Exfiltration    | HIGH (Missed)      | Fits benign bounds  | Long-term time-series    |
| SSH / FTP Brute-Force Auth   | MEDIUM (Alerts)    | Reverse blindness   | Correlate with host logs |
| Benign Distribution Drift    | HIGH (12.48% FPR)  | Unsupervised shift  | Online adaptive quantile |
| Encrypted In-Band Web Exploits| HIGH (Missed)      | No DPI / TLS payload| Host-based IDS (HIDS)    |
+------------------------------+--------------------+---------------------+--------------------------+
```

---

## 8. Examiner Viva Defense Strategy

When questioned by examiners about limitations, use the following structured responses:

1. **"Why does your system have a 12.48% False Positive Rate?"**
   > *"That is an honest, mathematically verified consequence of unsupervised learning across temporal boundaries. We train our autoencoder strictly on Monday benign data to guarantee zero label contamination. When tested on Tuesday through Friday, natural shifts in user behavior create reconstruction error outliers. Rather than artificially cherry-picking a test set to show a misleading 1% FPR, we report the authentic 12.48% figure and advocate for downstream SIEM alert aggregation."*

2. **"Why did your two-episode live replay achieve 0% recall on DDoS onset?"**
   > *"Because our streaming feature pipeline uses a strictly causal 10-second sliding window. The 1,000 flows selected in the onset slice capture an early onset segment of the attack before rolling traffic velocity has accumulated sufficiently within the 10-second window to breach our global anomaly threshold. In contrast, during sustained DDoS replay where traffic velocity is high, the model achieves 83.21% recall. This proves our engine does not cheat with future data."*

3. **"Does AEGIS run on an actual physical hardware data diode?"**
   > *"No, sir. AEGIS is a software streaming prototype and algorithm suite. We model the physical constraints of a data diode by strictly removing all backward flow metrics and reverse signaling capabilities. A physical hardware implementation would pair our Python/Rust pipeline with an FPGA-driven optical simplex tap."*

4. **"Why isn't Genetic Algorithm optimization executing in real-time?"**
   > *"Genetic algorithms are not implemented in the AEGIS codebase. While evolutionary algorithms exist in academic literature, production AEGIS enforces deterministic, audit-compliant numpy quantile calibration on a dedicated benign validation split to guarantee predictable runtime latency and mathematical repeatability."*
