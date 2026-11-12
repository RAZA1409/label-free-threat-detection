# AEGIS: Label-Free Threat Detection in One-Way Network Traffic
## Engineering & Technical Report — SIH26145 Prototype

---

### Executive Metadata
- **Project Title:** AEGIS — Label-Free Threat Detection in One-Way Network Traffic
- **Problem Statement ID:** SIH26145 (Smart India Hackathon / National Cybersecurity Research)
- **Domain:** Artificial Intelligence & Machine Learning / Network Security & Industrial Data Diodes
- **Target Deployment:** Air-Gapped Supervisory Control and Data Acquisition (SCADA) / Critical National Infrastructure (CNI) Ingress Monitoring
- **Repository Branch:** `main`
- **Execution Mode:** Software-Based Optical Tap / Data Diode Traffic Replay Simulator

---

## 1. Abstract

Critical National Infrastructure (CNI), industrial control systems (ICS), and intelligence enclaves frequently deploy physical data diodes—hardware devices that enforce strict simplex, unidirectional network communication at the physical layer (Layer 1)—to prevent exfiltration and lateral traversal. However, unidirectional network conduits present an acute challenge to conventional Network Intrusion Detection Systems (NIDS): signature-based tools (e.g., Snort, Suricata) and stateful machine learning detectors predominantly rely on bidirectional flow state machines (e.g., TCP handshakes, round-trip times, server response banners, and return payload telemetry). Furthermore, novel zero-day exploits and evasion attacks lack historical signatures, necessitating label-free (unsupervised) anomaly detection.

**AEGIS** is an unsupervised, label-free threat detection framework engineered specifically for simplex, one-way network traffic. Operating on forward-only flow vectors augmented by a causal, 10-second rolling temporal velocity context, AEGIS detects malicious activity without requiring attack signatures, reverse-channel telemetry, or bidirectional TCP handshakes. The production champion model (v1) utilizes a 32-dimensional forward-augmented feature vector fed into a deep undercomplete autoencoder with weighted Mean Squared Error (MSE), achieving a **93.08% overall attack recall** at an operational test False Positive Rate (FPR) of **12.48%** ($0.8020$ F1-score, $0.9538$ ROC-AUC) across 2,300,825 held-out CIC-IDS2017 network flows. To explore absolute physical Layer-1 diode compliance, an experimental research candidate (v2) implements a 26-dimensional pure-forward pipeline excluding all bidirectional aggregations.

AEGIS incorporates a fault-tolerant, stateful streaming simulation engine featuring causal event-time buffer eviction, backpressure isolation, rolling checkpoint persistence, real-time explainability (XAI) via reconstruction error attribution, and a five-view Security Operations Center (SOC) dashboard. This report details the complete engineering architecture, theoretical foundations, mathematical derivations, empirical evaluations, and operational limitations of the AEGIS prototype.

---

## 2. Introduction & Background

### 2.1 The Critical Infrastructure Conundrum
Modern industrial operations—such as nuclear power plants, electrical power distribution grids, municipal water treatment facilities, and naval weapons platforms—increasingly connect operational technology (OT) networks to enterprise information technology (IT) networks to facilitate monitoring and operational analytics. Because enterprise networks connect to the public Internet, traditional bidirectional software firewalls provide insufficient perimeter defense; misconfigurations, zero-day vulnerabilities in firewall firmware, or compromised credentials can permit bidirectional command-and-control (C2) tunnels and remote unauthorized actuator manipulation.

To achieve mathematically verifiable perimeter security, security architects employ **physical data diodes**. A data diode physically restricts transmission to a single direction using an optical transmitter (LED or laser) on the sender side connected via a severed fiber cable to an optical receiver (photodiode) on the listening side, with no physical return path.

```
+--------------------------+       Physical Optical Fiber      +--------------------------+
|  High-Security Enclave   | --------------------------------> |  Corporate IT / SOC Hub  |
|  (Nuclear / SCADA Core)  |    (Transmit Laser Only - TX)     |  (Analytics Collector)   |
|   [Optical Transmitter]  |   ==============================> |    [Photodiode - RX]     |
+--------------------------+     NO RETURN PATH (No RX)        +--------------------------+
```

### 2.2 The Threat Detection Problem on One-Way Conduits
While hardware data diodes physically guarantee zero data exfiltration from the high-security network, they introduce severe operational blind spots for downstream monitoring systems:
1. **Absence of Handshake Telemetry:** Downstream receivers cannot observe TCP SYN-ACK or ACK responses, server greeting banners, or return window size advertisements.
2. **Asymmetric Flow Metadata:** Flow meters deployed on the receiving side observe only outbound packets. Metrics that depend on client-server packet pairing (e.g., Round Trip Time (RTT), Flow Bytes/s, Forward-to-Backward packet ratios) collapse or corrupt.
3. **Emergence of Zero-Day Exploits:** Air-gapped networks are targeted by advanced persistent threats (APTs) using undocumented vulnerabilities (e.g., Stuxnet, Industroyer, BlackEnergy). Supervised classifiers trained on cataloged attacks cannot generalize to zero-day signatures.

AEGIS resolves this dilemma by modeling the normal statistical and temporal manifold of benign forward-only traffic using unsupervised autoencoders, flagging significant reconstruction deviations as threats.

---

## 3. Problem Statement & Objectives

### 3.1 Problem Statement (SIH26145)
Given an unlabelled network traffic stream traversing a one-way communication tap where return packets are physically absent, design and implement an end-to-end unsupervised threat detection engine capable of:
1. Extracting strictly unidirectional features without reverse-channel dependencies.
2. Learning normal behavioral distributions exclusively from unlabelled benign calibration data.
3. Performing real-time anomaly detection and scoring without ground-truth labels during inference.
4. Explaining the anatomical drivers of flagged anomalies to SOC operators in real time.
5. Providing bounded latency, memory safety, and operational resilience against bursty traffic.

### 3.2 Formal Mathematical Formulation
Let $\mathcal{F} = \{f_1, f_2, \dots, f_N\}$ represent a chronologically ordered sequence of network flows arriving at a one-way capture interface. Each flow $f_i$ is characterized by an arrival timestamp $t_i \in \mathbb{R}^+$ and a raw forward telemetry tuple $\mathbf{x}_i \in \mathbb{R}^d$.

We define a causal feature extraction mapping:
$$\Phi: (\mathbf{x}_i, \mathcal{H}_{t_i}) \mapsto \mathbf{z}_i \in \mathbb{R}^D$$
where $\mathcal{H}_{t_i} = \{f_k \mid t_i - \Delta t < t_k \le t_i\}$ is the historical flow buffer within a causal temporal lookback window of length $\Delta t = 10\text{ seconds}$, ensuring $\Phi$ never conditions on future observations ($t_k > t_i$).

We seek an unsupervised anomaly scoring function $S(\mathbf{z}_i) \in [0, \infty)$ parameterized by an undercomplete autoencoder $\mathcal{M}_\theta = \mathcal{D}_{\theta_D} \circ \mathcal{E}_{\theta_E}$ trained exclusively on benign traffic $\mathcal{D}_{\text{train}} = \{\mathbf{z}_k \mid y_k = \text{BENIGN}\}$ such that:
$$S(\mathbf{z}_i) = \|\mathbf{z}_i - \mathcal{M}_\theta(\mathbf{z}_i)\|_W^2$$
An alert decision $\hat{y}_i \in \{0, 1\}$ is rendered by comparing $S(\mathbf{z}_i)$ against an empirically calibrated decision threshold $\tau$:
$$\hat{y}_i = \mathbb{I}(S(\mathbf{z}_i) > \tau)$$
where $\tau$ is established on an independent benign calibration split $\mathcal{D}_{\text{cal}}$ to satisfy a strict calibration target False Positive Rate:
$$P(S(\mathbf{z}) > \tau \mid \mathbf{z} \in \mathcal{D}_{\text{cal}}) \le \alpha$$

---

## 4. System Architecture

The AEGIS architecture comprises five decoupled pipelines organized into a layered dataflow topology:

```
+----------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION & REPLAY                                       |
|  outputs/experiment/test_tue_fri.parquet (2,300,825 Chronological Flows, July 4-7, 2017)           |
|  - Option B-1 Two-Episode Stream Replay: [1,685,068:1,686,068] (PortScan) + [1,903,272:1,904,272] (DDoS) |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                         STREAMING FEATURE ENGINE (src/streaming_features.py)                       |
|  - Flow Sanitization (Sentinel clipping, Inf/NaN remediation, Protocol encoding)                  |
|  - StreamingCausalContext (Sliding Event-Time Window: (t - 10s, t])                                |
|  - Temporal Velocity Derivation: Rolling_Flow_Velocity_10s, Rolling_Bytes_Velocity_10s, etc.     |
|  - Memory-Bounded Deque (Max Capacity: 10,000 flows; Degraded State Fallback)                     |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                         DUAL INFERENCE ENGINE (src/simulate_stream.py)                             |
|                                                                                                    |
|   +----------------------------------------------+    +------------------------------------------+ |
|   | PRODUCTION CHAMPION (v1 - 32 Dim)            |    | SHADOW RESEARCH CANDIDATE (v2 - 26 Dim)   | |
|   | - 29 Forward Features + 3 Rolling Velocity   |    | - 23 Pure Forward + 3 Rolling Velocity   | |
|   | - Robust Median Imputation + Z-Score Scaling |    | - Log1p Feature Transform + Normalization| |
|   | - Deep Autoencoder [32 -> 24 -> 8 -> 24 -> 32]|    | - Pure Autoencoder [26 -> 18 -> 8 -> 18 -> 26]|
|   | - Weighted MSE Loss                          |    | - Unweighted MSE Loss                    | |
|   | - Threshold: tau = 0.190885 (alpha = 0.2%)   |    | - Primary: tau = 0.314036 (alpha = 0.2%) | |
|   |                                              |    | - Reference: tau = 0.099018 (alpha = 1%) | |
|   | [ACTIVE ALERT AUTHORITY FOR TELEMETRY & SOC] |    | [SILENT SHADOW METRICS ONLY]             | |
|   +----------------------------------------------+    +------------------------------------------+ |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                   EXPLAINABLE AI (XAI) & STATE CHECKPOINTING (src/explainability.py)               |
|  - Feature Attribution: Normalized Reconstruction Residual Decomposition (Top-5 Driving Features)  |
|  - Heuristic SOC Narrative Synthesis + Gemini 1.5 Forensic Explanations                             |
|  - State Persistence: Rolling JSON Checkpoints (src/checkpoint.py)                                 |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                         OPERATIONAL SOC DASHBOARD (dashboard/index.html & server.py)               |
|  - View 1: Executive KPI Radar & Real-Time Incident Stream                                         |
|  - View 2: Threat Radar & Spatial Dimensionality Decomposition                                     |
|  - View 3: Live Optical Diode Threat Feed Stream                                                   |
|  - View 4: Forensic XAI Inspector (Attribution Waterfall & Automated Analyst Narratives)          |
|  - View 5: Model Calibration Sandbox (Empirical Discrete Operating Point Calibration & H2H)       |
+----------------------------------------------------------------------------------------------------+
```

---

## 5. Dataset Description & Partitioning Methodology

### 5.1 CIC-IDS2017 Dataset
AEGIS is developed, calibrated, and benchmarked on the Canadian Institute for Cybersecurity **CIC-IDS2017** benchmark dataset. Generated at the University of New Brunswick, CIC-IDS2017 comprises five consecutive days of complete packet capture (PCAP) and bidirectional flow data generated by the `CICFlowMeter` utility.

The dataset captures authentic background network behavior alongside twelve diverse attack families across a five-day operational capture window:
- **Monday, July 3, 2017:** 100% Benign background traffic (normal workstation activities, web browsing, internal file transfers).
- **Tuesday, July 4, 2017:** Benign traffic mixed with Brute Force attacks (FTP-Patator, SSH-Patator).
- **Wednesday, July 5, 2017:** Benign traffic mixed with DoS attacks (DoS Slowhttptest, DoS slowloris, DoS Hulk, DoS GoldenEye) and Heartbleed.
- **Thursday, July 6, 2017:** Benign traffic mixed with Web Attacks (SQL Injection, XSS, Brute Force) and Infiltration.
- **Friday, July 7, 2017:** Benign traffic mixed with Botnet (ARES), PortScan, and massive volumetric DDoS (LOIC).

### 5.2 Strict Chronological Split
In contrast to standard machine learning studies that shuffle datasets randomly across days—causing extreme temporal data leakage and artificial benchmark inflation—AEGIS enforces a **strict, time-monotonic temporal split**:

```
Monday 01:00:01                      Monday 10:44:35          Monday 12:59:58     Tuesday 01:00:00                     Friday 12:59:00
+-------------------------------------------+------------------------+             +-------------------------------------------------+
|         TRAINING BENIGN PARTITION         |  CALIBRATION BENIGN    |             |          OPERATIONAL TEST PARTITION             |
|              (363,597 Flows)              |    (166,321 Flows)     |             |             (2,300,825 Flows)                   |
|           Unsupervised Training           | Threshold Calibration  |             |  1,743,179 Benign + 557,646 Mixed Attack Flows  |
|            (100% PURE BENIGN)             |   (100% PURE BENIGN)   |             |           (Multi-Day Temporal Drift)            |
+-------------------------------------------+------------------------+             +-------------------------------------------------+
```

1. **Training Partition (`train_benign.parquet`):**
   - Window: Monday, July 3, 2017, `01:00:01` to `10:44:33`.
   - Size: 363,597 flows (100% Benign).
   - Usage: Fitting feature scalers, median imputers, and training autoencoder weights.
2. **Calibration Partition (`calibration_benign.parquet`):**
   - Window: Monday, July 3, 2017, `10:44:35` to `12:59:58`.
   - Size: 166,321 flows (100% Benign).
   - Usage: Determining empirical decision thresholds ($\tau$) corresponding to target false alarm rates ($\alpha$). Zero model weights are tuned on this set.
3. **Operational Test Partition (`test_tue_fri.parquet`):**
   - Window: Tuesday, July 4, `01:00:00` through Friday, July 7, 2017, `12:59:00`.
   - Size: 2,300,825 flows (1,743,179 Benign flows + 557,646 Attack flows across 14 categories).
   - Usage: Evaluating realistic operational generalization across multi-day distribution drift.

---

## 6. Data Cleaning & Feature Engineering

### 6.1 Data Hygiene & Sanitization
Raw network flow records frequently contain protocol artifacts, measurement errors, and platform-dependent corruptions. The module `src/features.py` enforces five sanitization rules:
1. **Duplicate Feature Removal:** Raw CIC-IDS2017 files contain duplicate columns (`Fwd Header Length` and `Fwd Header Length.1`). The duplicate is dropped deterministically.
2. **Mojibake & Encoding Normalization:** The dataset exhibits character encoding corruptions in attack labels (e.g., `Web Attack  Brute Force` with UTF-8 byte sequences `\x96` or `\xe2\x80\x93`). The function `sanitize_labels()` normalizes all variants to standard ASCII strings (`Web Attack - Brute Force`).
3. **Negative Sentinel Handling:** Flow duration metrics and TCP initial window sizes occasionally contain `-1` or negative values due to integer sign-bit flips in the Java capture agent. Negative values in non-negative physical dimensions are clamped to `0.0`.
4. **Infinite & NaN Imputation:** Division-by-zero during rate calculations generates positive/negative infinities (`np.inf`). These are mapped to `np.nan` and imputed using training set medians ($\text{median}(X_{\text{train}})$) to preserve robustness against extreme outliers.

### 6.2 Feature Set Evolution: v1 Champion vs. v2 Pure One-Way

#### The Hidden Reverse-Channel Leakage in Flow Datasets
During Phase 1 architectural audits (`reports/PHASE_1_ONE_WAY_COMPLIANCE_REPORT.md`), deep code inspection of the `CICFlowMeter` Java engine revealed that several features conventionally labeled as forward features implicitly depend on reverse-channel packets:
- `Flow Duration`: Calculated as $t_{\text{last\_packet}} - t_{\text{first\_packet}}$. In **77.4%** of bidirectional flows, the final packet is a reverse server acknowledgment (ACK/FIN).
- `Flow Bytes/s`: The numerator sums forward and backward bytes: $(\text{Bytes}_{\text{fwd}} + \text{Bytes}_{\text{bwd}}) / \text{Duration}$.
- `Fwd Packets/s`: While the numerator is forward packet count, the denominator is the bidirectional `Flow Duration`.
- `ACK / FIN / RST / PSH / URG Flag Counts`: Computed across all packets in the bidirectional flow conversation.

#### Production Champion v1 (32 Features)
The production model preserves the established feature set to maintain operational continuity, pairing 29 forward-dominated features with 3 rolling temporal features:
- **Base Flow Features (29):** Forward packet counts, forward byte totals, `Fwd Header Length`, `Avg Fwd Segment Size`, `min_seg_size_forward`, forward packet length statistics (Max, Min, Mean, Std), forward inter-arrival times (Total, Mean, Std, Max, Min), `Init_Win_bytes_forward`, `Fwd PSH Flags`, `Fwd URG Flags`, `SYN Flag Count`, `FIN Flag Count`, `RST Flag Count`, `ACK Flag Count`, `PSH Flag Count`, `URG Flag Count`, `Flow Duration`, `Flow Bytes/s`, `Fwd Packets/s`.
- **Temporal Velocity Features (3):** `Rolling_Flow_Velocity_10s`, `Rolling_Bytes_Velocity_10s`, `Rolling_Single_Pkt_Ratio_10s`.

#### Experimental Candidate v2 (26 Pure Forward Features)
To achieve mathematical purity on physical Layer-1 data diodes where zero reverse bits can be observed, v2 purges all eight contaminated features and introduces pure forward rate derivations:
$$\text{Fwd\_Bytes\_Per\_Sec} = \frac{\text{Total Length of Fwd Packets}}{\max(\text{Fwd IAT Total} \times 10^{-6}, 10^{-6})}$$
$$\text{Fwd\_Packets\_Per\_Sec} = \frac{\text{Total Fwd Packets}}{\max(\text{Fwd IAT Total} \times 10^{-6}, 10^{-6})}$$
The resulting 26 features comprise:
1. **Forward Volume & Packet Counts (6):** `Total Fwd Packets`, `Total Length of Fwd Packets`, `act_data_pkt_fwd`, `Subflow Fwd Packets`, `Subflow Fwd Bytes`, `Fwd Header Length`.
2. **Forward Segment & TCP Parameters (3):** `min_seg_size_forward`, `Avg Fwd Segment Size`, `Init_Win_bytes_forward`.
3. **Forward Packet Length Distributions (4):** `Fwd Packet Length Max`, `Fwd Packet Length Min`, `Fwd Packet Length Mean`, `Fwd Packet Length Std`.
4. **Forward Timing & Inter-Arrival Times (5):** `Fwd IAT Total`, `Fwd IAT Mean`, `Fwd IAT Std`, `Fwd IAT Max`, `Fwd IAT Min`.
5. **Pure Forward Directional Rates (2):** `Fwd_Bytes_Per_Sec`, `Fwd_Packets_Per_Sec`.
6. **Forward Directional TCP Flags (3):** `Fwd PSH Flags`, `Fwd URG Flags`, `SYN Flag Count`.
7. **Rolling Temporal Velocity Features (3):** `Rolling_Flow_Velocity_10s`, `Rolling_Bytes_Velocity_10s`, `Rolling_Single_Pkt_Ratio_10s`.

---

## 7. Model Architecture & Theoretical Principles

### 7.1 Autoencoder Anomaly Detection Theory
An undercomplete autoencoder is a feedforward neural network trained to reconstruct its input through an informational bottleneck. Let the encoder $\mathcal{E}: \mathbb{R}^D \to \mathbb{R}^k$ and decoder $\mathcal{D}: \mathbb{R}^k \to \mathbb{R}^D$ be defined with bottleneck capacity $k \ll D$:
$$\mathbf{h} = \sigma(W_e \mathbf{z} + \mathbf{b}_e)$$
$$\hat{\mathbf{z}} = \sigma(W_d \mathbf{h} + \mathbf{b}_d)$$
When trained exclusively on benign traffic $\mathcal{D}_{\text{train}}$, the network optimizes its projection manifold to capture the normal correlations, volume ratios, and inter-arrival timing patterns of benign protocols.

Because malicious flows (e.g., rapid port scanning probes, volumetric DDoS floods, buffer overflows) exhibit anomalous feature correlations that violate the learned low-dimensional manifold, the decoder fails to accurately reconstruct them, producing an elevated reconstruction residual:
$$\mathbf{e} = \mathbf{z} - \hat{\mathbf{z}}$$

### 7.2 Champion v1 Model Specification
- **Input Dimension:** $D = 32$.
- **Hidden Topologies:** Fully connected multi-layer perceptron (`sklearn.neural_network.MLPRegressor`):
  $$32 \longrightarrow 24 \longrightarrow 8 \text{ (Latent Bottleneck)} \longrightarrow 24 \longrightarrow 32$$
- **Activation Functions:** ReLU on hidden layers; linear activation on output layer.
- **Loss Formulation:** Weighted Mean Squared Error:
  $$\mathcal{L}_{\text{WMSE}}(\mathbf{z}, \hat{\mathbf{z}}) = \sum_{j=1}^{32} w_j (z_j - \hat{z}_j)^2$$
  where weights $w_j$ penalize reconstruction errors on temporal velocity and forward packet counts more heavily (derived from calibration inverse-variance residuals).
- **Trained Parameters:** Model serialized via joblib at `models/temporal_autoencoder_model.joblib`.
- **Operating Threshold:** $\tau = 0.190885$, calibrated at $\alpha = 0.002$ ($0.2\%$) on `calibration_benign.parquet`.

### 7.3 Candidate v2 Model Specification
- **Input Dimension:** $D = 26$.
- **Hidden Topologies:** Fully connected symmetric compression (`sklearn.neural_network.MLPRegressor`):
  $$26 \longrightarrow 20 \longrightarrow 8 \text{ (Latent Bottleneck)} \longrightarrow 20 \longrightarrow 26$$
- **Preprocessing Pipeline:** `Log1p` transformation on heavy-tailed physical features followed by standard Z-score normalization:
  $$\tilde{x}_j = \log(1 + \max(0, x_j))$$
- **Loss Formulation:** Unweighted MSE:
  $$\mathcal{L}_{\text{MSE}}(\mathbf{z}, \hat{\mathbf{z}}) = \frac{1}{26} \sum_{j=1}^{26} (z_j - \hat{z}_j)^2$$
- **Serialized Artifacts:** `models/pure_one_way_v2/model.joblib`, `pipeline.joblib`.
- **Dual Operating Thresholds:**
  - **Primary Operating Point:** $\tau = 0.314036$ (calibrated at $\alpha = 0.2\%$, prioritizing minimum operational FPR).
  - **High-Recall Reference Point:** $\tau = 0.099018$ (calibrated at $\alpha = 1.0\%$, achieving recall parity with v1).

### 7.4 Threshold Calibration Methodology
Rather than assuming parametric Gaussian distribution of reconstruction errors, AEGIS determines thresholds empirically via quantile estimation on held-out benign calibration traffic:
$$\tau_\alpha = \text{Quantile}_{1 - \alpha}\left( \left\{ S(\mathbf{z}) \mid \mathbf{z} \in \mathcal{D}_{\text{cal}} \right\} \right)$$
For $N_{\text{cal}} = 166,321$ calibration flows, the $\alpha = 0.002$ cutoff yields exactly $\lfloor 166,321 \times 0.002 \rfloor = 332$ calibration outliers.

---

## 8. Genetic Algorithm Optimization: Status & Literature Context

### 8.1 Implementation Status in AEGIS
In the AEGIS codebase, feature weights and operational thresholds are determined strictly through deterministic algorithms:
1. Feature scaling is fitted via training medians and standard deviations (`LogPreprocessingPipeline`).
2. Anomaly thresholds are established via exact empirical quantile estimation (`calibrate_threshold` in `src/features.py`).
3. Loss function weights in v1 were derived deterministically from inverse-variance residual bounds during training.

**Genetic Algorithms (GA) are NOT implemented in the AEGIS codebase.** No evolutionary optimization, Pareto fronts, chromosome definitions, or feature-mask evolution routines exist in the repository. AEGIS deliberately avoids stochastic metaheuristics in favor of deterministic, auditable calibration to ensure mathematical reproducibility during university evaluation and critical infrastructure deployment.

### 8.2 Academic Literature Context: Theoretical Evolutionary Formulations
In academic literature, evolutionary search is often discussed as an alternative theoretical formulation for intrusion detection threshold exploration:
- **Theoretical Concept:** Representing feature inclusion masks or candidate thresholds as evolutionary parameters.
- **Why AEGIS Does Not Use GA:** In production and critical infrastructure contexts, evolutionary algorithms introduce non-deterministic outcomes, runtime stochasticity, and heavy compute overhead. AEGIS guarantees deterministic auditability by computing exact numpy empirical quantiles:
  $$\tau = \text{np.quantile}(\mathcal{L}_{\text{val\_benign}}, 1 - \alpha)$$

*Note for University Viva:* The candidate must clearly state that Genetic Algorithms are **not implemented** in AEGIS, and that AEGIS achieves its 93.08% recall using deterministic quantile calibration on benign validation traffic.

---

## 9. Real-Time Streaming Engine & Reliability Engineering

The streaming subsystem (`src/streaming_features.py`, `src/simulate_stream.py`) simulates a physical optical diode tap ingesting network flows chronologically.

```
Incoming Stream Batch (e.g., 50 flows)
                  │
                  ▼
   +------------------------------+
   | Flow Quarantine Manager      | ---> Malformed / Bad Timestamp Flows -> Quarantine Log
   +------------------------------+
                  │ (Clean Flows)
                  ▼
   +------------------------------+
   | Streaming Causal Context     | <--- Event-Time Eviction: Purge flows where t_k <= t_curr - 10.0s
   | - Bounded Deque (Max 10,000) |
   +------------------------------+
                  │
                  ▼
   +------------------------------+
   | Temporal Feature Calculation | ---> Derives Rolling Velocity Metrics (Flows/sec, Bytes/sec)
   +------------------------------+
                  │
                  ▼
   +------------------------------+
   | Dual Inference Engine        | ---> v1 (Alert Authority) + v2 (Shadow Only)
   +------------------------------+
                  │
                  ▼
   +------------------------------+
   | Telemetry Throttler          | ---> Disk Write to live_stream_feed.json (Every 0.25s / 20 flows)
   +------------------------------+
```

### 9.1 Streaming Causal Context & Temporal Lookback
Network traffic velocity cannot be measured from an isolated, stateless flow record. The class `StreamingCausalContext` maintains an in-memory sliding buffer of recent flows within a 10-second lookback window:
$$\mathcal{W}(t_{\text{current}}) = \{f \in \text{Buffer} \mid t_{\text{current}} - 10.0\text{ s} < t_f \le t_{\text{current}}\}$$
From this buffer, three temporal velocity features are derived in event-time:
1. `Rolling_Flow_Velocity_10s`: Total forward flows initiated in the last 10 seconds:
   $$V_{\text{flow}} = |\mathcal{W}(t_{\text{current}})|$$
2. `Rolling_Bytes_Velocity_10s`: Total forward payload bytes transmitted in the last 10 seconds:
   $$V_{\text{bytes}} = \sum_{f \in \mathcal{W}} \text{FwdBytes}_f$$
3. `Rolling_Single_Pkt_Ratio_10s`: Proportion of flows in the buffer containing exactly one packet (a key indicator of rapid SYN scanning and UDP sweeps):
   $$R_{\text{single}} = \frac{|\{f \in \mathcal{W} \mid \text{FwdPackets}_f = 1\}|}{\max(1, |\mathcal{W}|)}$$

### 9.2 Event-Time Clock vs. Wall-Clock Processing
To guarantee absolute determinism and prevent replay speed from distorting feature calculations:
- AEGIS operates strictly in **Event Time** (`clock_mode="event_time"`).
- Time advances according to the flow's recorded header timestamp ($t_{\text{header}}$), not the computer's CPU clock.
- When an old flow passes the 10-second boundary ($t_{\text{current}} - t_{\text{buffered}} > 10.0\text{ s}$), it is evicted via `deque.popleft()`.

### 9.3 Buffer Overflow Protection & Degraded Operational State
To prevent unbounded memory growth during high-throughput network bursts:
1. The causal buffer is constrained to a fixed maximum capacity: `max_capacity = 10,000` flows.
2. If the buffer reaches capacity, incoming flows are evaluated with fallback point-in-time statistics.
3. The engine logs a transition to `DEGRADED` state and asserts backpressure rather than crashing the process.

### 9.4 Fault Isolation & State Checkpointing
- **`FlowQuarantineManager`:** Isolates malformed records (e.g., non-parseable timestamps, invalid protocols) into a structured quarantine ledger without halting the streaming loop.
- **State Checkpointing (`src/checkpoint.py`):** Periodically serializes engine state (processed flow count, cumulative anomaly counts, causal buffer contents) to disk (`checkpoint.json`). On unexpected process restart, the engine restores its historical window seamlessly.

---

## 10. Explainable AI (XAI) & Threat Attribution

In an operational SOC environment, binary anomaly alerts without attribution create operator fatigue and cannot guide incident triage. AEGIS implements real-time feature attribution in `src/explainability.py`.

### 10.1 Mathematical Residual Decomposition
Because the autoencoder reconstruction error is a weighted sum over independent feature dimensions, the relative contribution of feature $j$ to the anomaly score is computed algebraically:
$$\text{Attribution}_j = \frac{w_j (z_j - \hat{z}_j)^2}{\sum_{k=1}^D w_k (z_k - \hat{z}_k)^2} \times 100\%$$
For any anomalous flow ($S(\mathbf{z}) > \tau$), the engine ranks all features by attribution percentage and isolates the **Top-5 Leading Drivers**.

```
+----------------------------------------------------------------------------------------------------+
|                                    XAI WATERFALL ATTRIBUTION EXAMPLE                               |
| Flow #FL-001092 | Ground Truth: PortScan | Score: 0.2894 (Threshold: 0.190885) -> ANOMALY VERDICT   |
+----------------------------------------------------------------------------------------------------+
| Feature Name                        Attribution %    Visual Decomposition Bar                      |
| ----------------------------------  -------------    --------------------------------------------- |
| Rolling_Single_Pkt_Ratio_10s               42.8%    [████████████████████                    ]   |
| Rolling_Flow_Velocity_10s                  28.4%    [█████████████                           ]   |
| min_seg_size_forward                       12.1%    [██████                                  ]   |
| Fwd IAT Min                                 9.5%    [████                                    ]   |
| Fwd Packet Length Max                       4.2%    [██                                      ]   |
+----------------------------------------------------------------------------------------------------+
```

### 10.2 Automated SOC Narrative Synthesis
The attribution vector is mapped through heuristic forensic attribution rules to generate plain-text incident narratives:
- If `Rolling_Single_Pkt_Ratio_10s` and `Rolling_Flow_Velocity_10s` dominate:
  *"High-velocity single-packet flow burst detected. Traffic signature exhibits structural characteristics of rapid reconnaissance scanning or horizontal port enumeration."*
- If `Rolling_Bytes_Velocity_10s` and `Total Length of Fwd Packets` dominate:
  *"Volumetric bandwidth surge detected. Payload byte volumes significantly exceed normal baseline thresholds, consistent with data exfiltration or DoS flood activity."*

If an optional `GEMINI_API_KEY` is present in the environment, AEGIS can asynchronously dispatch the structured attribution tuple to the Gemini 1.5 Flash API to synthesize executive-level incident briefs with remediation guidance.

---

## 11. Operational SOC Dashboard Architecture

The AEGIS presentation layer is hosted by a lightweight, zero-dependency Python HTTP server (`dashboard/server.py`) serving an asynchronous, single-page application (`dashboard/index.html`):

```
+----------------------------------------------------------------------------------------------------+
|                                      AEGIS SOC DASHBOARD (PORT 8501)                               |
+----------------------------------------------------------------------------------------------------+
|  [View 1: Overview]  [View 2: Threat Radar]  [View 3: Live Feed]  [View 4: XAI]  [View 5: Sandbox] |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  VIEW 1: EXECUTIVE OVERVIEW                                                                        |
|  - Real-time KPIs: Ingestion Rate, Cumulative Anomalies, Alert Rate %, Sensor Health               |
|  - 24-Hour Attack Category Radar Chart (13 Attack Families vs. Detection Rates)                    |
|  - Threat Severity Donut (Critical, High, Medium, Low)                                             |
|  - Recent Critical Incident Stream Table                                                           |
|                                                                                                    |
|  VIEW 2: THREAT RADAR & SPATIAL DECOMPOSITION                                                      |
|  - High-dimensional spatial topology scatter plot                                                  |
|  - Attack family detection rate breakdowns (Heartbleed 54.5%, PortScan 99.8%, Hulk 99.8%)          |
|                                                                                                    |
|  VIEW 3: LIVE OPTICAL DIODE THREAT FEED STREAM                                                     |
|  - Real-time streaming tabular monitor updating at up to 100 flows/sec                             |
|  - Streaming pause/resume toggle, playback speed slider, hardware diode link health indicator       |
|                                                                                                    |
|  VIEW 4: FORENSIC EXPLAINABLE AI (XAI)                                                             |
|  - Flow inspection drawer with Ground Truth vs. Autoencoder Verdict comparison                     |
|  - Top-5 Leading Feature Attribution Waterfall Chart                                               |
|  - Automated SOC Analyst Narrative Synthesis                                                       |
|                                                                                                    |
|  VIEW 5: MODEL CALIBRATION SANDBOX                                                                 |
|  - Empirical threshold calibration sandbox across 5 model architectures                            |
|  - Verified Discrete Operating Point Selector (Primary alpha = 0.2% vs. Reference alpha = 1.0%)     |
|  - Dynamic Model Evaluation Card (Card A) with strict field-level metric validation                |
|  - Head-to-Head (H2H) Benchmark Comparison Table                                                   |
+----------------------------------------------------------------------------------------------------+
```

### 11.1 API Design & Endpoint Contracts
The dashboard server provides ten REST endpoints:
1. `GET /api/overview`: Returns executive KPIs, category radar metrics, and sensor status.
2. `GET /api/threat_analytics`: Returns category-level anomaly rates and temporal distributions.
3. `GET /api/live_feed`: Returns the latest batch of processed flows and anomaly flags from `live_stream_feed.json`.
4. `GET /api/explain?flow_id={id}`: Returns reconstruction error attribution and SOC narrative for a specific flow.
5. `GET /api/system_status`: Returns optical tap link health, consecutive failures, and champion metadata.
6. `GET /api/metrics`: Returns comprehensive evaluation benchmarks for all cataloged models.
7. `GET /api/models`: Returns model catalog metadata, verifying production champion and candidate parameters.
8. `GET /api/compare_models`: Returns head-to-head comparison table across v1 and v2 operating points.
9. `GET /api/start_stream?events={N}&eps={R}`: Launches streaming replay subprocess.
10. `GET /api/stop_stream`: Gracefully halts streaming replay subprocess.

---

## 12. Experimental Results & Performance Evaluation

To prevent misleading claims, evaluation results must be strictly segmented into:
1. **Full-Dataset Benchmark Evaluation** (2,300,825 held-out flows on `test_tue_fri.parquet`).
2. **Streaming Demonstration Replay Scenarios** (2,000-flow live stream playback).

### 12.1 Full-Dataset Benchmark Evaluation (2.3M Flows)

```
+----------------------------------------------------------------------------------------------------+
|                            FULL-DATASET OPERATIONAL BENCHMARK (2,300,825 FLOWS)                    |
+--------------------------+------------------------------+------------------------------------------+
| Metric                   | Champion v1 (Production)     | Candidate v2 Primary (Shadow Research)   |
+--------------------------+------------------------------+------------------------------------------+
| Feature Space            | 32 Features (Forward+Leak)   | 26 Features (Strict Pure Forward)        |
| Threshold (tau)          | 0.190885 (alpha = 0.2%)      | 0.314036 (alpha = 0.2%)                  |
| True Positives (TP)      | 519,031                      | 476,125                                  |
| False Positives (FP)     | 217,598                      | 186,489                                  |
| True Negatives (TN)      | 1,525,581                    | 1,556,690                                |
| False Negatives (FN)     | 38,615                       | 81,521                                   |
| Overall Recall           | 93.08%                       | 85.38%                                   |
| Operational Test FPR     | 12.48%                       | 10.70%                                   |
| Precision                | 70.46%                       | 71.86%                                   |
| F1-Score                 | 0.8020                       | 0.7804                                   |
| PR-AUC (Avg Precision)   | 0.8783                       | 0.8687                                   |
| ROC-AUC                  | 0.9538                       | 0.9478                                   |
| Operational Status       | Production Alert Authority   | Experimental Shadow Only                 |
+--------------------------+------------------------------+------------------------------------------+
```

#### Category-Level Detection Breakdown (Champion v1)
Across the 14 attack families in the operational test set, Champion v1 demonstrates distinct detection profiles:
- **High-Velocity & Reconnaissance Attacks:**
  - `PortScan`: **99.79%** (158,601 / 158,930 detected)
  - `DoS Hulk`: **99.85%** (230,726 / 231,073 detected)
  - `DDoS`: **97.83%** (125,253 / 128,027 detected)
- **Low-and-Slow Application Attacks:**
  - `Infiltration`: **58.33%** (21 / 36 detected)
  - `Heartbleed`: **54.55%** (6 / 11 detected)
  - `DoS Slowhttptest`: **25.02%** (1,376 / 5,499 detected)
  - `DoS slowloris`: **22.07%** (1,279 / 5,796 detected)
  - `DoS GoldenEye`: **15.00%** (1,544 / 10,293 detected)
- **Stealthy Evasion Attacks (Challenging for Forward-Only Telemetry):**
  - `FTP-Patator`: **2.27%** (180 / 7,938 detected)
  - `Bot (ARES)`: **2.29%** (45 / 1,966 detected)
  - `SSH-Patator`: **0.00%** (0 / 5,897 detected)
  - `Web Attacks`: **0.00%** (0 / 2,180 detected)

*Key Finding:* Brute-force credentials and web injection attacks produce minimal forward volume anomalies during authentication exchanges; their primary anomalous signals reside in server error response codes (HTTP 401/403/500) and reverse packet timing—telemetry that physical data diodes strictly cannot observe.

---

### 12.2 Live Demonstration Replay Scenarios (2,000 Flows)

To facilitate demonstration within university evaluation sessions, AEGIS provides two replay configurations:

#### Scenario 1: Default Two-Episode Replay (`--replay-scenario two_episode`)
A composite stream concatenating two non-contiguous, highly authentic chronological episodes:
1. **Episode 1: PortScan** (`[1,685,068 : 1,686,068]`, 1,000 flows @ `02:50:00 - 02:51:00`):
   - First 100 flows: 100% pure genuine `BENIGN` lead-in.
   - Attack onset: Flow index 100 (`PortScan`).
   - Episode Composition: 897 PortScan, 103 Benign.
   - **v1 Performance:** $\text{TP} = 893$, $\text{FP} = 0$, $\text{FN} = 4$, $\text{TN} = 103$.
   - **Recall: 99.55% | FPR: 0.00% | Precision: 100.00% | F1: 99.78%**.
2. **Boundary Transition:**
   - Temporal gap between Episode 1 end (`02:51:00`) and Episode 2 start (`03:56:00`) is $\Delta t = 3,900.0\text{ seconds}$ (1 hour, 5 minutes).
   - Because $3,900.0\text{ s} \gg 10.0\text{ s}$, the causal lookback buffer naturally evicts all historical flows, starting Episode 2 with a clean context.
3. **Episode 2: DDoS Onset** (`[1,903,272 : 1,904,272]`, 1,000 flows @ `03:56:00`):
   - First 100 flows: 100% pure genuine `BENIGN` lead-in.
   - Attack onset: Flow index 100 (`DDoS`).
   - Episode Composition: 460 DDoS, 540 Benign.
   - **v1 Performance:** $\text{TP} = 0$, $\text{FP} = 79$, $\text{FN} = 460$, $\text{TN} = 461$.
   - **Recall: 0.00% | FPR: 14.63% | F1: 0.00%**.
   - *Technical Cause of 0% Recall in Episode 2:* In the complete dataset, DDoS floods comprise 128,027 consecutive flows that saturate the 10-second window to over 5,000 flows/sec. At the initial onset boundary (first 1,000 flows), rolling velocity has not yet reached the critical flood threshold, causing reconstruction residuals to remain below the static cut-off.
4. **Combined Two-Episode Summary (2,000 Flows):**
   - Total Flows: 2,000 (1,357 Attacks, 643 Benign).
   - Overall Confusion Matrix: $\text{TP} = 893$, $\text{FP} = 79$, $\text{FN} = 464$, $\text{TN} = 564$.
   - **Overall Recall: 65.81% | FPR: 12.29% | Precision: 91.87% | F1: 76.69%**.

#### Scenario 2: Legacy DDoS Replay (`--replay-scenario legacy_ddos`)
The original baseline replay slice selecting 2,000 contiguous flows deep within the Friday DDoS flood:
- Dataset Slice: `[1,942,302 : 1,944,302]` (Friday, `04:00:00`).
- Composition: 1,942 DDoS flows, 58 Benign flows.
- **v1 Performance:** $\text{TP} = 1,616$, $\text{FP} = 5$, $\text{FN} = 326$, $\text{TN} = 53$.
- **Recall: 83.21% | FPR: 8.62% | Precision: 99.69% | F1: 90.71%**.

---

## 13. System Limitations

A rigorous engineering evaluation must articulate explicit boundary conditions and limitations:

1. **Software Emulation vs. Physical Diode Enforcement:**
   AEGIS demonstrates threat detection on forward-only feature extracts from PCAP/flow data. It is a software prototype and does not include physical Layer-1 hardware transceiver hardware or FPGA-based diode enforcement.
2. **Reverse Telemetry Blindness:**
   Attacks characterized primarily by return signaling (e.g., failed authentication responses in SSH/FTP brute force, HTTP error codes in SQL injection) cannot be reliably detected from forward flow headers alone without deep packet payload inspection.
3. **Distribution Drift & Calibration Inflation:**
   Calibrating thresholds on Monday benign data at $\alpha = 0.2\%$ resulted in an operational test FPR of $12.48\%$ on Tuesday–Friday. This discrepancy highlights temporal distribution drift in benign background traffic over multi-day operational windows.
4. **Velocity Feature Warm-Up & Slice Starvation:**
   Causal rolling velocity features require a populated lookback window. Isolated 1,000-flow attack slices can starve velocity features, causing low-volume attack onsets to pass sub-threshold.
5. **CIC-IDS2017 Synthetic Artifacts:**
   The underlying benchmark contains synthetic simulation artifacts (e.g., artificial packet generation rates, missing payload entropy distributions) that may not fully reflect operational SCADA/ICS telemetry.

---

## 14. Future Scope & Roadmap

1. **Hardware FPGA / SmartNIC Offload:** Implementing the feature extraction engine and quantized autoencoder inference on an FPGA/SmartNIC (e.g., AMD Xilinx Alveo) to achieve wire-speed Layer-1 processing at 10 Gbps+.
2. **Adaptive Dynamic Thresholding:** Replacing static empirical quantile cutoffs with online conformal prediction or streaming extreme value theory (EVT) to mitigate multi-day benign drift.
3. **Deep Packet Inspection (DPI) for One-Way Payloads:** Integrating forward payload byte entropy and n-gram embedding models (e.g., Char-CNN or lightweight Transformer) to detect stealthy SQL injection and buffer overflows in forward packet payloads.
4. **Theoretical Hyperparameter Exploration:** Exploring evolutionary or Bayesian meta-optimization in offline research to explore feature subsets without compromising production determinism.

---

## 15. Conclusion

The AEGIS project demonstrates that unsupervised, label-free anomaly detection is technically viable for strictly one-way network traffic. By coupling deep undercomplete autoencoders with causal event-time rolling temporal velocity context, AEGIS successfully isolates malicious flow patterns—achieving **93.08% overall attack recall** on the full 2.3-million-flow CIC-IDS2017 benchmark—without requiring attack signatures or bidirectional TCP telemetry. The system incorporates memory-bounded streaming safety, transparent XAI feature attribution, and a comprehensive SOC management dashboard, providing an architecturally sound foundation for critical infrastructure defense.

---

## 16. References & Standards

1. **Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).** *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), 108–116.
2. **NIST Special Publication 800-82, Rev. 2 (2015).** *Guide to Industrial Control Systems (ICS) Security.* National Institute of Standards and Technology.
3. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press (Chapter 14: Autoencoders).
4. **Chandola, V., Banerjee, A., & Kumar, V. (2009).** *Anomaly detection: A survey.* ACM Computing Surveys (CSUR), 41(3), 1–58.
5. **Smart India Hackathon (SIH 2026).** *Problem Statement SIH26145: Label-Free Threat Detection in One-Way Network Traffic.*
