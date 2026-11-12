# AEGIS: Comprehensive Viva Voce & Oral Examination Guide
## 65 Master Technical Questions & Detailed Answers for B.Tech AI/ML Examination

---

### Navigation Index
1. [Project Fundamentals (Q1–Q5)](#1-project-fundamentals)
2. [Computer Networking Principles (Q6–Q10)](#2-computer-networking-principles)
3. [Network Flows & One-Way Simplex Traffic (Q11–Q15)](#3-network-flows--one-way-simplex-traffic)
4. [Dataset & Chronological Preprocessing (Q16–Q20)](#4-dataset--chronological-preprocessing)
5. [Unsupervised Anomaly Detection Theory (Q21–Q25)](#5-unsupervised-anomaly-detection-theory)
6. [Autoencoder Architecture & Mathematics (Q26–Q30)](#6-autoencoder-architecture--mathematics)
7. [Genetic Algorithm & Optimization (Q31–Q35)](#7-genetic-algorithm--optimization)
8. [Feature Engineering & Information Leakage (Q36–Q40)](#8-feature-engineering--information-leakage)
9. [Streaming Engine, Causal Windows & Memory Safety (Q41–Q45)](#9-streaming-engine-causal-windows--memory-safety)
10. [Empirical Threshold Calibration (Q46–Q50)](#10-empirical-threshold-calibration)
11. [Evaluation Metrics & Confusion Matrices (Q51–Q55)](#11-evaluation-metrics--confusion-matrices)
12. [System Architecture, APIs & Reliability (Q56–Q60)](#12-system-architecture-apis--reliability)
13. [Explainable AI (XAI) & SOC Operations (Q61–Q63)](#13-explainable-ai-xai--soc-operations)
14. [Limitations, Tricky Examiner Traps & Future Scope (Q64–Q65)](#14-limitations-tricky-examiner-traps--future-scope)

---

### 1. Project Fundamentals

#### Q1: What is the core objective of project AEGIS?
**Answer:** AEGIS is an unsupervised threat detection framework engineered specifically for simplex (one-way) network conduits, such as industrial data diodes protecting nuclear or electrical power grids. In such networks, return packets (TCP ACKs, server responses) are physically absent, rendering standard signature-based and bidirectional machine learning NIDS inoperable. AEGIS detects attacks without attack signatures or ground-truth labels using undercomplete deep autoencoders and causal 10-second temporal velocity features.

#### Q2: What problem statement ID does AEGIS target?
**Answer:** It addresses Problem Statement **SIH26145** from the Smart India Hackathon / National Cybersecurity Research initiative: *"Label-Free Threat Detection in One-Way Network Traffic"*.

#### Q3: Why is the problem referred to as "label-free"?
**Answer:** "Label-free" means the detection engine requires zero malicious ground-truth attack labels during training or inference. The model trains exclusively on normal, benign background network traffic to learn baseline behavior. Any flow that deviates significantly from this learned normal manifold is flagged as an anomaly. This enables detection of novel zero-day exploits that lack known signatures.

#### Q4: Does AEGIS connect to a physical hardware data diode?
**Answer:** No. AEGIS is a software-based research prototype that simulates streaming one-way traffic by playing back chronologically ordered flow records extracted from the benchmark CIC-IDS2017 dataset. We do not claim physical Layer-1 hardware transceiver enforcement.

#### Q5: What are the primary deliverables of AEGIS?
**Answer:**
1. A feature engineering pipeline isolating forward-only telemetry.
2. A dual-model autoencoder architecture (Champion v1 and Candidate v2).
3. A fault-tolerant streaming engine with a 10-second causal rolling window.
4. An Explainable AI (XAI) attribution engine decomposing reconstruction errors.
5. A five-view real-time SOC web dashboard hosted over REST APIs on port 8501.

---

### 2. Computer Networking Principles

#### Q6: What is the fundamental difference between a packet and a flow?
**Answer:**
- A **packet** is a single protocol data unit (PDU) at Layer 3/4 containing raw headers and payload bytes.
- A **flow** is an aggregated statistical abstraction representing a sequence of packets sharing a 5-tuple: (Source IP, Destination IP, Source Port, Destination Port, Transport Protocol) within a specific temporal inactivity timeout.

#### Q7: What is the TCP three-way handshake, and why does it fail on a data diode?
**Answer:** In standard TCP, establishing a connection requires:
1. Client sends `SYN`.
2. Server responds with `SYN-ACK`.
3. Client acknowledges with `ACK`.
Across a physical data diode, the return path is severed. The listening server can never transmit the `SYN-ACK` back to the sender. Therefore, connections across data diodes typically use raw UDP streaming, modified simplex TCP proxies, or forward optical taps monitoring outbound traffic.

#### Q8: What is the difference between TCP and UDP in unidirectional traffic?
**Answer:** TCP is inherently connection-oriented and requires bidirectional acknowledgment for flow control and retransmission. UDP is connectionless and best-effort; it transmits datagrams without expecting return signaling, making UDP the natural protocol for simplex data diode transmission.

#### Q9: What are TCP control flags, and which ones are visible on a one-way forward tap?
**Answer:** TCP flags include `SYN`, `ACK`, `FIN`, `RST`, `PSH`, and `URG`. On a strict forward-only optical tap, only flags set by the transmitting source (`SYN`, `Fwd PSH`, `Fwd URG`, and initial `FIN`) are physically observable. Return flags (`ACK`, return `FIN/RST`) are generated by the destination and cannot be observed across a true data diode.

#### Q10: How does Port Scanning work at the network level?
**Answer:** Port scanning involves transmitting probe packets (`SYN`, `ACK`, or `NULL`) across a range of destination port numbers on a target host to identify listening network services. On a one-way monitor, a port scan is characterized by a rapid succession of single-packet flows across varying destination ports with short inter-arrival times.

---

### 3. Network Flows & One-Way Simplex Traffic

#### Q11: What is a data diode, and where is it deployed?
**Answer:** A data diode is a physical hardware cybersecurity device enforcing unidirectional Layer-1 data transmission using an optical transmitter (LED/laser) connected to a photodiode receiver with no return fiber. It is deployed at the boundary between high-security control zones (e.g., nuclear reactor cores, SCADA control networks) and lower-security enterprise IT networks to prevent data exfiltration.

#### Q12: Why do conventional NIDS tools (like Snort, Suricata, or Zeek) fail on one-way traffic?
**Answer:**
1. They maintain stateful bidirectional connection tables to track TCP sequence numbers, window states, and handshake completion.
2. They inspect application-layer request-response pairs (e.g., matching an HTTP GET to its 200 OK response).
3. On a one-way link, half of the conversation is missing; standard NIDS classify incomplete handshakes as invalid traffic or flood operators with false connection timeout errors.

#### Q13: What flow statistics become unavailable or corrupted when traffic is unidirectional?
**Answer:**
- Backward packet count and backward byte volume.
- Flow duration (if calculated using return packet timestamps).
- Round-Trip Time (RTT) and TCP handshake latency.
- Client-to-server and server-to-client packet ratio (`Down/Up Ratio`).
- Destination TCP window size advertisements.

#### Q14: How does AEGIS extract flow telemetry without a physical return path?
**Answer:** AEGIS restricts flow feature extraction strictly to forward-direction packet headers: forward packet counts, forward byte lengths, forward inter-arrival times (IAT), forward TCP flags, and streaming temporal velocity computed over recent forward flows.

#### Q15: What is an Optical Tap?
**Answer:** An optical network tap is a passive hardware device spliced into a fiber-optic cable. Using an internal beam-splitter prism, it splits optical light into two beams (e.g., 70% continuing to destination, 30% diverted to a monitoring interface). It operates strictly passively without injecting packets into the network.

---

### 4. Dataset & Chronological Preprocessing

#### Q16: What dataset is used in AEGIS, and what are its key statistics?
**Answer:** AEGIS uses the **CIC-IDS2017** benchmark dataset. The complete dataset contains five days of network traffic with over 2.8 million flows. Our processed operational split consists of 363,597 training flows, 166,321 calibration flows, and 2,300,825 test flows across 14 attack families.

#### Q17: What is the difference between a random split and a chronological split?
**Answer:**
- A **random split** shuffles flows uniformly across the entire timeline. This introduces severe **temporal data leakage**: the model memorizes IP subnets, temporal patterns, and attack artifacts from the test set during training.
- A **chronological split** strictly respects time: training data occurs prior to calibration data, which occurs prior to operational test data. AEGIS trains on Monday (`01:00` to `10:44`), calibrates on Monday (`10:44` to `12:59`), and evaluates on Tuesday through Friday.

#### Q18: What is temporal distribution drift, and how does AEGIS reflect it?
**Answer:** In real networks, benign user behavior shifts over days (e.g., Tuesday workload differs from Friday file backups). Because AEGIS tests on subsequent days (Tue–Fri) without retraining, it captures authentic distribution drift: our operational test FPR is 12.48% despite being calibrated at 0.2% on Monday benign data.

#### Q19: Why must ground-truth labels NEVER be fed into the model?
**Answer:** In operational deployment, incoming network packets do not carry labels indicating whether they are malicious. Using labels during inference is an invalid data leakage fallacy. AEGIS uses labels strictly for offline benchmarking and post-detection evaluation.

#### Q20: What data hygiene steps are executed before feeding flows into the model?
**Answer:**
1. Duplicate column removal (`Fwd Header Length.1`).
2. Label encoding and mojibake normalization (fixing UTF-8 em-dash issues).
3. Clamping negative sentinels (`-1` duration or window size) to 0.
4. Imputing `np.inf` and `np.nan` using training set medians.
5. Log-transformation (`Log1p`) on heavy-tailed volume metrics.

---

### 5. Unsupervised Anomaly Detection Theory

#### Q21: What is the fundamental difference between supervised and unsupervised intrusion detection?
**Answer:**
- **Supervised NIDS** trains on labeled examples of both benign and attack classes (e.g., Random Forest or XGBoost). It detects known attack patterns accurately but fails on novel or zero-day exploits.
- **Unsupervised NIDS** trains exclusively on normal benign data without attack labels. It models the boundary of normal behavior and flags any statistically significant deviation as an anomaly.

#### Q22: What is the core assumption of unsupervised anomaly detection?
**Answer:** The core assumption is that **anomalies are statistically rare and topologically distinct** from the normal baseline distribution in feature space.

#### Q23: Why do we train our autoencoder exclusively on benign traffic?
**Answer:** By exposing the autoencoder only to benign traffic, the neural network learns to compress and reconstruct normal protocol correlations with low error. Because it has never learned the manifold of attack flows (e.g., port scan bursts or volumetric floods), it fails to reconstruct them, producing large reconstruction errors that trigger alerts.

#### Q24: What is the "zero-day attack advantage" of unsupervised autoencoders?
**Answer:** An attacker cannot evade detection simply by mutating their signature or changing byte strings, because the detector is not matching signatures. As long as the attack behaviorally violates normal feature relationships (e.g., abnormal packet rates or invalid flag ratios), the autoencoder flags it.

#### Q25: What is the primary operational drawback of unsupervised anomaly detection?
**Answer:** **False Positive Rate (FPR)**. Legitimate but rare benign activity (e.g., an unusual network backup or system administrator maintenance task) may also produce high reconstruction errors and trigger false alarms.

---

### 6. Autoencoder Architecture & Mathematics

#### Q26: What are the three core components of an autoencoder?
**Answer:**
1. **Encoder ($\mathcal{E}$):** Compresses high-dimensional input $\mathbf{z} \in \mathbb{R}^D$ into a low-dimensional bottleneck representation $\mathbf{h} \in \mathbb{R}^k$.
2. **Latent Bottleneck ($\mathbf{h}$):** The compact informational constraint ($k \ll D$) that forces the model to discard noise and retain only salient manifold correlations.
3. **Decoder ($\mathcal{D}$):** Reconstructs the compressed representation back into input space: $\hat{\mathbf{z}} \in \mathbb{R}^D$.

#### Q27: What is the mathematical formulation of reconstruction error?
**Answer:** For unweighted Mean Squared Error (MSE):
$$\mathcal{L}_{\text{MSE}}(\mathbf{z}, \hat{\mathbf{z}}) = \frac{1}{D} \sum_{j=1}^D (z_j - \hat{z}_j)^2$$
For Weighted Mean Squared Error (WMSE, used in Champion v1):
$$\mathcal{L}_{\text{WMSE}}(\mathbf{z}, \hat{\mathbf{z}}) = \sum_{j=1}^D w_j (z_j - \hat{z}_j)^2$$

#### Q28: What is the exact layer topology of Champion v1?
**Answer:** Implemented using `sklearn.neural_network.MLPRegressor`:
- Input Layer: 32 dimensions (29 forward flow + 3 temporal velocity).
- Hidden Layer 1: 24 neurons (ReLU).
- Latent Bottleneck: 8 neurons.
- Hidden Layer 2: 24 neurons (ReLU).
- Output Layer: 32 dimensions (Linear reconstruction).
- Weights serialized at `models/temporal_autoencoder_model.joblib`.

#### Q29: What is the exact layer topology of Candidate v2?
**Answer:** Implemented using `sklearn.neural_network.MLPRegressor`:
- Input Layer: 26 dimensions (strict pure-forward features).
- Hidden Layer 1: 20 neurons (ReLU).
- Latent Bottleneck: 8 neurons.
- Hidden Layer 2: 20 neurons (ReLU).
- Output Layer: 26 dimensions (Linear reconstruction).
- Weights serialized at `models/pure_one_way_v2/pure_oneway_autoencoder.joblib`.

#### Q30: Why is an undercomplete autoencoder preferred over a PCA detector?
**Answer:** Principal Component Analysis (PCA) is strictly a linear orthogonal projection. An autoencoder incorporates non-linear activation functions (ReLU), allowing it to model non-linear manifold geometries, complex timing correlations, and multi-modal distribution surfaces that linear PCA cannot capture.

---

### 7. Optimization & Genetic Algorithms

#### Q31: Are Genetic Algorithms implemented in the AEGIS codebase?
**Answer:** **No.** Genetic Algorithms (GA) are **not implemented** in the AEGIS codebase. There are no evolutionary search routines, multi-objective Pareto optimization loops, chromosome encodings, crossover, or mutation operators in the repository.

#### Q32: Why does AEGIS avoid Genetic Algorithms in its production implementation?
**Answer:** Genetic Algorithms are non-deterministic, stochastic metaheuristics with high computational overhead. In critical infrastructure environments (like nuclear and power grid data diodes) and academic grading, **deterministic mathematical reproducibility, predictable latency, and auditable threshold derivation** are required. Running a randomized evolutionary loop would produce fluctuating thresholds across runs.

#### Q33: How does AEGIS optimize its feature weights without a Genetic Algorithm?
**Answer:** In Champion v1, feature weights $w_j$ for Weighted Mean Squared Error (WMSE) are calculated deterministically from calibration inverse-variance residuals on benign traffic:
$$w_j = \frac{1}{\text{var}_j + \epsilon}$$
normalized by mean weight. Features with low natural variance (such as forward packet counts and temporal velocities) are assigned higher penalty weights when anomalous deviations occur.

#### Q34: How does AEGIS calibrate its decision thresholds?
**Answer:** Through exact deterministic empirical quantile estimation on the Monday benign calibration dataset (`calibration_benign.parquet`):
$$\tau = \text{np.quantile}(\mathcal{L}_{\text{val\_benign}}, 1 - \alpha)$$
For Champion v1, target $\alpha = 0.002$ ($0.2\%$) yields $\tau = 0.190885$. For Candidate v2, primary $\alpha = 0.002$ yields $\tau = 0.314036$ and reference $\alpha = 0.01$ yields $\tau = 0.099018$.

#### Q35: (Tricky Examiner Question) If an examiner asks whether GA optimizes the model during the live stream, how do you respond?
**Answer:** **Honest, confident answer:** *"No, sir/ma'am. AEGIS does not use Genetic Algorithms in its codebase. We deliberately use exact empirical quantile calibration on benign validation traffic. This guarantees 100% mathematical reproducibility, zero training variance, and sub-millisecond per-flow scoring latency without stochastic drift."*

---

### 8. Feature Engineering & Information Leakage

#### Q36: What is "reverse-channel data leakage" in flow datasets?
**Answer:** Reverse-channel leakage occurs when features labeled as forward telemetry secretly incorporate packet data from the reverse direction. In CIC-IDS2017, the `CICFlowMeter` tool calculated `Flow Duration` using reverse packet timestamps in 77.4% of flows, added backward bytes into `Flow Bytes/s`, and recorded return `ACK` flags.

#### Q37: How many features does Champion v1 use, and what are they?
**Answer:** Champion v1 uses **32 features**:
- 29 forward-dominated flow features (packet counts, forward header length, forward packet length statistics, forward inter-arrival times, initial window size, TCP flags, duration, forward rates).
- 3 causal rolling velocity features derived from recent history.

#### Q38: How many features does Candidate v2 use, and why was it created?
**Answer:** Candidate v2 uses **26 features**. It was created during our Phase 1 audit to achieve strict Layer-1 physical compliance. It purges all 8 leaked flow-wide features (`Flow Duration`, `Flow Bytes/s`, `Fwd Packets/s`, and flow-wide TCP flags) and replaces them with pure forward rates:
$$\text{Fwd\_Bytes\_Per\_Sec} = \frac{\text{Total Length of Fwd Packets}}{\max(\text{Fwd IAT Total}, 10^{-6})}$$
$$\text{Fwd\_Packets\_Per\_Sec} = \frac{\text{Total Fwd Packets}}{\max(\text{Fwd IAT Total}, 10^{-6})}$$

#### Q39: What are the 3 temporal velocity features, and how are they calculated?
**Answer:**
1. `Rolling_Flow_Velocity_10s`: Number of flows in the causal lookback window $(t - 10\text{ s}, t]$.
2. `Rolling_Bytes_Velocity_10s`: Sum of forward payload bytes across flows in $(t - 10\text{ s}, t]$.
3. `Rolling_Single_Pkt_Ratio_10s`: Ratio of flows in the buffer that contain exactly one packet:
   $$R_{\text{single}} = \frac{|\{f \in \mathcal{W} \mid \text{FwdPackets} = 1\}|}{\max(1, |\mathcal{W}|)}$$

#### Q40: Why is `Rolling_Single_Pkt_Ratio_10s` critical for detecting PortScans?
**Answer:** Normal web traffic contains multi-packet request-response flows. Port scans transmit thousands of isolated probe packets (e.g., SYN packets to unmapped ports). When a scan occurs, `Rolling_Single_Pkt_Ratio_10s` surges toward $1.0$, producing an immediate reconstruction error spike in the autoencoder.

---

### 9. Streaming Engine, Causal Windows & Memory Safety

#### Q41: What is a causal sliding window, and why must it be causal?
**Answer:** A causal window only observes historical events occurring at or before the current flow's timestamp ($t_k \le t_{\text{current}}$). If the window peeked ahead to flows where $t_k > t_{\text{current}}$, it would constitute future data leakage, which is physically impossible in live real-time streams.

#### Q42: What is the difference between Event Time and Wall-Clock Time?
**Answer:**
- **Wall-clock time** is the local CPU time of the running machine.
- **Event time** is the timestamp stamped in the packet/flow header.
AEGIS operates strictly in event time so that historical network traffic replayed at 15 flows/sec or 1,000 flows/sec yields identical, deterministic feature values.

#### Q43: How does buffer eviction work at the 10-second boundary?
**Answer:** When an incoming flow arrives with timestamp $t_{\text{current}}$, the engine inspects the oldest flow in the deque. If $t_{\text{current}} - t_{\text{oldest}} > 10.0\text{ seconds}$, the oldest flow is removed via `deque.popleft()`. This continues until all remaining buffered flows fall within the 10-second lookback window.

#### Q44: How does AEGIS prevent memory exhaustion during volumetric DDoS floods?
**Answer:** The buffer is bounded to `max_capacity = 10,000` flows. If traffic exceeds this capacity, the engine enters a `DEGRADED` state: it drops expired entries aggressively, computes velocity from bounded statistics, and asserts backpressure without crashing the operating system.

#### Q45: What does `FlowQuarantineManager` do?
**Answer:** It inspects incoming records for malformed data (unparseable timestamp formats, negative values, missing protocol fields). Corrupt flows are diverted to a quarantine memory structure and logged without interrupting streaming inference.

---

### 10. Empirical Threshold Calibration

#### Q46: How is the decision threshold $\tau$ determined in AEGIS?
**Answer:** Through non-parametric quantile estimation on the held-out benign calibration dataset (`calibration_benign.parquet`, 166,321 flows). For a target calibration FPR $\alpha = 0.002$ ($0.2\%$), the threshold $\tau$ is set to the 99.8th percentile of reconstruction scores.

#### Q47: What is the exact value of the Champion v1 threshold?
**Answer:** $\tau = \mathbf{0.190885}$.

#### Q48: What are the dual operating thresholds of Candidate v2?
**Answer:**
1. **Primary Operating Point:** $\tau = \mathbf{0.314036}$ (calibrated at $\alpha = 0.2\%$, optimized for low test FPR: $10.70\%$).
2. **High-Recall Reference Point:** $\tau = \mathbf{0.099018}$ (calibrated at $\alpha = 1.0\%$, achieving recall parity with v1: $93.01\%$).

#### Q49: Why is Calibration Target FPR ($\alpha$) different from Operational Test FPR?
**Answer:**
- $\alpha = 0.2\%$ is the target calibration false alarm rate established strictly on **Monday benign calibration traffic**.
- **Operational Test FPR ($12.48\%$)** is the false alarm rate observed when applying that static threshold to **Tuesday through Friday test traffic**.
The increase from $0.2\%$ to $12.48\%$ is caused by authentic temporal distribution drift across different days of the week.

#### Q50: Can an examiner adjust the threshold dynamically in the dashboard?
**Answer:** In View 5 (Model Calibration Sandbox), the examiner can switch between discrete verified operating points (Primary vs. Reference for v2). Continuous sliders were intentionally removed to prevent operators from arbitrarily inventing non-validated operational thresholds.

---

### 11. Evaluation Metrics & Confusion Matrices

#### Q51: Define True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) in AEGIS.
**Answer:**
- **TP:** An attack flow correctly flagged as an anomaly ($S(\mathbf{z}) > \tau$).
- **FP:** A benign flow incorrectly flagged as an anomaly ($S(\mathbf{z}) > \tau$).
- **TN:** A benign flow correctly classified as normal ($S(\mathbf{z}) \le \tau$).
- **FN:** An attack flow that failed to trigger an alert ($S(\mathbf{z}) \le \tau$).

#### Q52: State the full-dataset evaluation results for Champion v1 across 2.3M flows.
**Answer:**
- Total Flows: 2,300,825 (557,646 Attacks, 1,743,179 Benign).
- $\text{TP} = 519,031 \mid \text{FP} = 217,598 \mid \text{TN} = 1,525,581 \mid \text{FN} = 38,615$.
- **Recall:** $93.08\%$.
- **False Positive Rate (FPR):** $12.48\%$.
- **Precision:** $70.46\%$.
- **F1-Score:** $0.8020$.
- **PR-AUC (Average Precision):** $0.8783$.
- **ROC-AUC:** $0.9538$.

#### Q53: State the results of the default Two-Episode demonstration replay (2,000 flows).
**Answer:**
- Total Flows: 2,000 (1,357 Attacks, 643 Benign).
- **Episode 1 (PortScan):** Recall: **$99.55\%$** | FPR: **$0.00\%$** ($\text{TP}=893, \text{FP}=0, \text{FN}=4, \text{TN}=103$).
- **Episode 2 (DDoS Onset):** Recall: **$0.00\%$** | FPR: **$14.63\%$** ($\text{TP}=0, \text{FP}=79, \text{FN}=460, \text{TN}=461$).
- **Combined Replay:** Overall Recall: **$65.81\%$** | Overall FPR: **$12.29\%$** | Precision: **$91.87\%$** | F1: **$76.69\%$** ($\text{TP}=893, \text{FP}=79, \text{FN}=464, \text{TN}=564$).

#### Q54: (Crucial Examiner Question) Why is the recall 0% in the DDoS onset episode of the 2,000-flow replay?
**Answer:** In the complete CIC-IDS2017 dataset, the Friday DDoS attack spans over 128,000 flows that overwhelm the 10-second window to over 5,000 flows/second, driving full-dataset DDoS recall to $97.83\%$. However, in our isolated 1,000-flow replay slice, we observe only the initial onset of the flood. Because velocity features require time to build up in the lookback window, the rolling velocity remains below the threshold required to trigger an alert.

#### Q55: What are the metrics for the Legacy DDoS Replay?
**Answer:** On the contiguous 2,000-flow DDoS burst slice at index 1,942,302:
- Total Flows: 2,000 (1,942 DDoS, 58 Benign).
- $\text{TP} = 1,616 \mid \text{FP} = 5 \mid \text{TN} = 53 \mid \text{FN} = 326$.
- **Recall:** **$83.21\%$** | **FPR:** **$8.62\%$** | **Precision:** **$99.69\%$** | **F1:** **$90.71\%$**.

---

### 12. System Architecture, APIs & Reliability

#### Q56: What web server technology hosts the AEGIS dashboard?
**Answer:** AEGIS uses a custom zero-dependency Python HTTP server (`ThreatDetectionDashboardHandler` extending `http.server.SimpleHTTPRequestHandler`) running in a background thread on port 8501. It requires no heavyweight external web framework like Django or Flask.

#### Q57: Name three core REST endpoints exposed by `dashboard/server.py`.
**Answer:**
1. `GET /api/live_feed`: Returns streaming flow records and anomaly flags from `live_stream_feed.json`.
2. `GET /api/explain?flow_id={id}`: Returns flow reconstruction error attribution and SOC analyst narrative.
3. `GET /api/metrics`: Returns evaluation metrics and benchmark parameters across all models.

#### Q58: How does AEGIS ensure the dashboard stays accessible after stream completion?
**Answer:** `run_demo.py` implements a main-thread keep-alive loop (`while True: time.sleep(1.0)`) that prevents the Python process and daemon HTTP thread from terminating when the 2,000-flow simulation finishes. The server remains alive until the operator explicitly issues a `Ctrl+C` interrupt, which executes `stop_server()`.

#### Q59: What is the telemetry throttling mechanism in `src/telemetry.py`?
**Answer:** Writing to disk for every single flow packet would saturate storage I/O and throttle streaming playback. `TelemetryManager` batches disk writes, updating `live_stream_feed.json` only once every 20 flows or every 0.25 seconds.

#### Q60: How does AEGIS handle state persistence across process restarts?
**Answer:** The module `src/checkpoint.py` serializes processed flow counts, cumulative anomaly counters, and causal context deque timestamps into a JSON checkpoint. The simulator automatically restores from the latest checkpoint on restart unless disabled with `--no-auto-restore`.

---

### 13. Explainable AI (XAI) & SOC Operations

#### Q61: How does AEGIS explain an anomaly without a supervised classifier?
**Answer:** It performs **Reconstruction Residual Decomposition**. Because the autoencoder's anomaly score is an additive sum of squared errors across individual features, the percentage contribution of feature $j$ is calculated as:
$$\text{Attribution}_j = \frac{w_j (z_j - \hat{z}_j)^2}{\sum_k w_k (z_k - \hat{z}_k)^2} \times 100\%$$
The Top-5 features with the highest attribution are displayed in the dashboard's Waterfall chart.

#### Q62: What is the purpose of the Automated SOC Analyst Narrative?
**Answer:** SOC analysts under high cognitive load cannot inspect raw 32-dimensional floating-point vectors during an ongoing cyber attack. The narrative engine translates the leading mathematical feature drivers into plain-English security intelligence (e.g., *"Single-packet flow burst detected; reconnaissance scanning profile"*).

#### Q63: What role does Google Gemini 1.5 play in the XAI module?
**Answer:** If the user configures a `GEMINI_API_KEY`, AEGIS asynchronously forwards the Top-5 feature attributions to Gemini 1.5 Flash to synthesize executive-level incident briefs with mitigation recommendations. If no API key is present, AEGIS seamlessly falls back to local heuristic forensic attribution rules.

---

### 14. Limitations, Tricky Examiner Traps & Future Scope

#### Q64: (Examiner Trap) "Can your system detect SSH or FTP Brute Force attacks across a physical data diode?"
**Answer:** **Honest, technically accurate answer:** *"No, sir/ma'am. In our benchmark evaluation, SSH-Patator recall was 0.0% and FTP-Patator recall was 2.27%. This is an inherent physical limitation: brute force credential attacks transmit standard authentication payloads in the forward direction. Their anomalous behavior is reflected almost entirely in return error codes (e.g., SSH authentication failure banners or FTP 530 Login Incorrect responses)—telemetry that physical data diodes strictly cannot observe without deep packet inspection of forward credential dictionaries."*

#### Q65: What are the three primary directions for future work?
**Answer:**
1. **FPGA / SmartNIC Hardware Offload:** Implementing the autoencoder forward pass on FPGA hardware to achieve wire-speed Layer-1 processing at 10 Gbps+.
2. **Adaptive Conformal Prediction Thresholds:** Mitigating multi-day distribution drift by updating thresholds dynamically using online conformal prediction.
3. **Deep Packet Payload Inspection:** Integrating character-level CNNs or byte-level transformers on forward packet payloads to catch stealthy web injections and buffer overflows.
