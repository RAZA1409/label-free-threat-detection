# AEGIS: Live Demonstration & Examiner Presentation Guide
## Step-by-Step Operator Manual for University Viva & Evaluation

---

### Executive Overview
This guide provides precise, step-by-step procedures for launching, demonstrating, explaining, and troubleshooting **AEGIS (Label-Free Threat Detection in One-Way Network Traffic)** during academic evaluations, viva voce, and technical demonstrations.

---

## 1. System Requirements & Pre-Flight Checklist

Before launching the demonstration before an examiner:
1. **Operating System:** Windows 10/11 (PowerShell or Command Prompt).
2. **Python Environment:** Python 3.11+ within `.venv` (dependencies installed via `requirements.txt`).
3. **Port Availability:** Ensure port `8501` is free. (AEGIS checks port availability automatically).
4. **Data Files Present:**
   - `outputs/experiment/test_tue_fri.parquet` (~269 MB)
   - `models/temporal_autoencoder_model.joblib`
   - `models/pure_one_way_v2/model.joblib`
5. **Display Setup:** Recommended browser: Chrome, Edge, or Firefox at 100% or 90% zoom for optimal dashboard visibility.

---

## 2. Launching AEGIS

### Method 1: The One-Click Batch Launcher (Recommended for Presentation)
Double-click `START_AEGIS.bat` in the repository root directory (or use the Desktop shortcut if configured).

```cmd
C:\D drive\Label-free-threat-detection> START_AEGIS.bat
```

**Terminal Verification Output:**
When launched correctly, the terminal will display:
```text
======================================================================
  AEGIS: Label-Free Threat Detection in One-Way Network Traffic
  Problem ID: SIH26145 | Unsupervised SOC Radar & Forensic XAI
======================================================================

[+] Port 8501 is available. Dashboard will launch at http://localhost:8501.

[*] Starting AEGIS (Dashboard + Optical Tap Stream Replay)...
    Press Ctrl+C anytime in this window to stop.
----------------------------------------------------------------------

[*] Starting AEGIS HTTP Server on http://localhost:8501 ...
[+] Dashboard active at: http://localhost:8501
[+] Server thread running.
[+] Opening browser to http://localhost:8501 ...

================================================================================
  STARTING LIVE OPTICAL TAP TRAFFIC STREAM (2000 FLOWS)
  Watching incoming flows in real-time. Press Ctrl+C anytime to stop.
================================================================================
```

### Method 2: Direct Command-Line Execution
To control streaming parameters precisely via terminal:

```powershell
# Default: 2,000 flows at 15 flows/sec with authentic Two-Episode replay
.\.venv\Scripts\python.exe run_demo.py --events 2000 --eps 15

# High-Speed Replay: 2,000 flows at 50 flows/sec
.\.venv\Scripts\python.exe run_demo.py --events 2000 --eps 50

# Unthrottled Benchmark Execution:
.\.venv\Scripts\python.exe src/simulate_stream.py --events 2000 --eps 0
```

### Method 3: Running the Legacy DDoS Burst Replay
To reproduce the original 2,000-flow contiguous DDoS flood slice ($83.21\%$ recall):

```powershell
.\.venv\Scripts\python.exe src/simulate_stream.py --events 2000 --eps 20 --replay-scenario legacy_ddos
```

---

## 3. How to Confirm the System is Running Properly

1. **Terminal Feed:** You will see continuous API requests logging HTTP 200 responses:
   ```text
   127.0.0.1 - - [12/Nov/2026 19:30:53] "GET /api/live_feed HTTP/1.1" 200 -
   127.0.0.1 - - [12/Nov/2026 19:30:54] "GET /api/system_status HTTP/1.1" 200 -
   ```
2. **Browser Interface:** Opening `http://localhost:8501` displays the dark-mode AEGIS SOC Dashboard.
3. **Live Indicator:** The green pulsating radar indicator in the top header displays **"SYSTEM STATUS: OPERATIONAL"**.
4. **Post-Simulation Keep-Alive:** When the 2,000-flow replay completes, the terminal outputs:
   ```text
   ================================================================================
     [+] STREAM COMPLETE — SOC DASHBOARD REMAINS ACTIVE & FULLY INTERACTIVE
     URL: http://localhost:8501
     All 5 views (Overview, Radar, Threats, XAI, Calibration) are live.
     Press Ctrl+C anytime to stop the dashboard server and exit.
   ================================================================================
   ```
   *Crucial Exam Point:* The dashboard server **remains fully accessible and interactive** after the simulation completes. It will not close port 8501 until you press `Ctrl+C`.

---

## 4. Tour of the 5 Dashboard Views

The navigation sidebar provides instant access to all five SOC operational views:

```
+---------------------------------------------------------------------------------------+
|  AEGIS SOC NAVIGATION BAR                                                            |
|  [1. Executive Overview]  [2. Threat Radar]  [3. Live Optical Tap]                    |
|  [4. Forensic XAI]        [5. Model Calibration Sandbox]                              |
+---------------------------------------------------------------------------------------+
```

### View 1: Executive Overview
- **What it shows:** High-level operational telemetry for SOC supervisors.
- **Key Elements to Point Out to Examiner:**
  - **KPI Cards:** Live Fleet Ingest Rate, Cumulative Flows Processed, Total Anomalies Detected, and Operational Anomaly Rate (%).
  - **Sensor Uptime & Link Health:** Confirms the simplex optical data diode receiver status.
  - **24-Hour Attack Category Radar Chart:** Displays model sensitivity across 13 distinct attack families (e.g., PortScan 99.8%, DoS Hulk 99.8%, Heartbleed 54.5%).
  - **Severity Breakdown Donut:** Classifies flagged threats into Critical, High, Medium, and Low severity based on reconstruction error magnitude.
  - **Recent Incidents Table:** Real-time log of the latest flagged anomalous flows.

### View 2: Threat Radar & Spatial Decomposition
- **What it shows:** Anomaly distribution and feature clustering in dimensional space.
- **Key Elements to Point Out to Examiner:**
  - **Dimensional Projection Scatter:** Visualizes flows in residual space where normal benign flows cluster tightly near the origin, while anomalous attacks project outward.
  - **Category Detection Rate Table:** Direct empirical proof that AEGIS detects diverse attack signatures without labeled supervised training.

### View 3: Live Optical Diode Threat Feed
- **What it shows:** Simulated physical diode tap receiving flows in real time.
- **Key Elements to Point Out to Examiner:**
  - **Streaming Flow Table:** Real-time stream of incoming packets displaying Timestamp, Protocol, Duration, Forward Packets, Payload Bytes, and Reconstruction Score.
  - **Color-Coded Verdict Badges:** Green badge for `NORMAL` flows ($S(\mathbf{z}) \le 0.190885$); Red flashing badge for `ANOMALY` ($S(\mathbf{z}) > 0.190885$).
  - **Interactive Stream Controls:** Click **"Pause Stream"** to freeze the feed for examiner inspection; adjust the speed slider to throttle between 5 and 100 flows/sec.

### View 4: Forensic Explainable AI (XAI)
- **What it shows:** Why a specific flow was flagged as a threat (Feature Attribution Waterfall).
- **Key Elements to Point Out to Examiner:**
  - Click on any anomalous flow row in the Live Feed or select a flow ID from the drawer.
  - **Ground Truth vs. Verdict:** Demonstrates that the autoencoder arrived at the correct verdict **without ever seeing the ground-truth label**.
  - **Reconstruction Residual Waterfall (Top-5 Leading Drivers):** Visualizes the exact mathematical percentage contribution of each feature to the total reconstruction error:
    $$\text{Attribution}_j = \frac{w_j (z_j - \hat{z}_j)^2}{\sum_k w_k (z_k - \hat{z}_k)^2} \times 100\%$$
  - **Automated SOC Analyst Narrative:** Heuristic natural language synthesis translating feature math into actionable security intelligence (e.g., *"High-velocity single-packet flow burst detected..."*).

### View 5: Model Calibration Sandbox
- **What it shows:** Empirical threshold calibration across architectures without retraining.
- **Key Elements to Point Out to Examiner:**
  - **Model Switcher:** Toggle between *Temporal Autoencoder (v1 Champion)*, *Pure One-Way v2 (Experimental Candidate)*, *Flow Autoencoder*, and *One-Way Log-PCA*.
  - **Discrete Operating Point Selector:** For v2, toggle between **Primary ($\alpha = 0.2\%$, $\tau = 0.314036$)** for lowest false alarms, and **Reference ($\alpha = 1.0\%$, $\tau = 0.099018$)** for maximum recall.
  - **Distinction Banner:** Demonstrates the conceptual separation between **Calibration Target FPR ($\alpha$)** on Monday benign calibration data versus **Operational Test FPR** on held-out Tuesday–Friday evaluation traffic.
  - **Head-to-Head (H2H) Benchmark Table:** Direct comparison of feature dimensionality (32 vs. 26), thresholds, Precision, Recall, and ROC-AUC.
  - **Retry Sync Button:** Demonstrates dynamic synchronization with the backend `/api/metrics` endpoint.

---

## 5. Explaining the v1 Champion vs. v2 Shadow Roles

Examiners frequently ask: *"Which model is actually making the decisions?"*

| Concept | Champion v1 (Production) | Candidate v2 (Experimental Shadow) |
| :--- | :--- | :--- |
| **Role** | **Sole Production Alert Authority** | **Silent Shadow-Mode Research Candidate** |
| **Feature Space** | **32 Features** (Forward flow + 3 temporal velocity) | **26 Features** (Strict pure forward-only) |
| **Operating Threshold** | $\tau = \mathbf{0.190885}$ ($\alpha = 0.2\%$) | $\tau = \mathbf{0.314036}$ (Primary) / $\mathbf{0.099018}$ (Ref) |
| **Loss Function** | Weighted Mean Squared Error (WMSE) | Unweighted Mean Squared Error (MSE) |
| **Alert Power** | Triggers live alerts, increments dashboard counters | Evaluated silently in background telemetry only |
| **Academic Purpose** | Delivers proven high operational recall ($93.08\%$) | Proves feasibility of strict physical Layer-1 compliance |

**Scripted Examiner Answer:**
> *"Sir/Ma'am, in our production pipeline, Champion v1 is the sole alert authority. It processes 32 features and uses a calibrated threshold of 0.190885 to generate all live dashboard alerts. Candidate v2 is an experimental prototype operating strictly in shadow mode; it uses 26 pure-forward features to explore absolute Layer-1 data diode purity without altering live operational alerts."*

---

## 6. The 3-Minute Live Demonstration Script (For University Examiners)

Follow this exact chronological script during your presentation:

### Minute 1: The Context & Launch (Overview View)
1. **Action:** Double-click `START_AEGIS.bat`. Show the terminal starting cleanly.
2. **Speech:**
   > *"Good morning, respected examiners. My project is AEGIS: Label-Free Threat Detection in One-Way Network Traffic. Physical data diodes protect critical infrastructure by enforcing simplex transmission at Layer 1. However, conventional intrusion detection systems break because there is no return traffic, no TCP handshakes, and no reverse RTT telemetry. AEGIS solves this using an unsupervised deep autoencoder operating strictly on forward-only traffic and causal 10-second temporal velocity features."*
3. **Action:** Open `http://localhost:8501`. Point out the Executive Overview KPIs.

### Minute 2: Live Ingestion & Replay (Live Feed & Threat Radar)
1. **Action:** Switch to **View 3 (Live Optical Diode Feed)**.
2. **Speech:**
   > *"We are currently observing simulated optical diode replay of CIC-IDS2017 traffic. Notice the first 100 flows are pure benign baseline traffic—all verdicts are green 'NORMAL'. At flow index 100, the PortScan attack begins. Immediately, our autoencoder flags the incoming flows as red 'ANOMALY'. Notice that we do not use attack signatures or ground-truth labels during inference; detection is driven entirely by reconstruction error exceeding our calibrated cutoff of 0.190885."*
3. **Action:** Click **"Pause Stream"** on an anomalous flow row.

### Minute 3: Explainability & Calibration (Forensic XAI & Sandbox)
1. **Action:** Switch to **View 4 (Forensic XAI)**.
2. **Speech:**
   > *"For any flagged threat, our Forensic XAI engine decomposes the autoencoder's reconstruction error in real-time. Here, in the Attribution Waterfall, we see that 42.8% of the anomaly score was driven by 'Rolling_Single_Pkt_Ratio_10s' and 28.4% by 'Rolling_Flow_Velocity_10s'. This automated analyst narrative correctly identifies a rapid horizontal reconnaissance sweep without manual intervention."*
3. **Action:** Switch to **View 5 (Model Calibration Sandbox)**.
4. **Speech:**
   > *"Finally, in our Model Calibration Sandbox, we benchmark our 32-dimensional production champion against our 26-dimensional pure-forward v2 research candidate. Notice our discrete operating-point selector: we can toggle between an operational primary point optimized for 10.7% test false alarms, and a reference point matching v1's 93% recall. Switching models here is strictly for forensic inspection; live production alert authority remains anchored to v1."*
5. **Action:** Return to terminal, press `Ctrl+C`, and show clean graceful shutdown.

---

## 7. Common Demonstration Errors & Fast Troubleshooting

| Symptom | Probable Cause | Immediate Remediation |
| :--- | :--- | :--- |
| **`ERR_CONNECTION_REFUSED` on port 8501** | The server was closed or previous simulation exited prematurely. | Ensure you are using the updated `run_demo.py` with keep-alive loop. Launch `START_AEGIS.bat` again. |
| **Port 8501 Already in Use** | A previous Python or background process is still bound to port 8501. | Open PowerShell and run: `Get-Process python \| Stop-Process -Force`. Then re-launch `START_AEGIS.bat`. |
| **Missing Module Error (e.g. `No module named 'src'`)** | Script launched from outside project root directory without virtual environment. | Always run commands using `.\.venv\Scripts\python.exe` from `C:\D drive\Label-free-threat-detection`. |
| **Terminal Closes Immediately** | Virtual environment `.venv` interpreter is missing or moved. | Verify `.venv\Scripts\python.exe` exists. If corrupted, run: `python -m venv .venv` and install `requirements.txt`. |
| **`Incomplete Evaluation Data` Warning in Sandbox** | Browser cache holding outdated frontend state before the baseline mapping fix. | Press `Ctrl + Shift + R` (hard refresh) in browser, or click the **"Retry Sync"** button in View 5. |
| **Simulation Runs Too Fast to Observe** | Playback speed `--eps` set to 0 (unthrottled). | Launch with `--eps 10` or `--eps 15` (e.g., `.\.venv\Scripts\python.exe run_demo.py --eps 15`). |
