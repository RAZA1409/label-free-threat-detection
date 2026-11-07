# AEGIS: Label-Free Threat Detection in One-Way Network Traffic

**SIH 2026 — Problem ID: SIH26145**
**Repository:** [RAZA1409/label-free-threat-detection](https://github.com/RAZA1409/label-free-threat-detection)

AEGIS is an unsupervised machine learning cybersecurity system designed for unidirectional data diodes and optical network taps. It learns baseline benign network behavior and detects volumetric, scanning, and zero-day threats in real time without access to reverse traffic or ground-truth attack labels.

---

## Architecture & Detection Pipeline

- **Production Champion (v1):** 32-dimensional Residual-Weighted Temporal Autoencoder operating at threshold `0.190885` (93.08% overall recall across 2.3M evaluation flows).
- **Experimental Shadow Candidate (v2):** 26-dimensional Pure One-Way Autoencoder operating without reverse-direction features.
- **Real-Time SOC Dashboard:** Zero-dependency HTTP server with 5 live views (Executive Overview, Live Radar Ingest, Threat Analytics, Forensic XAI Sandbox, and Model Calibration).
- **Causal Streaming Engine:** Preserves strict chronological event order with rolling 10-second forward velocities.

---

## One-Click Launch & Operation Guide

### 1. First-Time Setup
Ensure Python 3.11+ is installed on your Windows system, then set up the virtual environment:

```powershell
# From the project root:
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### 2. Create the Desktop Shortcut
Generate a one-click desktop shortcut:

```powershell
.\.venv\Scripts\python.exe scripts/create_desktop_shortcut.py
```
This detects your Windows Desktop directory and creates `AEGIS – Threat Detection.lnk` configured with project working directory and system security icon.

### 3. Launching AEGIS
Launch AEGIS using either of the following methods:
- **Desktop Shortcut:** Double-click `AEGIS – Threat Detection.lnk` on your Desktop.
- **Batch Launcher:** Double-click or run `START_AEGIS.bat` from the project root.

The launcher will:
1. Verify the `.venv` virtual environment and project files.
2. Check for `GEMINI_API_KEY` in environment or local `.env`.
3. Check port `8501` to prevent duplicate server instances.
4. Launch the local SOC Dashboard on `http://localhost:8501`.
5. Automatically open your default browser.
6. Begin the 2,000-flow live optical tap demonstration with real-time XAI alerts.

### 4. Stopping AEGIS
To shut down the demonstration and dashboard:
- Simply press `Ctrl+C` in the launcher console window, or close the console window.
- The simulation will flush remaining telemetry, safely shut down child threads, and release port `8501`.

### 5. Gemini API Key Configuration (Optional)
The system operates autonomously with offline heuristic XAI explanations. To enable optional LLM-assisted forensic narratives:
- **Option A (User Environment Variable):** Set `GEMINI_API_KEY` in Windows System/User Environment Variables.
- **Option B (Local `.env` File):** Create a `.env` file in the project root containing:
  ```text
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
  *(Note: `.env` is ignored by Git and will never be committed).*

### 6. Troubleshooting Port 8501 Conflicts
If port `8501` is already occupied by a previously running server or another application:
- `START_AEGIS.bat` detects the active port and connects to the existing dashboard without crashing.
- To free port `8501` manually:
  ```powershell
  # Find process ID using port 8501:
  Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | Select-Object OwningProcess
  # Terminate specific process ID if needed:
  Stop-Process -Id <PID> -Force
  ```

---

## Dataset
Evaluated on **CIC-IDS2017** (Tuesday–Friday test split: 2,300,825 flows across 14 attack classes).
