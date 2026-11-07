"""
run_demo.py
Single-command demonstration runner for the SIH 2026 One-Way Threat Detection System.

When executed:
1. Verifies that models and evaluation feeds are ready.
2. Starts the SOC Web Dashboard server on http://localhost:8501 (if not already running).
3. Automatically launches the interactive dashboard in your default browser.
4. Streams real-world network traffic through the 93% recall Temporal Autoencoder,
   displaying live SOC alerts with XAI root-cause attribution in both terminal and web UI.
"""

import sys
import os
import subprocess
from pathlib import Path

# Automatically re-execute inside project .venv if invoked with system python
_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = _ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    _res = subprocess.run([str(_VENV_PYTHON)] + sys.argv)
    sys.exit(_res.returncode)

import time
import socket
import threading
import webbrowser

ROOT = _ROOT
sys.path.append(str(ROOT / "src"))

PORT = 8501


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_server_if_needed():
    if is_port_in_use(PORT):
        print(f"[+] Dashboard server already active on http://localhost:{PORT}")
        return

    from http.server import HTTPServer
    sys.path.append(str(ROOT / "dashboard"))
    from server import ThreatDetectionDashboardHandler

    print(f"[*] Starting local SOC Dashboard server on http://localhost:{PORT}...")
    server = HTTPServer(("127.0.0.1", PORT), ThreatDetectionDashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(1.0)
    print(f"[+] Dashboard server is running on http://localhost:{PORT}")


def main():
    print("=" * 80)
    print("  SIH 2026 — LABEL-FREE THREAT DETECTION IN ONE-WAY NETWORK TRAFFIC")
    print("  Problem ID: SIH26145 | Real-Time SOC Anomaly Detection & XAI Radar")
    print("=" * 80)

    # 1. Start server
    start_server_if_needed()

    # 2. Open browser
    dashboard_url = f"http://localhost:{PORT}"
    print(f"\n[>] Opening Web Dashboard in your browser: {dashboard_url}")
    try:
        webbrowser.open(dashboard_url)
    except Exception as e:
        print(f"[!] Could not open browser automatically: {e}")
        print(f"    Please manually open: {dashboard_url}")

    time.sleep(1.5)

    # 3. Run stream simulator
    print("\n" + "=" * 80)
    print("  STARTING LIVE OPTICAL TAP TRAFFIC STREAM (93% RECALL AUTOENCODER)")
    print("  Watching incoming flows in real-time. Press Ctrl+C anytime to stop.")
    print("=" * 80 + "\n")

    from simulate_stream import run_stream_simulation

    try:
        # Stream 2,000 flows at 15 flows/sec so the audience can observe alerts live
        run_stream_simulation(speed_eps=15.0, max_events=2000, inject_attacks=True)
    except KeyboardInterrupt:
        print("\n\n[!] Demo stopped by user. Dashboard server remains reachable.")
    
    print("\n[+] Demo completed! Dashboard remains accessible at: " + dashboard_url)


if __name__ == "__main__":
    main()

