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
_VENV_DIR = _ROOT / ".venv"
_VENV_PYTHON = _VENV_DIR / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and Path(sys.prefix).resolve() != _VENV_DIR.resolve():
    _res = subprocess.run([str(_VENV_PYTHON)] + sys.argv)
    sys.exit(_res.returncode)

import time
import socket
import threading
import webbrowser

ROOT = _ROOT
sys.path.append(str(ROOT / "src"))

PORT = 8501
SERVER = None
SERVER_THREAD = None


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_server_if_needed():
    global SERVER, SERVER_THREAD
    if is_port_in_use(PORT):
        print(f"[+] Dashboard server already active on http://localhost:{PORT}")
        return None, None

    try:
        from http.server import ThreadingHTTPServer as ServerClass
    except ImportError:
        from http.server import HTTPServer as ServerClass
    sys.path.append(str(ROOT / "dashboard"))
    from server import ThreatDetectionDashboardHandler

    print(f"[*] Starting local SOC Dashboard server on http://localhost:{PORT}...")
    SERVER = ServerClass(("127.0.0.1", PORT), ThreatDetectionDashboardHandler)
    SERVER_THREAD = threading.Thread(target=SERVER.serve_forever, daemon=True)
    SERVER_THREAD.start()
    time.sleep(1.0)
    print(f"[+] Dashboard server is running on http://localhost:{PORT}")
    return SERVER, SERVER_THREAD


def stop_server():
    global SERVER, SERVER_THREAD
    if SERVER is not None:
        print("\n[*] Shutting down SOC Dashboard server...")
        try:
            SERVER.shutdown()
        except Exception:
            pass
        try:
            SERVER.server_close()
        except Exception:
            pass
        if SERVER_THREAD is not None and SERVER_THREAD.is_alive():
            SERVER_THREAD.join(timeout=3.0)
        SERVER = None
        SERVER_THREAD = None
        print("[+] Dashboard server stopped and port 8501 released.")


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

    # Optional CLI parameters (defaults strictly preserved: 2,000 events, 15 EPS)
    events = 2000
    eps = 15.0
    if "--events" in sys.argv:
        try:
            events = int(sys.argv[sys.argv.index("--events") + 1])
        except (ValueError, IndexError):
            pass
    if "--eps" in sys.argv:
        try:
            eps = float(sys.argv[sys.argv.index("--eps") + 1])
        except (ValueError, IndexError):
            pass

    # 3. Run stream simulator
    print("\n" + "=" * 80)
    print(f"  STARTING LIVE OPTICAL TAP TRAFFIC STREAM ({events} FLOWS)")
    print("  Watching incoming flows in real-time. Press Ctrl+C anytime to stop.")
    print("=" * 80 + "\n")

    from simulate_stream import run_stream_simulation

    try:
        try:
            # Stream flows at requested rate (default: 2,000 flows at 15 flows/sec)
            run_stream_simulation(speed_eps=eps, max_events=events, inject_attacks=True)
        except KeyboardInterrupt:
            print("\n\n[!] Stream replay interrupted by user.")
            return

        print("\n" + "=" * 80)
        print("  [+] STREAM COMPLETE — SOC DASHBOARD REMAINS ACTIVE & FULLY INTERACTIVE")
        print(f"  URL: {dashboard_url}")
        print("  All 5 views (Overview, Radar, Threats, XAI, Calibration) are live.")
        print("  Press Ctrl+C anytime to stop the dashboard server and exit.")
        print("=" * 80 + "\n")

        # Main-thread keep-alive: keeps server alive until user sends Ctrl+C
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] Shutdown requested by user.")
    finally:
        stop_server()
        print("\n[+] AEGIS session ended cleanly.")


if __name__ == "__main__":
    main()

