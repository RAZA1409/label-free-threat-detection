"""
dashboard/server.py
Zero-dependency Python HTTP server for the SIH 2026 Threat Detection Dashboard.
Runs locally on http://localhost:8501
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import sys

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT_DIR = DASHBOARD_DIR.parent
EVAL_DIR = ROOT_DIR / "outputs" / "evaluation"

PORT = 8501


class ThreatDetectionDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
            return super().do_GET()

        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Load actual evaluated metrics
            models_data = {}

            # 1. One-Way Autoencoder
            ae_path = EVAL_DIR / "oneway_autoencoder_metrics.json"
            if ae_path.exists():
                with open(ae_path, encoding="utf-8") as f:
                    models_data["oneway_autoencoder"] = json.load(f)

            # 2. One-Way Log PCA
            pca_path = EVAL_DIR / "oneway_log_pca_metrics.json"
            if pca_path.exists():
                with open(pca_path, encoding="utf-8") as f:
                    models_data["oneway_log_pca"] = json.load(f)

            # 3. Bidirectional Reduced PCA (Benchmark)
            reduced_path = EVAL_DIR / "reduced_pca_metrics.json"
            if reduced_path.exists():
                with open(reduced_path, encoding="utf-8") as f:
                    models_data["bidirectional_reference"] = json.load(f)

            self.wfile.write(json.dumps(models_data).encode("utf-8"))
            return

        elif self.path == "/api/live_feed":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            live_path = EVAL_DIR / "live_stream_feed.json"
            if live_path.exists():
                with open(live_path, encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(json.dumps({"status": "idle", "recent_alerts": []}).encode("utf-8"))
            return

        return super().do_GET()


def main():
    print("=" * 70, flush=True)
    print("SIH 2026 — Label-Free Threat Detection Dashboard", flush=True)
    print(f"Starting local dashboard server at: http://localhost:{PORT}", flush=True)
    print("Serving dashboard/index.html", flush=True)
    print("=" * 70, flush=True)

    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, ThreatDetectionDashboardHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()

