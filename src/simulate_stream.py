"""
src/simulate_stream.py
Real-Time Network Traffic Streaming Simulator & Live Threat Radar.

Simulates real-world one-way packet capture / flow arrival:
- Plays back real flows from the Tuesday-Friday test set.
- Evaluates each incoming flow in sub-millisecond real-time using the Deep Autoencoder.
- Emits structured SOC alerts with XAI forensic root-cause attribution.
- Writes live stream state to outputs/evaluation/live_stream_feed.json for the dashboard.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from features import ONE_WAY_FEATURES, sanitize_labels
from explainability import ThreatExplainer

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "outputs" / "experiment" / "test_tue_fri.parquet"
LIVE_FEED_FILE = ROOT / "outputs" / "evaluation" / "live_stream_feed.json"
LIVE_FEED_FILE.parent.mkdir(parents=True, exist_ok=True)


def run_stream_simulation(
    speed_eps: float = 20.0,
    max_events: int = 500,
    inject_attacks: bool = True,
):
    """
    Simulates a streaming network tap / data diode ingest.
    speed_eps: Events (flows) per second to simulate.
    """
    print("=" * 75)
    print("SIH 2026 — REAL-TIME ONE-WAY THREAT DETECTION STREAM SIMULATOR")
    print(f"Target playback rate: {speed_eps} flows/sec")
    print("=" * 75)

    print("1. Initializing Threat Explainer & Deep Autoencoder...")
    explainer = ThreatExplainer()
    print(f"   Model threshold: {explainer.threshold:.6f}")

    print("2. Sampling stream sequence from Tuesday-Friday test capture...")
    df = pd.read_parquet(TEST_FILE)
    df["Clean_Label"] = sanitize_labels(df["Label"])

    if inject_attacks:
        # Sample realistic sequence: mostly benign with interspersed attack bursts
        benign_pool = df[df["Clean_Label"] == "BENIGN"].sample(n=min(len(df), int(max_events * 0.75)), random_state=42)
        attack_pool = df[df["Clean_Label"] != "BENIGN"].sample(n=min(len(df), int(max_events * 0.25)), random_state=42)
        stream_pool = pd.concat([benign_pool, attack_pool]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        stream_pool = df.sample(n=min(len(df), max_events), random_state=42).reset_index(drop=True)

    print(f"   Stream buffer ready: {len(stream_pool)} flows queued.\n")
    print(f"{'TIME':<10} | {'GROUND TRUTH':<20} | {'SCORE':<8} | {'STATUS':<15} | {'TOP ANOMALOUS FEATURE'}")
    print("-" * 85)

    delay = 1.0 / max(speed_eps, 1.0)
    recent_alerts: List[Dict[str, Any]] = []
    total_processed = 0
    total_anomalies = 0

    for idx, row in stream_pool.iterrows():
        flow_dict = row[ONE_WAY_FEATURES].to_dict()
        true_label = str(row["Clean_Label"])

        # Real-time inference & attribution
        t0 = time.time()
        explanation = explainer.explain_flow(flow_dict)
        latency_ms = (time.time() - t0) * 1000.0

        is_alert = explanation["is_anomaly"]
        score = explanation["anomaly_score"]
        top_driver = explanation["top_attributions"][0]["feature"] if explanation["top_attributions"] else "None"
        driver_pct = explanation["top_attributions"][0]["residual_contribution_pct"] if explanation["top_attributions"] else 0.0

        total_processed += 1
        if is_alert:
            total_anomalies += 1
            status_str = f"\033[91mALERT [{explanation['severity']}]\033[0m"
        else:
            status_str = "\033[92mPASS [NORMAL]\033[0m"

        curr_time = time.strftime("%H:%M:%S")
        print(f"{curr_time:<10} | {true_label:<20} | {score:<8.4f} | {status_str:<24} | {top_driver} ({driver_pct:.1f}%)")

        alert_record = {
            "timestamp": curr_time,
            "flow_index": total_processed,
            "ground_truth": true_label,
            "score": score,
            "threshold": explanation["threshold"],
            "verdict": explanation["verdict"],
            "severity": explanation["severity"],
            "latency_ms": round(latency_ms, 2),
            "top_drivers": explanation["top_attributions"][:3],
            "narrative": explanation["analyst_narrative"],
        }
        recent_alerts.append(alert_record)
        if len(recent_alerts) > 50:
            recent_alerts.pop(0)

        # Write latest state to shared live feed for Dashboard consumption
        feed_state = {
            "last_updated": curr_time,
            "total_processed": total_processed,
            "total_anomalies": total_anomalies,
            "anomaly_rate_pct": round((total_anomalies / total_processed) * 100, 2),
            "recent_alerts": recent_alerts,
        }
        with open(LIVE_FEED_FILE, "w", encoding="utf-8") as f:
            json.dump(feed_state, f, indent=2)

        if max_events and total_processed >= max_events:
            break

        time.sleep(delay)

    print("-" * 85)
    print(f"\nStream simulation complete. Processed {total_processed} flows, detected {total_anomalies} anomalies.")


def main():
    parser = argparse.ArgumentParser(description="One-Way Network Traffic Streaming Simulator")
    parser.add_argument("--eps", type=float, default=25.0, help="Events per second (simulation speed)")
    parser.add_argument("--events", type=int, default=50, help="Total events to simulate (default: 50 for quick demo)")
    args = parser.parse_args()

    run_stream_simulation(speed_eps=args.eps, max_events=args.events)


if __name__ == "__main__":
    main()

