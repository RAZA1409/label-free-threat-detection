"""
src/generate_evaluation_charts.py
Generates presentation-ready evaluation charts and diagrams in reports/figures/
for SIH 2026 hackathon deliverables.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})


def plot_model_comparison():
    """Bar chart comparing overall model detection rates, FPRs, and precision."""
    report_file = EVAL_DIR / "consolidated_model_report.csv"
    if not report_file.exists():
        print(f"Skipping model comparison: {report_file} not found.")
        return

    df = pd.read_csv(report_file)
    df = df.sort_values("detection_rate_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["model"], df["detection_rate_pct"], color="#2b5c8f", edgecolor="#1a365d", height=0.6)

    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontweight="bold", color="#1a365d")

    ax.set_xlim(0, max(df["detection_rate_pct"].max() + 8, 55))
    ax.set_xlabel("Attack Detection Rate (%) — Recall")
    ax.set_title("Unsupervised Threat Detection Recall across Models (Test Set)")
    plt.tight_layout()

    out_path = FIGURES_DIR / "model_detection_rate_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_attack_category_detection():
    """Horizontal bar chart showing detection recall per attack category."""
    # Check if any attack comparison file exists
    candidate_files = [
        EVAL_DIR / "temporal_autoencoder_breakdown.csv",
        EVAL_DIR / "oneway_autoencoder_breakdown.csv",
        EVAL_DIR / "reduced_pca_category_detection.csv",
        EVAL_DIR / "all_model_attack_comparison.csv",
        EVAL_DIR / "oneway_pca_attack_breakdown.csv",
    ]

    for f in candidate_files:
        if f.exists():
            df = pd.read_csv(f)
            cat_col = "category" if "category" in df.columns else df.columns[0]
            rate_col = "detection_rate_pct" if "detection_rate_pct" in df.columns else None

            if rate_col and cat_col:
                # Exclude BENIGN
                df = df[df[cat_col].astype(str).str.upper() != "BENIGN"].copy()
                df = df.sort_values(rate_col, ascending=True)

                fig, ax = plt.subplots(figsize=(10, 7))
                colors = ["#d9534f" if r < 20 else "#f0ad4e" if r < 60 else "#5cb85c" for r in df[rate_col]]
                bars = ax.barh(df[cat_col], df[rate_col], color=colors, edgecolor="#333333", height=0.65)

                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 1.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontweight="bold")

                ax.set_xlim(0, 110)
                ax.set_xlabel("Detection Rate (%)")
                ax.set_title("Zero-Day Anomaly Detection Rate by Attack Category")
                plt.tight_layout()

                out_path = FIGURES_DIR / "attack_detection_by_category.png"
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"Saved: {out_path}")
                break


def plot_fpr_sensitivity():
    """Sensitivity curve of Target Calibration FPR vs Test Attack Detection Rate."""
    sens_file = EVAL_DIR / "pca_threshold_sensitivity.csv"
    if not sens_file.exists():
        return

    df = pd.read_csv(sens_file)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = df["target_calibration_fpr"]
    y_recall = df["test_detection_rate_pct"]
    y_fpr = df["test_false_positive_rate_pct"]

    ax1.plot(x, y_recall, marker="o", color="#2b5c8f", linewidth=2.2, label="Attack Recall (%)")
    ax1.set_xlabel("Target Calibration FPR")
    ax1.set_ylabel("Detection Rate (Recall %)", color="#2b5c8f")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")

    ax2 = ax1.twinx()
    ax2.plot(x, y_fpr, marker="s", color="#d9534f", linewidth=2.2, linestyle="--", label="Observed Test FPR (%)")
    ax2.set_ylabel("Observed Test FPR (%)", color="#d9534f")
    ax2.tick_params(axis="y", labelcolor="#d9534f")

    plt.title("Threshold Calibration Sensitivity: Trade-off Curve")
    plt.tight_layout()

    out_path = FIGURES_DIR / "calibration_fpr_sensitivity.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating evaluation figures in reports/figures/...")
    plot_model_comparison()
    plot_attack_category_detection()
    plot_fpr_sensitivity()
    print("Done generating charts.")


if __name__ == "__main__":
    main()

