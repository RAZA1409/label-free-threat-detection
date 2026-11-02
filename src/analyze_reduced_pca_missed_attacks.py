
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "outputs/experiment/test_tue_fri.parquet"
PREDICTIONS_PATH = ROOT / "outputs/evaluation/reduced_pca_predictions.parquet"
METRICS_PATH = ROOT / "outputs/evaluation/reduced_pca_metrics.json"

OUT_DIR = ROOT / "outputs/evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading test data and 39-feature PCA predictions...")

    test = pd.read_parquet(TEST_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    # Verify row alignment before combining labels and predictions.
    if len(test) != len(predictions):
        raise ValueError(
            f"Row mismatch: test={len(test):,}, "
            f"predictions={len(predictions):,}"
        )

    required_columns = {"Label"}
    if not required_columns.issubset(test.columns):
        raise ValueError("Test data does not contain the Label column.")

    if not {"anomaly_score", "is_anomaly"}.issubset(predictions.columns):
        raise ValueError(
            "Prediction file must contain anomaly_score and is_anomaly."
        )

    df = pd.DataFrame({
        "category": test["Label"].astype(str).str.strip(),
        "anomaly_score": predictions["anomaly_score"].astype(float),
        "detected": predictions["is_anomaly"].astype(bool),
    })

    if not np.isfinite(df["anomaly_score"]).all():
        raise ValueError("Non-finite anomaly scores found.")

    print(f"Total test flows: {len(df):,}")

    # 1. Detection summary by category.
    summary = (
        df.groupby("category")
        .agg(
            flows=("detected", "size"),
            detected=("detected", "sum"),
            score_median=("anomaly_score", "median"),
            score_mean=("anomaly_score", "mean"),
            score_p95=("anomaly_score", lambda x: x.quantile(0.95)),
            score_p99=("anomaly_score", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )

    summary["missed"] = summary["flows"] - summary["detected"]
    summary["detection_rate_pct"] = (
        summary["detected"] / summary["flows"] * 100
    )

    # 2. Score distributions for detected and missed flows.
    score_rows = []

    for category, group in df.groupby("category"):
        for status, subset in [
            ("Detected", group[group["detected"]]),
            ("Missed", group[~group["detected"]]),
        ]:
            if subset.empty:
                continue

            scores = subset["anomaly_score"]

            score_rows.append({
                "category": category,
                "status": status,
                "flows": len(subset),
                "score_min": scores.min(),
                "score_median": scores.median(),
                "score_mean": scores.mean(),
                "score_p75": scores.quantile(0.75),
                "score_p90": scores.quantile(0.90),
                "score_p95": scores.quantile(0.95),
                "score_max": scores.max(),
            })

    score_summary = pd.DataFrame(score_rows)

    # 3. Save reports.
    summary = summary.sort_values(
        "detection_rate_pct", ascending=True
    )

    summary_path = OUT_DIR / "reduced_pca_category_detection.csv"
    scores_path = OUT_DIR / "reduced_pca_category_score_distribution.csv"

    summary.to_csv(summary_path, index=False)
    score_summary.to_csv(scores_path, index=False)

    # 4. Print readable results.
    print("\n=== CATEGORY DETECTION SUMMARY ===")
    print(
        summary[
            [
                "category",
                "flows",
                "detected",
                "missed",
                "detection_rate_pct",
                "score_median",
                "score_p99",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n=== DETECTED VS MISSED SCORE DISTRIBUTIONS ===")
    print(
        score_summary[
            [
                "category",
                "status",
                "flows",
                "score_median",
                "score_p90",
                "score_p95",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nSaved reports:")
    print(summary_path)
    print(scores_path)

    print(
        "\nNote: This is exploratory test-set diagnosis. "
        "Do not use these test labels to select features or tune "
        "the model threshold."
    )


if __name__ == "__main__":
    main()
