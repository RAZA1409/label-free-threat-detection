
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "experiment"
EVAL = ROOT / "outputs" / "evaluation"
MODELS = ROOT / "models"

TEST_FILE = DATA / "test_tue_fri.parquet"
PREDICTIONS_FILE = EVAL / "pca_predictions.parquet"
BATCH_SIZE = 20_000

TARGET_ATTACKS = ["PortScan", "FTP-Patator", "SSH-Patator"]


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def main():
    print("Loading test data and PCA predictions...")

    test = pd.read_parquet(require_file(TEST_FILE))
    predictions = pd.read_parquet(require_file(PREDICTIONS_FILE))

    if len(test) != len(predictions):
        raise ValueError("Test and prediction row counts do not match.")

    
    if not test["Timestamp"].reset_index(drop=True).equals(
    	predictions["Timestamp"].reset_index(drop=True)
    ):
    	raise ValueError("Test and prediction rows do not match in Timestamp.")

    test_labels = test["Label"].astype(str).reset_index(drop=True)
    prediction_labels = predictions["Label"].astype(str).reset_index(drop=True)

    if not test_labels.equals(prediction_labels):
    	raise ValueError("Test and prediction rows do not match in Label.")

    labels = test["Label"].astype(str).to_numpy()
    predicted = predictions["predicted_anomaly"].astype(bool).to_numpy()
    scores = predictions["anomaly_score"].to_numpy()

    print("\n=== PCA Anomaly-Score Distributions ===")
    score_rows = []

    categories = ["BENIGN"] + TARGET_ATTACKS

    for category in categories:
        mask = labels == category
        if not mask.any():
            print(f"Warning: category not found: {category}")
            continue

        category_scores = scores[mask]
        missed_mask = mask & ~predicted

        row = {
            "category": category,
            "flows": int(mask.sum()),
            "detected": int((mask & predicted).sum()),
            "missed": int(missed_mask.sum()),
            "detection_rate_pct": round(
                100 * predicted[mask].mean(), 3
            ),
            "score_median": float(np.median(category_scores)),
            "score_p90": float(np.quantile(category_scores, 0.90)),
            "score_p95": float(np.quantile(category_scores, 0.95)),
            "score_p99": float(np.quantile(category_scores, 0.99)),
        }
        score_rows.append(row)

        print(
            f"{category:12} | flows={row['flows']:>9,} "
            f"| detected={row['detected']:>8,} "
            f"| missed={row['missed']:>8,} "
            f"| detection={row['detection_rate_pct']:>7.3f}% "
            f"| median={row['score_median']:.6f} "
            f"| p95={row['score_p95']:.6f}"
        )

    score_report = pd.DataFrame(score_rows)
    score_report.to_csv(
        EVAL / "pca_missed_attack_score_summary.csv", index=False
    )

    print("\nLoading PCA model and preprocessing objects...")

    features = joblib.load(MODELS / "pca_feature_columns.joblib")
    imputer = joblib.load(MODELS / "pca_imputer.joblib")
    scaler = joblib.load(MODELS / "pca_scaler.joblib")
    pca = joblib.load(MODELS / "pca_model.joblib")

    if not all(feature in test.columns for feature in features):
        raise ValueError("One or more PCA features are missing from test data.")

    # Aggregate squared reconstruction errors per feature without
    # keeping the entire test residual matrix in memory.
    groups = {
        "BENIGN": labels == "BENIGN",
        "PortScan_missed": (labels == "PortScan") & ~predicted,
        "FTP_Patator_missed": (labels == "FTP-Patator") & ~predicted,
        "SSH_Patator_missed": (labels == "SSH-Patator") & ~predicted,
    }

    sums = {
        name: np.zeros(len(features), dtype=np.float64)
        for name in groups
    }
    counts = {name: 0 for name in groups}

    print("\nCalculating per-feature reconstruction errors...")

    for start in range(0, len(test), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(test))
        batch = test.iloc[start:stop][features]

        values = imputer.transform(batch)
        values = scaler.transform(values)
        reconstructed = pca.inverse_transform(pca.transform(values))

        residuals = np.square(values - reconstructed)
        batch_labels = labels[start:stop]
        batch_predicted = predicted[start:stop]

        batch_groups = {
            "BENIGN": batch_labels == "BENIGN",
            "PortScan_missed": (
                (batch_labels == "PortScan") & ~batch_predicted
            ),
            "FTP_Patator_missed": (
                (batch_labels == "FTP-Patator") & ~batch_predicted
            ),
            "SSH_Patator_missed": (
                (batch_labels == "SSH-Patator") & ~batch_predicted
            ),
        }

        for name, mask in batch_groups.items():
            if mask.any():
                sums[name] += residuals[mask].sum(axis=0)
                counts[name] += int(mask.sum())

        if stop % 200_000 == 0 or stop == len(test):
            print(f"Processed {stop:,} / {len(test):,} rows")

    benign_mean = sums["BENIGN"] / max(counts["BENIGN"], 1)
    feature_rows = []

    for name in [
        "PortScan_missed",
        "FTP_Patator_missed",
        "SSH_Patator_missed",
    ]:
        if counts[name] == 0:
            print(f"No missed rows found for {name}; skipping.")
            continue

        attack_mean = sums[name] / counts[name]
        ratio = attack_mean / np.maximum(benign_mean, 1e-12)

        for i, feature in enumerate(features):
            feature_rows.append({
                "missed_attack_group": name,
                "feature": feature,
                "missed_attack_mean_squared_error": float(attack_mean[i]),
                "benign_mean_squared_error": float(benign_mean[i]),
                "attack_to_benign_error_ratio": float(ratio[i]),
                "error_difference": float(
                    attack_mean[i] - benign_mean[i]
                ),
            })

    feature_report = pd.DataFrame(feature_rows)
    feature_path = EVAL / "pca_missed_attack_feature_errors.csv"
    feature_report.to_csv(feature_path, index=False)

    print("\n=== Top 10 Feature Error Ratios ===")

    for name in [
        "PortScan_missed",
        "FTP_Patator_missed",
        "SSH_Patator_missed",
    ]:
        subset = feature_report[
            feature_report["missed_attack_group"] == name
        ].sort_values("attack_to_benign_error_ratio", ascending=False)

        if subset.empty:
            continue

        print(f"\n{name}")
        print(
            subset[
                [
                    "feature",
                    "attack_to_benign_error_ratio",
                    "missed_attack_mean_squared_error",
                    "benign_mean_squared_error",
                ]
            ].head(10).to_string(index=False)
        )

    print("\nReports saved:")
    print(EVAL / "pca_missed_attack_score_summary.csv")
    print(feature_path)
    print(
        "\nNote: This is exploratory analysis on the previously examined "
        "test split. Do not use these results to tune the final threshold."
    )


if __name__ == "__main__":
    main()
