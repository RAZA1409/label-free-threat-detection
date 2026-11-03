
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "evaluation"

MODEL_FILES = {
    "Isolation Forest": "isolation_forest_predictions.parquet",
    "Robust Isolation Forest": "isolation_forest_robust_predictions.parquet",
    "Signed-log Isolation Forest": "isolation_forest_log_predictions.parquet",
    "Local Outlier Factor": "lof_predictions.parquet",
    "PCA Reconstruction Error": "pca_predictions.parquet",
    "Combined IF + PCA": "combined_predictions.parquet",
    "PCA + Feature Errors": "pca_feature_detector_predictions.parquet",
    "MiniBatchKMeans": "kmeans_predictions.parquet",
    "Targeted Isolation Forest": "targeted_if_predictions.parquet",
}


def load_predictions(model_name, filename):
    path = EVAL / filename

    if not path.exists():
        print(f"Skipping {model_name}: missing {filename}")
        return None

    frame = pd.read_parquet(path)

    required = {"Label", "predicted_anomaly"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {sorted(missing)}"
        )

    if frame["predicted_anomaly"].isna().any():
        raise ValueError(f"{filename} contains missing predictions.")

    values = set(frame["predicted_anomaly"].unique())
    if not values.issubset({0, 1, False, True}):
        raise ValueError(
            f"{filename} has unexpected prediction values: {values}"
        )

    return frame


def main():
    EVAL.mkdir(parents=True, exist_ok=True)

    predictions = {}

    for model_name, filename in MODEL_FILES.items():
        frame = load_predictions(model_name, filename)
        if frame is not None:
            predictions[model_name] = frame

    if len(predictions) < 2:
        raise RuntimeError("At least two prediction files are needed.")

    # Check row count, label order, and timestamps where available.
    reference_name = next(iter(predictions))
    reference = predictions[reference_name]

    for model_name, frame in predictions.items():
        if len(frame) != len(reference):
            raise ValueError(
                f"{model_name} has {len(frame)} rows; "
                f"{reference_name} has {len(reference)}."
            )

        if not frame["Label"].astype(str).reset_index(drop=True).equals(
            reference["Label"].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"Label order mismatch between {reference_name} and {model_name}."
            )

        if "Timestamp" in reference.columns and "Timestamp" in frame.columns:
            if not frame["Timestamp"].reset_index(drop=True).equals(
                reference["Timestamp"].reset_index(drop=True)
            ):
                raise ValueError(
                    f"Timestamp order mismatch for {model_name}."
                )

    # Build one row per attack category and model.
    rows = []

    for model_name, frame in predictions.items():
        grouped = frame.assign(
            is_attack=frame["Label"].astype(str).str.upper() != "BENIGN",
            predicted_anomaly=frame["predicted_anomaly"].astype(int),
        ).groupby("Label", sort=True)

        for label, group in grouped:
            total = len(group)
            detected = int(group["predicted_anomaly"].sum())

            rows.append({
                "model": model_name,
                "category": label,
                "flows": total,
                "detected": detected,
                "missed": total - detected,
                "detection_rate_pct": round(
                    100 * detected / total, 4
                ) if total else 0.0,
            })

    report = pd.DataFrame(rows)

    output_csv = EVAL / "all_model_attack_comparison.csv"
    report.to_csv(output_csv, index=False)

    # Print the most important hard-to-detect categories.
    focus_categories = [
        "PortScan",
        "FTP-Patator",
        "SSH-Patator",
    ]

    focus = report[
        report["category"].isin(focus_categories)
    ].pivot(
        index="model",
        columns="category",
        values="detection_rate_pct",
    )

    print("\n=== Detection rate for selected difficult attack categories (%) ===")
    print(focus.to_string(na_rep="N/A"))

    print("\n=== Models and categories included ===")
    print(f"Models: {len(predictions)}")
    print(f"Categories: {report['category'].nunique()}")
    print(f"Rows in report: {len(report)}")
    print(f"\nSaved report: {output_csv}")
    print(
        "\nNote: exploratory comparison on a previously examined test set; "
        "independent validation is still required."
    )


if __name__ == "__main__":
    main()
