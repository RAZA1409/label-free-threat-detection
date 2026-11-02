
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "experiment"
OUTPUTS = ROOT / "outputs" / "evaluation"

TRAIN_FILE = DATA / "train_benign.parquet"
MAX_ROWS = 100_000
RANDOM_STATE = 42

# Numeric features that may have skewed distributions or extreme values.
# These are candidates for investigation, not an automatic transformation list.
CANDIDATE_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Down/Up Ratio",
    "PSH Flag Count",
    "SYN Flag Count",
]


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing training data: {TRAIN_FILE}")

    train = pd.read_parquet(TRAIN_FILE)
    features = [
        col for col in CANDIDATE_FEATURES
        if col in train.columns
        and pd.api.types.is_numeric_dtype(train[col])
    ]

    if not features:
        raise ValueError(
            "None of the candidate features exist in the training data. "
            "Check the actual column names."
        )

    sample = train.sample(
        n=min(MAX_ROWS, len(train)),
        random_state=RANDOM_STATE,
    )

    rows = []

    for feature in features:
        values = pd.to_numeric(sample[feature], errors="coerce")
        finite = values[np.isfinite(values)]

        if finite.empty:
            continue

        q01 = finite.quantile(0.01)
        q25 = finite.quantile(0.25)
        q50 = finite.quantile(0.50)
        q75 = finite.quantile(0.75)
        q99 = finite.quantile(0.99)

        rows.append({
            "feature": feature,
            "rows_sampled": len(values),
            "missing_pct": values.isna().mean() * 100,
            "zero_pct": (finite == 0).mean() * 100,
            "negative_pct": (finite < 0).mean() * 100,
            "mean": finite.mean(),
            "std": finite.std(),
            "min": finite.min(),
            "p01": q01,
            "p25": q25,
            "median": q50,
            "p75": q75,
            "p99": q99,
            "max": finite.max(),
            "p99_to_median_abs_ratio": (
                abs(q99) / max(abs(q50), 1e-12)
            ),
        })

    report = pd.DataFrame(rows)
    report = report.sort_values(
        "p99_to_median_abs_ratio", ascending=False
    )

    output_file = OUTPUTS / "benign_feature_distribution_report.csv"
    report.to_csv(output_file, index=False)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.5g}",
    ):
        print("\n=== Benign Training Feature Distributions ===")
        print(report.to_string(index=False))

    print(f"\nSaved report: {output_file}")
    print(
        "\nThis report uses benign training data only. "
        "No attack labels are used to select transformations."
    )


if __name__ == "__main__":
    main()
