
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "experiment"
OUTPUT = ROOT / "outputs" / "evaluation"

TRAIN_FILE = DATA / "train_benign.parquet"
TEST_FILE = DATA / "test_tue_fri.parquet"

ATTACKS = ["PortScan", "FTP-Patator", "SSH-Patator"]

FEATURES = [
    "PSH Flag Count",
    "Protocol",
    "Init_Win_bytes_backward",
    "Packet Length Std",
    "Down/Up Ratio",
]


def main():
    print("Loading training and test data...")
    train = pd.read_parquet(TRAIN_FILE)
    test = pd.read_parquet(TEST_FILE)

    missing = [
        feature for feature in FEATURES
        if feature not in train.columns or feature not in test.columns
    ]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    rows = []

    # Training data provides the benign reference distribution.
    benign = train[FEATURES]

    groups = [("BENIGN_TRAIN", benign)]

    for attack in ATTACKS:
        subset = test.loc[test["Label"].astype(str) == attack, FEATURES]
        if len(subset):
            groups.append((attack, subset))

    for group_name, frame in groups:
        for feature in FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            if finite.empty:
                continue

            rows.append({
                "traffic_group": group_name,
                "feature": feature,
                "rows": int(len(finite)),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "zero_pct": float((finite == 0).mean() * 100),
                "p50": float(finite.quantile(0.50)),
                "p90": float(finite.quantile(0.90)),
                "p99": float(finite.quantile(0.99)),
                "min": float(finite.min()),
                "max": float(finite.max()),
            })

    report = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT / "attack_feature_distributions.csv"
    report.to_csv(output_file, index=False)

    print("\n=== Feature Distribution Summary ===")
    print(report.to_string(index=False))
    print(f"\nSaved: {output_file}")
    print(
        "\nExploratory analysis only. Test attack labels are not being "
        "used to select a model or threshold."
    )


if __name__ == "__main__":
    main()
