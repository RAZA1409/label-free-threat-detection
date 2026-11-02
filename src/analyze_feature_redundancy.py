
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "outputs/experiment/train_benign.parquet"
OUT_DIR = ROOT / "outputs/evaluation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.95


def main():
    print("Loading benign training data...")
    df = pd.read_parquet(TRAIN_PATH)

    # Only numeric model features. Never include the target label.
    features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c.lower() != "label"
    ]

    if not features:
        raise ValueError("No numeric features found.")

    sample = df[features].sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE
    )

    # Replace infinities for the quality report and correlation analysis.
    sample = sample.replace([np.inf, -np.inf], np.nan)

    print(f"Rows sampled: {len(sample):,}")
    print(f"Numeric features: {len(features)}")

    # 1. Feature quality and variability.
    rows = []

    for col in features:
        s = sample[col]
        valid = s.dropna()

        rows.append({
            "feature": col,
            "missing_pct": s.isna().mean() * 100,
            "zero_pct": (valid == 0).mean() * 100 if len(valid) else np.nan,
            "unique_values": valid.nunique(),
            "std": valid.std(),
            "median": valid.median(),
            "p25": valid.quantile(0.25),
            "p75": valid.quantile(0.75),
            "iqr": valid.quantile(0.75) - valid.quantile(0.25),
        })

    profile = pd.DataFrame(rows)

    # A feature with one observed value has no useful variation in this sample.
    profile["constant_or_near_constant"] = profile["unique_values"] <= 1

    profile = profile.sort_values(
        ["constant_or_near_constant", "missing_pct", "unique_values"],
        ascending=[False, False, True]
    )

    profile_path = OUT_DIR / "feature_quality_profile.csv"
    profile.to_csv(profile_path, index=False)

    # 2. Spearman correlation identifies monotonic redundancy.
    print("Calculating Spearman correlations...")
    correlation = sample.corr(method="spearman").abs()

    pairs = []
    columns = list(correlation.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = correlation.iat[i, j]

            if pd.notna(value) and value >= CORRELATION_THRESHOLD:
                pairs.append({
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "abs_spearman_correlation": value,
                })

    pair_df = pd.DataFrame(
        pairs,
        columns=[
            "feature_1", "feature_2",
            "abs_spearman_correlation"
        ]
    )

    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            "abs_spearman_correlation", ascending=False
        )

    pairs_path = OUT_DIR / "high_correlation_pairs.csv"
    pair_df.to_csv(pairs_path, index=False)

    # 3. Create a preliminary redundancy-pruned candidate set.
    # Preference order: fewer missing values, more distinct values,
    # then larger IQR. This is a transparent heuristic, not attack importance.
    ranked = profile.sort_values(
        ["missing_pct", "unique_values", "iqr"],
        ascending=[True, False, False]
    )

    selected = []
    excluded = []

    for _, row in ranked.iterrows():
        feature = row["feature"]

        if row["constant_or_near_constant"]:
            excluded.append({
                "feature": feature,
                "reason": "Constant in sample",
                "correlated_with": "",
                "correlation": np.nan,
            })
            continue

        redundant_with = None
        redundant_corr = None

        for kept in selected:
            value = correlation.loc[feature, kept]

            if pd.notna(value) and value >= CORRELATION_THRESHOLD:
                redundant_with = kept
                redundant_corr = value
                break

        if redundant_with is None:
            selected.append(feature)
        else:
            excluded.append({
                "feature": feature,
                "reason": "Highly correlated with selected feature",
                "correlated_with": redundant_with,
                "correlation": redundant_corr,
            })

    selected_path = OUT_DIR / "correlation_pruned_features.csv"
    pd.DataFrame({
        "feature": selected,
        "selection_note": [
            "Candidate retained after quality and redundancy screening"
        ] * len(selected),
    }).to_csv(selected_path, index=False)

    excluded_path = OUT_DIR / "excluded_redundant_features.csv"
    pd.DataFrame(
        excluded,
        columns=["feature", "reason", "correlated_with", "correlation"]
    ).to_csv(excluded_path, index=False)

    print("\n=== FEATURE QUALITY SUMMARY ===")
    print(f"Total features: {len(features)}")
    print(
        "Constant features:",
        int(profile["constant_or_near_constant"].sum())
    )
    print(
        "Features with missing values:",
        int((profile["missing_pct"] > 0).sum())
    )

    print("\n=== MOST REDUNDANT FEATURE PAIRS ===")
    if pair_df.empty:
        print("No pairs exceeded the correlation threshold.")
    else:
        print(pair_df.head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        ))

    print("\n=== PRELIMINARY FEATURE REDUCTION ===")
    print(f"Original features: {len(features)}")
    print(f"Retained candidates: {len(selected)}")
    print(f"Excluded candidates: {len(excluded)}")

    print("\nRetained features:")
    for feature in selected:
        print(f"  - {feature}")

    print("\nSaved reports:")
    print(profile_path)
    print(pairs_path)
    print(selected_path)
    print(excluded_path)

    print(
        "\nImportant: correlation pruning identifies redundancy, "
        "not proven attack-detection importance. All results use "
        "benign training data only."
    )


if __name__ == "__main__":
    main()
