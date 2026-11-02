import pandas as pd
from pathlib import Path

DATA_PATH = Path("outputs/experiment/train_benign.parquet")
OUTPUT_PATH = Path("outputs/evaluation/feature_family_summary.csv")

df = pd.read_parquet(DATA_PATH)

numeric_df = df.select_dtypes(include="number")

feature_families = {
    "Flow Duration": ["Flow Duration"],

    "Packet Counts": [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Subflow Fwd Packets",
        "Subflow Bwd Packets"
    ],

    "Packet Length": [
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std"
    ],

    "Flow Inter Arrival Time": [
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Min"
    ],

    "Forward Inter Arrival Time": [
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Min"
    ],

    "Backward Inter Arrival Time": [
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Min"
    ],

    "Flow Rates": [
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s"
    ],

    "TCP Flags": [
        "ACK Flag Count",
        "FIN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "URG Flag Count",
        "Fwd PSH Flags"
    ],

    "TCP Window": [
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward"
    ],

    "Header Information": [
        "Fwd Header Length",
        "Bwd Header Length",
        "min_seg_size_forward"
    ],

    "Other": [
        "Protocol",
        "Down/Up Ratio",
        "act_data_pkt_fwd",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Idle Mean",
        "Idle Std"
    ]
}

rows = []

for family, features in feature_families.items():

    available = [
        f for f in features
        if f in numeric_df.columns
    ]

    if not available:
        continue

    for feature in available:

        series = numeric_df[feature]

        rows.append({
            "family": family,
            "feature": feature,
            "missing_pct": series.isna().mean() * 100,
            "zero_pct": (series == 0).mean() * 100,
            "unique_values": series.nunique(),
            "mean": series.mean(),
            "std": series.std(),
            "median": series.median(),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99)
        })

report = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

report.to_csv(OUTPUT_PATH, index=False)

print("\n=== FEATURE FAMILY SUMMARY ===")

for family, group in report.groupby("family"):

    print(f"\n{family}")
    print(group[
        [
            "feature",
            "missing_pct",
            "zero_pct",
            "unique_values",
            "median",
            "p95",
            "p99"
        ]
    ].to_string(index=False))

print(f"\nSaved report: {OUTPUT_PATH}")