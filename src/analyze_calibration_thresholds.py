
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "experiment"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "outputs" / "evaluation"

TARGET_FPRS = [0.005, 0.01, 0.02, 0.03, 0.05]

print("Loading benign calibration data...")
calibration = pd.read_parquet(
    DATA_DIR / "calibration_benign.parquet"
)

feature_columns = joblib.load(
    MODEL_DIR / "feature_columns.joblib"
)
imputer = joblib.load(MODEL_DIR / "feature_imputer.joblib")
model = joblib.load(MODEL_DIR / "isolation_forest.joblib")

X_cal = calibration[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
X_cal = imputer.transform(X_cal)

scores = -model.decision_function(X_cal)

rows = []
for target_fpr in TARGET_FPRS:
    threshold = float(np.quantile(scores, 1 - target_fpr))
    observed_fpr = float(np.mean(scores >= threshold))

    rows.append({
        "target_calibration_fpr": target_fpr,
        "threshold": threshold,
        "observed_calibration_fpr": observed_fpr,
        "calibration_rows": len(scores),
        "calibration_alerts": int(np.sum(scores >= threshold)),
    })

results = pd.DataFrame(rows)

EVAL_DIR.mkdir(parents=True, exist_ok=True)
output = EVAL_DIR / "calibration_thresholds.csv"
results.to_csv(output, index=False)

print("\n=== CALIBRATION-ONLY THRESHOLD ANALYSIS ===")
print(results.to_string(
    index=False,
    formatters={
        "target_calibration_fpr": "{:.2%}".format,
        "threshold": "{:.6f}".format,
        "observed_calibration_fpr": "{:.2%}".format,
    },
))
print(f"\nSaved results to: {output}")
