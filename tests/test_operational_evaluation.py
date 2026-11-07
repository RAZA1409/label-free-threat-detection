"""
tests/test_operational_evaluation.py
Automated validation of AEGIS Phase 2 Stage 3 operational evaluation artifacts.

Validates:
1. File integrity and schema of evaluation artifacts in reports/phase_2_operational_results/
2. Invariant conservation: TP + FP + TN + FN == 2,300,825
3. Label distribution: 1,743,179 benign, 557,646 attacks across 14 categories
4. Metric mathematical consistency across all models and thresholds
5. Per-day partition alignment (Tuesday to Friday)
"""

import json
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "reports" / "phase_2_operational_results"
OVERALL_FILE = RESULTS_DIR / "overall_metrics.json"
CONFUSION_FILE = RESULTS_DIR / "confusion_matrices.json"
ATTACK_FILE = RESULTS_DIR / "per_attack_metrics.csv"
DAY_FILE = RESULTS_DIR / "per_day_metrics.csv"

TOTAL_FLOWS = 2_300_825
BENIGN_FLOWS = 1_743_179
ATTACK_FLOWS = 557_646


def test_01_operational_result_files_exist():
    """Verify all machine-readable operational evaluation files exist and are non-empty."""
    assert OVERALL_FILE.exists(), f"Missing {OVERALL_FILE}"
    assert CONFUSION_FILE.exists(), f"Missing {CONFUSION_FILE}"
    assert ATTACK_FILE.exists(), f"Missing {ATTACK_FILE}"
    assert DAY_FILE.exists(), f"Missing {DAY_FILE}"

    assert OVERALL_FILE.stat().st_size > 100
    assert CONFUSION_FILE.stat().st_size > 50
    assert ATTACK_FILE.stat().st_size > 100
    assert DAY_FILE.stat().st_size > 100


def test_02_confusion_matrix_invariants():
    """Verify TP + FP + TN + FN == 2,300,825 and marginal totals match ground truth."""
    with open(CONFUSION_FILE, "r", encoding="utf-8") as f:
        cm_data = json.load(f)

    expected_models = [
        "pure_one_way_v2_primary",
        "pure_one_way_v2_reference",
        "legacy_champion_v1",
    ]

    for model_key in expected_models:
        assert model_key in cm_data, f"Missing model {model_key} in confusion matrices"
        cm = cm_data[model_key]
        tp = cm["tp"]
        fp = cm["fp"]
        tn = cm["tn"]
        fn = cm["fn"]

        # Sum of all elements must equal total dataset rows
        assert (tp + fp + tn + fn) == TOTAL_FLOWS, f"{model_key} total sum mismatch"

        # Marginal ground truth sums must be strictly preserved
        assert (tn + fp) == BENIGN_FLOWS, f"{model_key} benign total mismatch"
        assert (tp + fn) == ATTACK_FLOWS, f"{model_key} attack total mismatch"


def test_03_metric_mathematical_consistency():
    """Verify precision, recall, FPR, FNR, specificity, accuracy, and F1 calculations."""
    with open(OVERALL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for model_key, res in data["models"].items():
        cm = res["confusion_matrix"]
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

        expected_recall = round((tp / (tp + fn)) * 100, 4)
        expected_fpr = round((fp / (fp + tn)) * 100, 4)
        expected_precision = round((tp / (tp + fp)) * 100, 4)
        expected_fnr = round((fn / (tp + fn)) * 100, 4)
        expected_specificity = round((tn / (fp + tn)) * 100, 4)
        expected_f1 = round(2 * tp / (2 * tp + fp + fn), 4)

        assert res["rates"]["detection_rate_pct"] == expected_recall
        assert res["rates"]["false_positive_rate_pct"] == expected_fpr
        assert res["rates"]["false_negative_rate_pct"] == expected_fnr
        assert res["rates"]["specificity_pct"] == expected_specificity
        assert res["classification_metrics"]["precision_pct"] == expected_precision
        assert res["classification_metrics"]["f1_score"] == expected_f1


def test_04_per_attack_category_coverage():
    """Verify all 14 attack categories + BENIGN are evaluated in per_attack_metrics.csv."""
    df_attack = pd.read_csv(ATTACK_FILE)
    assert len(df_attack) == 15, f"Expected 15 categories, found {len(df_attack)}"

    expected_categories = {
        "BENIGN",
        "Bot",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris",
        "FTP-Patator",
        "Heartbleed",
        "Infiltration",
        "PortScan",
        "SSH-Patator",
        "Web Attack - Brute Force",
        "Web Attack - Sql Injection",
        "Web Attack - XSS",
    }
    actual_categories = set(df_attack["Category"])
    assert actual_categories == expected_categories, f"Category mismatch: {expected_categories - actual_categories}"

    # Sum of counts must equal total flows
    assert df_attack["Total_Count"].sum() == TOTAL_FLOWS


def test_05_per_day_partition_coverage():
    """Verify Tuesday through Friday coverage and flow count conservation."""
    df_day = pd.read_csv(DAY_FILE)
    assert len(df_day) == 4

    expected_days = ["Tuesday", "Wednesday", "Thursday", "Friday"]
    assert list(df_day["Day"]) == expected_days

    assert df_day["Total_Flows"].sum() == TOTAL_FLOWS
    assert df_day["Benign_Flows"].sum() == BENIGN_FLOWS
    assert df_day["Attack_Flows"].sum() == ATTACK_FLOWS


def test_06_model_performance_invariants():
    """Verify key comparative invariants between pure one-way v2 and legacy champion."""
    with open(OVERALL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)["models"]

    v2_prim = data["pure_one_way_v2_primary"]
    v2_ref = data["pure_one_way_v2_reference"]
    leg = data["legacy_champion_v1"]

    # 1. Pure one-way primary has lower FPR than legacy champion
    assert v2_prim["rates"]["false_positive_rate_pct"] < leg["rates"]["false_positive_rate_pct"]
    assert v2_prim["rates"]["false_positive_rate_pct"] == 10.6982
    assert leg["rates"]["false_positive_rate_pct"] == 12.4828

    # 2. Pure one-way primary has higher precision than legacy champion
    assert v2_prim["classification_metrics"]["precision_pct"] > leg["classification_metrics"]["precision_pct"]
    assert v2_prim["classification_metrics"]["precision_pct"] == 71.8556
    assert leg["classification_metrics"]["precision_pct"] == 70.4603

    # 3. Pure one-way reference matches legacy recall within 0.1%
    assert abs(v2_ref["rates"]["detection_rate_pct"] - leg["rates"]["detection_rate_pct"]) < 0.1
    assert v2_ref["rates"]["detection_rate_pct"] == 93.0065
    assert leg["rates"]["detection_rate_pct"] == 93.0754

    # 4. DoS Hulk recall is exactly 99.85% across all models
    df_attack = pd.read_csv(ATTACK_FILE).set_index("Category")
    assert df_attack.loc["DoS Hulk", "v2_Primary_Rate_Pct"] == 99.8498
    assert df_attack.loc["DoS Hulk", "v2_Ref_Rate_Pct"] == 99.8498
    assert df_attack.loc["DoS Hulk", "v1_Legacy_Rate_Pct"] == 99.8498
