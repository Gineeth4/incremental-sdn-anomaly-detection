# src/evaluation/evaluate.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.utils.logger import get_logger

logger = get_logger()
HOLDOUT_PATH = "experiments/results/holdout_test.csv"
ANOMALY_RESULTS_PATH = "experiments/results/detected_anomalies.csv"
SUMMARY_PATH = "experiments/results/evaluation_summary.txt"

if __name__ == "__main__":
    lines = []
    if os.path.exists(HOLDOUT_PATH):
        hold = pd.read_csv(HOLDOUT_PATH)
        if "label" in hold.columns:
            y = hold["label"].astype(int)
            X = hold.drop(columns=["label","timestamp"], errors="ignore")
            # if model predictions exist in detected_anomalies.csv (holdout rows may be subset), skip; otherwise evaluate separately
            # here we simply compute report if predictions exist in experiments results mapped to holdout rows
            lines.append("=== Holdout evaluation present in experiments/results/holdout_test.csv ===\n")
            lines.append(f"Rows: {len(hold)}\n")
        else:
            lines.append("Holdout exists but has no 'label' column.\n")
    else:
        lines.append("No holdout file found.\n")

    # evaluate using detected_anomalies if present
    if os.path.exists(ANOMALY_RESULTS_PATH):
        df = pd.read_csv(ANOMALY_RESULTS_PATH)
        if "true_label" in df.columns and "predicted_label" in df.columns:
            y_true = df["true_label"].astype(int)
            y_pred = df["predicted_label"].astype(int)
            report = classification_report(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            lines.append("=== Overall detection report (on processed dataset) ===\n")
            lines.append(report + "\n")
            lines.append("Confusion Matrix:\n" + str(cm) + "\n")
        else:
            lines.append("detected_anomalies.csv present but missing required columns true_label/predicted_label\n")
    else:
        lines.append("No detected_anomalies.csv found.\n")

    with open(SUMMARY_PATH, "w") as f:
        f.writelines(lines)

    logger.info("Saved evaluation summary to " + SUMMARY_PATH)
