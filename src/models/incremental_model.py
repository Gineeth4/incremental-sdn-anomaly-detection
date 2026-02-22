# src/models/incremental_model.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import get_logger
import yaml
import json

logger = get_logger()
CFG_PATH = "configs/config.yaml"
cfg = {}
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
BATCH_SIZE = int(cfg.get("training", {}).get("batch_size", 500))

PROCESSED_DATA_PATH = "data/processed/sdn_key_usage_processed.csv"
MODEL_PATH = "experiments/results/base_model.pkl"
META_PATH = "experiments/results/model_meta.json"
ANOMALY_OUTPUT_PATH = "experiments/results/detected_anomalies.csv"
PERF_PATH = "experiments/results/stream_performance.csv"

def load_stream_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error("Processed data not found.")
        raise FileNotFoundError(PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["label","timestamp"], errors="ignore")
    y = df["label"].astype(int)
    return X, y, df

def stream_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X.iloc[i:i+batch_size], y.iloc[i:i+batch_size], i//batch_size

def safe_auc(y_true, proba):
    try:
        y_true = np.asarray(y_true)
        proba = np.asarray(proba)
        if len(np.unique(y_true)) == 2:
            return float(roc_auc_score(y_true, proba))
    except Exception:
        pass
    return None

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        logger.error("Base model missing. Run base_model.py first.")
        raise FileNotFoundError(MODEL_PATH)

    model = joblib.load(MODEL_PATH)
    # meta
    model_type = None
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                model_type = json.load(f).get("model_type")
        except Exception:
            model_type = None

    X, y, df = load_stream_data()
    results = []
    perf = []
    all_true = []
    all_proba = []

    for X_batch, y_batch, batch_id in stream_batches(X, y, BATCH_SIZE):
        if len(X_batch) == 0:
            continue
        preds = model.predict(X_batch)
        proba = model.predict_proba(X_batch)[:, 1] if hasattr(model, "predict_proba") else None

        # compute metrics if both classes present
        valid = set(np.unique(y_batch))
        metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None}
        if 0 in valid and 1 in valid:
            try:
                metrics["accuracy"] = float(accuracy_score(y_batch, preds))
                metrics["precision"] = float(precision_score(y_batch, preds, zero_division=0))
                metrics["recall"] = float(recall_score(y_batch, preds, zero_division=0))
                metrics["f1"] = float(f1_score(y_batch, preds, zero_division=0))
            except Exception:
                pass
        else:
            logger.info(f"Batch {batch_id} contains classes {sorted(list(valid))}; supervised batch metrics skipped.")

        if proba is not None:
            metrics["auc"] = safe_auc(y_batch, proba)

        perf.append({"batch_id": batch_id, **metrics})
        logger.info(f"Batch {batch_id} metrics: Acc={metrics['accuracy']} Prec={metrics['precision']} Rec={metrics['recall']} F1={metrics['f1']} AUC={metrics['auc']}")

        for i in range(len(X_batch)):
            results.append({
                "batch_id": batch_id,
                "true_label": int(y_batch.iloc[i]),
                "predicted_label": int(preds[i]),
                "score": float(proba[i]) if proba is not None else None
            })

        if proba is not None:
            all_true.extend(y_batch.tolist())
            all_proba.extend(list(proba))

    os.makedirs(os.path.dirname(ANOMALY_OUTPUT_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(ANOMALY_OUTPUT_PATH, index=False)
    pd.DataFrame(perf).to_csv(PERF_PATH, index=False)
    overall_auc = safe_auc(all_true, all_proba) if len(all_proba) > 0 else None
    logger.info(f"Overall ROC AUC (if computed): {overall_auc}")
    joblib.dump(model, MODEL_PATH)
    logger.info("Incremental detection finished.")
