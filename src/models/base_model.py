# src/models/base_model.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.utils.logger import get_logger
import yaml
import json

logger = get_logger()
CFG_PATH = "configs/config.yaml"
cfg = {}
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}

PROCESSED_DATA_PATH = "data/processed/sdn_key_usage_processed.csv"
MODEL_PATH = "experiments/results/base_model.pkl"
META_PATH = "experiments/results/model_meta.json"
HOLDOUT_PATH = "experiments/results/holdout_test.csv"

TEST_SIZE = 0.2
RANDOM_STATE = int(cfg.get("training", {}).get("random_state", 42))
ABS_CLIP_VALUE = float(cfg.get("preprocess", {}).get("abs_clip_value", 1e9))

def load_processed():
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error("Processed data not found. Run preprocess first.")
        raise FileNotFoundError(PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df

def _prepare_Xy(df):
    X = df.drop(columns=["label","timestamp"], errors="ignore")
    y = df["label"].astype(int)
    # ensure numeric dtype
    X = X.apply(pd.to_numeric, errors="coerce")
    # replace inf, nan; clip
    inf_count = np.isinf(X.values).sum()
    nan_count = np.isnan(X.values).sum()
    if inf_count > 0 or nan_count > 0:
        logger.info(f"Found inf={inf_count}, nan={nan_count} in feature matrix; replacing and filling.")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # clip extreme values
    X = X.clip(lower=-ABS_CLIP_VALUE, upper=ABS_CLIP_VALUE)
    return X, y

def train_random_forest(df):
    labeled = df[df["label"].isin([0,1])].reset_index(drop=True)
    if len(labeled) == 0:
        return None
    X_all, y_all = _prepare_Xy(labeled)
    X_train, X_hold, y_train, y_hold = train_test_split(X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE)
    # final check: ensure finite
    finite_mask = np.isfinite(X_train.values).all()
    if not finite_mask:
        logger.warning("Non-finite values remain in X_train after sanitization.")
    # optionally cast to float32/64
    X_train = X_train.astype(float)
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight='balanced', random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    # save holdout for evaluation
    holdout = X_hold.copy()
    holdout["label"] = y_hold.values
    os.makedirs(os.path.dirname(HOLDOUT_PATH), exist_ok=True)
    holdout.to_csv(HOLDOUT_PATH, index=False)
    # save model and meta
    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"model_type": "random_forest", "features": list(X_all.columns)}, f)
    logger.info(f"Trained RandomForest (n={len(X_all.columns)} features). Model saved to {MODEL_PATH}")
    # log basic holdout performance
    try:
        preds = model.predict(X_hold)
        logger.info("Holdout performance:\n" + classification_report(y_hold, preds))
    except Exception as e:
        logger.warning(f"Could not compute holdout report: {e}")
    return model

if __name__ == "__main__":
    os.makedirs("experiments/results", exist_ok=True)
    df = load_processed()
    model = train_random_forest(df)
    if model is None:
        logger.error("No labeled data found (0/1). Cannot train supervised RandomForest.")
        raise ValueError("No labeled data")
    logger.info("Base model training complete.")
