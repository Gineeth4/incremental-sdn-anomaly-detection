# src/preprocessing/preprocess.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
from src.utils.logger import get_logger
import yaml

logger = get_logger()

RAW_SDN_PATH = "data/raw/sdn_key_usage.csv"
RAW_ORIG_PATH = "data/raw/original_cic.csv"
PROCESSED_DATA_PATH = "data/processed/sdn_key_usage_processed.csv"
ENC_PATH = "experiments/results/preprocess_encoders_full.pkl"
CFG_PATH = "configs/config.yaml"

cfg = {}
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, "r") as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

TRAINING_CFG = cfg.get("training", {})
FULL_FEATURES = bool(TRAINING_CFG.get("full_features", True))
RANDOM_STATE = int(TRAINING_CFG.get("random_state", 42))

# clip absolute feature values to this threshold to avoid float overflow in sklearn internals
ABS_CLIP_VALUE = float(cfg.get("preprocess", {}).get("abs_clip_value", 1e9))

def load_raw_full():
    if not os.path.exists(RAW_ORIG_PATH):
        logger.error(f"Original CIC file not found at {RAW_ORIG_PATH}. Adapter must copy it first.")
        raise FileNotFoundError(RAW_ORIG_PATH)
    return pd.read_csv(RAW_ORIG_PATH, low_memory=False)

def load_raw_mapped():
    if not os.path.exists(RAW_SDN_PATH):
        logger.error(f"Mapped SDN file not found at {RAW_SDN_PATH}.")
        raise FileNotFoundError(RAW_SDN_PATH)
    return pd.read_csv(RAW_SDN_PATH)

def _sanitize_numeric_df(df, numeric_cols):
    """
    Ensure numeric columns contain finite, reasonably-sized numbers.
    Replaces inf/-inf, fills NaN with 0, and clips magnitude to ABS_CLIP_VALUE.
    Returns how many replaced/filled across numeric_cols.
    """
    if len(numeric_cols) == 0:
        return 0
    sub = df[numeric_cols].copy()
    # replace string 'inf' if present and convert
    sub = sub.replace([np.inf, -np.inf], np.nan)
    # count NaNs before fill
    nan_count_before = np.isnan(sub.values).sum()
    # fill NaN with 0
    sub = sub.fillna(0.0)
    # clip extreme values
    clipped = sub.clip(lower=-ABS_CLIP_VALUE, upper=ABS_CLIP_VALUE)
    # count how many were clipped (difference)
    clipped_mask = (sub.values != clipped.values)
    clipped_count = int(clipped_mask.sum())
    # assign back
    df.loc[:, numeric_cols] = clipped.astype(float).values
    replaced_total = nan_count_before + clipped_count
    return replaced_total

def preprocess_full(df):
    # detect label column
    label_col = None
    for c in df.columns:
        if str(c).strip().lower() == "label":
            label_col = c; break
    if label_col is None:
        for c in df.columns:
            try:
                sample = df[c].astype(str).str.lower()
                if sample.str.contains("benign").any():
                    label_col = c
                    logger.info(f"Detected label column by content: {label_col}")
                    break
            except Exception:
                continue
    if label_col is None:
        logger.warning("No label column found in full CSV. Will set label = -1 for all rows.")
        df["label"] = -1
        label_col = "label"

    # drop obvious ID / IP and purely timestamp-like columns
    drop_tokens = ["flow id", "id", "src ip", "dst ip", "source ip", "destination ip",
                   "srcip", "dstip", "sourceip", "destinationip", "labelname"]
    to_drop = []
    for c in df.columns:
        lc = str(c).lower()
        if any(tok in lc for tok in drop_tokens):
            to_drop.append(c)
    # drop obvious timestamp columns (except label column)
    for c in df.columns:
        lc = str(c).lower()
        if ("timestamp" in lc or ("time" in lc and c != label_col)):
            to_drop.append(c)
    to_drop = list(set(to_drop))
    logger.info(f"Dropping columns (ids/timestamps): {to_drop[:30]}")
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    # try to convert numeric-like columns; otherwise mark as candidate categorical
    numeric_cols = []
    candidate_categorical = []
    for c in df.columns:
        if c == label_col:
            continue
        # attempt numeric conversion
        converted = pd.to_numeric(df[c], errors="coerce")
        non_na_ratio = converted.notna().mean() if len(converted)>0 else 0
        if non_na_ratio >= 0.95:
            df[c] = converted.fillna(0)
            numeric_cols.append(c)
        else:
            nunique = df[c].nunique(dropna=True)
            if nunique <= 50:
                candidate_categorical.append(c)
            else:
                # high-cardinality non-numeric -> drop (likely free-text)
                logger.debug(f"Dropping high-cardinality non-numeric column '{c}' (unique={nunique})")
                df.drop(columns=[c], inplace=True, errors="ignore")

    # encode the candidate categorical columns (small-cardinality)
    encoders = {}
    for c in list(candidate_categorical):
        if c not in df.columns:
            continue
        try:
            le = LabelEncoder()
            df[c] = df[c].astype(str).fillna("NA")
            df[c] = le.fit_transform(df[c])
            encoders[c] = le
            numeric_cols.append(c)
        except Exception as e:
            logger.warning(f"Failed to encode column {c}: {e}")
            df.drop(columns=[c], inplace=True, errors="ignore")

    # Ensure numeric columns exist; sanitize numeric frame
    replaced = _sanitize_numeric_df(df, numeric_cols)
    if replaced > 0:
        logger.info(f"Sanitized numeric columns: replaced/filled/clipped {replaced} values across {len(numeric_cols)} columns.")

    # create final label numeric column
    try:
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()
        df["label"] = df[label_col].apply(lambda x: 0 if "benign" in str(x) else 1)
    except Exception:
        try:
            df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int)
        except Exception:
            df["label"] = -1

    # drop original label column if it's not exactly 'label'
    if label_col != "label" and label_col in df.columns:
        try:
            df.drop(columns=[label_col], inplace=True, errors=True)
            logger.info(f"Dropped original label column '{label_col}' after mapping to 'label'.")
        except Exception:
            logger.warning(f"Could not drop original label column '{label_col}'.")

    # After conversions: ensure all feature columns (except 'label') are numeric.
    remaining_non_numeric = [c for c in df.columns if c != "label" and not is_numeric_dtype(df[c])]
    for c in remaining_non_numeric:
        try:
            converted = pd.to_numeric(df[c], errors="coerce")
            non_na_ratio = converted.notna().mean() if len(converted)>0 else 0
            if non_na_ratio >= 0.95:
                df[c] = converted.fillna(0)
                logger.info(f"Converted column '{c}' to numeric (fallback).")
                if c not in numeric_cols:
                    numeric_cols.append(c)
            else:
                nunique = df[c].nunique(dropna=True)
                if nunique <= 50:
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c].astype(str).fillna("NA"))
                    encoders[c] = le
                    numeric_cols.append(c)
                    logger.info(f"Label-encoded low-cardinality column '{c}'.")
                else:
                    df.drop(columns=[c], inplace=True, errors=True)
                    logger.info(f"Dropped remaining non-numeric high-card column '{c}'.")
        except Exception as e:
            df.drop(columns=[c], inplace=True, errors=True)
            logger.warning(f"Dropped column '{c}' due to conversion error: {e}")

    # final column ordering: features then label
    if "label" in df.columns:
        cols = [c for c in df.columns if c != "label"] + ["label"]
        df = df[cols]

    # final numeric sanitization pass (clip again)
    numeric_cols_final = [c for c in df.columns if c != "label" and is_numeric_dtype(df[c])]
    replaced_final = _sanitize_numeric_df(df, numeric_cols_final)
    if replaced_final > 0:
        logger.info(f"Final sanitization: replaced/filled/clipped {replaced_final} values across {len(numeric_cols_final)} numeric columns.")

    # save encoders
    os.makedirs(os.path.dirname(ENC_PATH), exist_ok=True)
    joblib.dump(encoders, ENC_PATH)
    logger.info(f"Saved preprocess encoders to {ENC_PATH} (encoded cols: {list(encoders.keys())})")
    return df

def preprocess_mapped(df):
    # existing SDN mapped preprocess (kept minimal) with sanitization
    expected = ["timestamp","controller_id","switch_id","key_id","key_usage_count",
                "flow_request_rate","packet_size","session_duration","auth_status","label"]
    for col in expected:
        if col not in df.columns:
            if col == "label":
                df[col] = -1
            elif col == "auth_status":
                df[col] = 1
            elif col == "timestamp":
                df[col] = pd.Timestamp.now()
            else:
                df[col] = 0
    numcols = ["key_usage_count","flow_request_rate","packet_size","session_duration","auth_status","label"]
    for c in numcols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["session_duration"] = df["session_duration"].replace(0, 0.1)
    # sanitize numeric features
    numeric_cols = [c for c in df.columns if c != "label" and is_numeric_dtype(df[c])]
    _sanitize_numeric_df(df, numeric_cols)
    return df

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    if FULL_FEATURES:
        df = load_raw_full()
        logger.info(f"Loaded original CIC CSV rows: {len(df)}")
        df = preprocess_full(df)
        logger.info(f"Preprocessed full features -> {len(df.columns)} columns, {len(df)} rows")
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved processed file to {PROCESSED_DATA_PATH}")
    else:
        df = load_raw_mapped()
        logger.info(f"Loaded mapped SDN CSV rows: {len(df)}")
        df = preprocess_mapped(df)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved processed file to {PROCESSED_DATA_PATH}")
