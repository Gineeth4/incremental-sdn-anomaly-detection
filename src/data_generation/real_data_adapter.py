# src/data_generation/real_data_adapter.py
import os, sys, shutil
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger()

CFG_PATH = "configs/config.yaml"
cfg = {}
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, "r") as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

EXTERNAL_SPECIFIED = cfg.get("external_file") or ""
USE_FIRST = cfg.get("adapter", {}).get("use_first_csv_if_not_specified", True)
OUTPUT_PATH = "data/raw/sdn_key_usage.csv"
ORIGINAL_COPY_PATH = "data/raw/original_cic.csv"

def safe_div(a, b, fill=0.0):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.divide(a, b)
        c[~np.isfinite(c)] = fill
    return c

def detect_columns(cols_lower):
    mapping = {}
    for c in ["flow duration", "flowduration", "duration"]:
        if c in cols_lower: mapping["duration"] = cols_lower[c]; break
    for c in ["total fwd packets", "totalfwdpackets", "total fwd packets", "total forward packets"]:
        if c in cols_lower: mapping["tot_fwd_pkts"] = cols_lower[c]; break
    for c in ["total bwd packets", "totalbwdpackets", "total backward packets", "total bwd packets"]:
        if c in cols_lower: mapping["tot_bwd_pkts"] = cols_lower[c]; break
    for c in ["total length of fwd packets", "totlen fwd pkts", "total_length_fwd", "total length fwd"]:
        if c in cols_lower: mapping["tot_len_fwd"] = cols_lower[c]; break
    for c in ["total length of bwd packets", "totlen bwd pkts", "total_length_bwd", "total length bwd"]:
        if c in cols_lower: mapping["tot_len_bwd"] = cols_lower[c]; break
    for c in ["flow packets/s", "flow packets per second", "flow packets", "flow packets/s"]:
        if c in cols_lower: mapping["flow_pps"] = cols_lower[c]; break
    for c in ["average packet size", "avg packet size", "averagepacket"]:
        if c in cols_lower: mapping["avg_pkt_size"] = cols_lower[c]; break
    for c in ["destination port", "dst port", "dport", "dstport"]:
        if c in cols_lower: mapping["dst_port"] = cols_lower[c]; break
    for c in ["label", "class", "attack", "activity"]:
        if c in cols_lower: mapping["label"] = cols_lower[c]; break
    return mapping

def convert_one(df):
    cols_lower = {col.lower(): col for col in df.columns}
    mapping = detect_columns(cols_lower)
    # if adapter didn't detect a 'label' column by name, try to find any column whose values include 'benign'
    if "label" not in mapping:
        for col in df.columns:
            try:
                sample = df[col].astype(str).str.lower()
                if sample.str.contains("benign").any():
                    mapping["label"] = col
                    logger.info(f"Detected label column by content: {col}")
                    break
            except Exception:
                continue

    mapped = pd.DataFrame()

    # timestamp
    ts_col = None
    for c in ["timestamp", "time", "starttime", "flow start", "start time"]:
        if c in cols_lower:
            ts_col = cols_lower[c]; break
    if ts_col:
        mapped["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce").fillna(method="ffill").fillna(datetime.now())
    else:
        mapped["timestamp"] = pd.date_range(start=datetime.now(), periods=len(df), freq="s")

    mapped["controller_id"] = "C_main"
    if "dst_port" in mapping:
        mapped["switch_id"] = df[mapping["dst_port"]].astype(str).apply(lambda s: "S_" + s)
        mapped["key_id"] = df[mapping["dst_port"]].astype(str).apply(lambda s: "K_" + s)
    else:
        mapped["switch_id"] = ["S_" + str(i) for i in df.index]
        mapped["key_id"] = ["K_" + str(i) for i in df.index]

    # key usage count
    if "tot_fwd_pkts" in mapping and "tot_bwd_pkts" in mapping:
        fwd = pd.to_numeric(df[mapping["tot_fwd_pkts"]], errors="coerce").fillna(0)
        bwd = pd.to_numeric(df[mapping["tot_bwd_pkts"]], errors="coerce").fillna(0)
        mapped["key_usage_count"] = (fwd + bwd).astype(int)
    else:
        pkt_col = None
        for c in df.columns:
            if "packet" in c.lower() and ("total" in c.lower() or "count" in c.lower()):
                pkt_col = c; break
        if pkt_col:
            mapped["key_usage_count"] = pd.to_numeric(df[pkt_col], errors="coerce").fillna(0).astype(int)
        else:
            mapped["key_usage_count"] = 0

    # duration
    if "duration" in mapping:
        dur = pd.to_numeric(df[mapping["duration"]], errors="coerce").fillna(0)
        med = dur.median() if len(dur) > 0 else 1
        if med > 1e6:
            dur = dur / 1e6
        elif med > 1e3:
            dur = dur / 1e3
        mapped["session_duration"] = dur.clip(lower=0.1)
    else:
        mapped["session_duration"] = 1.0

    # packet_size
    if "avg_pkt_size" in mapping:
        mapped["packet_size"] = pd.to_numeric(df[mapping["avg_pkt_size"]], errors="coerce").fillna(0)
    else:
        if "tot_len_fwd" in mapping and "tot_len_bwd" in mapping:
            totlen = pd.to_numeric(df[mapping["tot_len_fwd"]], errors="coerce").fillna(0) + \
                     pd.to_numeric(df[mapping["tot_len_bwd"]], errors="coerce").fillna(0)
            mapped["packet_size"] = safe_div(totlen, mapped["key_usage_count"], fill=0.0)
        else:
            mapped["packet_size"] = 0.0

    # flow rate
    if "flow_pps" in mapping:
        mapped["flow_request_rate"] = pd.to_numeric(df[mapping["flow_pps"]], errors="coerce").fillna(0)
    else:
        mapped["flow_request_rate"] = safe_div(mapped["key_usage_count"], mapped["session_duration"], fill=0.0)

    # Label mapping: robust
    if "label" in mapping:
        try:
            lab = df[mapping["label"]].astype(str).str.strip().str.lower()
            mapped["label"] = lab.apply(lambda x: 0 if "benign" in x else 1)
            logger.info(f"Mapped labels using column: {mapping['label']}")
        except Exception:
            mapped["label"] = -1
    else:
        mapped["label"] = -1

    mapped["auth_status"] = 1

    mapped["key_usage_count"] = mapped["key_usage_count"].fillna(0).astype(int)
    mapped["flow_request_rate"] = mapped["flow_request_rate"].astype(float)
    mapped["packet_size"] = mapped["packet_size"].astype(float)
    mapped["session_duration"] = mapped["session_duration"].astype(float)
    mapped["auth_status"] = mapped["auth_status"].astype(int)
    mapped["label"] = mapped["label"].astype(int)

    return mapped

def convert():
    chosen = None
    if EXTERNAL_SPECIFIED:
        if os.path.exists(EXTERNAL_SPECIFIED):
            chosen = EXTERNAL_SPECIFIED
        elif os.path.exists(os.path.join("data/external", EXTERNAL_SPECIFIED)):
            chosen = os.path.join("data/external", EXTERNAL_SPECIFIED)
        else:
            logger.error(f"external_file in config not found: {EXTERNAL_SPECIFIED}")
            raise FileNotFoundError(EXTERNAL_SPECIFIED)
    else:
        csvs = glob.glob("data/external/*.csv")
        if not csvs:
            logger.error("No CSVs found under data/external/. Place CIC CSV(s) there.")
            raise FileNotFoundError("data/external/*.csv")
        chosen = csvs[0] if USE_FIRST else csvs[0]

    logger.info(f"Converting CSV: {chosen}")
    df = pd.read_csv(chosen, low_memory=False)
    mapped = convert_one(df)

    # save mapped sdn-style file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    mapped.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved mapped file to {OUTPUT_PATH} ({len(mapped)} rows)")

    # also copy original CSV for full-feature training
    try:
        os.makedirs(os.path.dirname(ORIGINAL_COPY_PATH), exist_ok=True)
        shutil.copy(chosen, ORIGINAL_COPY_PATH)
        logger.info(f"Copied original CSV to {ORIGINAL_COPY_PATH}")
    except Exception as e:
        logger.warning(f"Could not copy original CSV: {e}")

    return mapped

if __name__ == "__main__":
    convert()
