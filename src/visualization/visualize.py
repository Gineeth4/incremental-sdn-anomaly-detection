# src/visualization/visualize.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from src.utils.logger import get_logger

logger = get_logger()

PROCESSED_DATA_PATH = "data/processed/sdn_key_usage_processed.csv"
HOLDOUT_PATH = "experiments/results/holdout_test.csv"
ANOMALY_RESULTS_PATH = "experiments/results/detected_anomalies.csv"
MODEL_PATH = "experiments/results/base_model.pkl"
META_PATH = "experiments/results/model_meta.json"
FIG_DIR = "experiments/results/figures"

os.makedirs(FIG_DIR, exist_ok=True)


def safe_load_model_and_holdout():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Cannot plot holdout ROC/PR/importance.")
        return None, None, None
    model = joblib.load(MODEL_PATH)
    hold = None
    if os.path.exists(HOLDOUT_PATH):
        hold = pd.read_csv(HOLDOUT_PATH)
    else:
        logger.warning("Holdout file missing; some plots will be skipped.")
    # features list if available
    features = None
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                meta = json.load(f)
                features = meta.get("features")
        except Exception:
            features = None
    return model, hold, features


def plot_holdout_roc_pr(model, hold):
    """Plot ROC and PR using holdout (saves two files)."""
    if hold is None:
        logger.warning("No holdout data for ROC/PR.")
        return
    if "label" not in hold.columns:
        logger.warning("Holdout exists but has no 'label' column; skipping ROC/PR.")
        return
    X_hold = hold.drop(columns=["label", "timestamp"], errors="ignore")
    y_hold = hold["label"].astype(int)
    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba; ROC/PR skipped.")
        return

    probs = model.predict_proba(X_hold)[:, 1]
    # ROC
    fpr, tpr, _ = roc_curve(y_hold, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Holdout ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "holdout_roc.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_hold, probs)
    pr_auc = np.trapz(prec[::-1], rec[::-1]) if len(rec) > 1 else 0.0
    plt.figure(figsize=(7,6))
    plt.plot(rec, prec, lw=2, label=f"PR (AUC ≈ {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Holdout Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "holdout_pr.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")


def plot_confusion_matrix_holdout(model, hold):
    """Compute confusion matrix on holdout and save heatmap."""
    if hold is None or "label" not in hold.columns:
        logger.warning("Holdout missing for confusion matrix.")
        return
    X_hold = hold.drop(columns=["label", "timestamp"], errors="ignore")
    y_hold = hold["label"].astype(int)
    preds = model.predict(X_hold)
    cm = confusion_matrix(y_hold, preds)
    plt.figure(figsize=(5,4))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    classes = ["Benign (0)", "Attack (1)"]
    plt.xticks([0,1], classes, rotation=45)
    plt.yticks([0,1], classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color="black", fontsize=10)
    plt.title("Confusion Matrix (Holdout)")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "holdout_confusion_matrix.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")


def plot_feature_importance_top20(model, features):
    """Plot top 20 feature importances (if available)."""
    if features is None:
        logger.warning("No feature list in meta; skipping feature importance.")
        return
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_; skipping.")
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    top_feats = [features[i] for i in idx]
    top_vals = importances[idx]
    plt.figure(figsize=(10,6))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_vals[::-1])
    plt.yticks(y_pos, top_feats[::-1])
    plt.xlabel("Feature importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "feature_importance_top20.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")


def plot_calibration_curve(model, hold, n_bins=10):
    """Reliability diagram + histogram of probabilities."""
    if hold is None or "label" not in hold.columns:
        logger.warning("Holdout missing for calibration plot.")
        return
    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba; skipping calibration plot.")
        return
    X_hold = hold.drop(columns=["label", "timestamp"], errors="ignore")
    y_hold = hold["label"].astype(int)
    probs = model.predict_proba(X_hold)[:, 1]
    prob_true, prob_pred = calibration_curve(y_hold, probs, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(7,6))
    plt.plot(prob_pred, prob_true, marker='o', label="Calibration")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (Reliability) Curve")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "calibration_curve.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")

    # probability histogram
    plt.figure(figsize=(7,3))
    plt.hist(probs, bins=30, density=False)
    plt.xlabel("Predicted probability")
    plt.title("Predicted probability histogram (holdout)")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "probability_histogram.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")


def plot_score_distribution_by_true_class(model, hold):
    """Show predicted-probability distribution per true class (violin + box overlay)."""
    if hold is None or "label" not in hold.columns:
        logger.warning("Holdout missing for score distribution by class.")
        return
    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba; skipping score-distribution plot.")
        return
    X_hold = hold.drop(columns=["label", "timestamp"], errors="ignore")
    y_hold = hold["label"].astype(int)
    probs = model.predict_proba(X_hold)[:, 1]
    df = pd.DataFrame({"true": y_hold, "proba": probs})
    # create side-by-side histograms and violin-like density
    plt.figure(figsize=(8,5))
    bins = np.linspace(0, 1, 50)
    plt.hist(df[df["true"] == 0]["proba"], bins=bins, alpha=0.6, label="Benign (0)", density=True)
    plt.hist(df[df["true"] == 1]["proba"], bins=bins, alpha=0.6, label="Attack (1)", density=True)
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Predicted probability distribution by true class")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "score_distribution_by_true_class.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")


def plot_pca_true_vs_pred(processed_path, anomaly_results_path, sample_size=20000):
    """PCA projection of processed features colored by true label and marker for predicted label."""
    if not os.path.exists(processed_path) or not os.path.exists(anomaly_results_path):
        logger.warning("Processed data or anomaly results missing for PCA plot.")
        return
    proc = pd.read_csv(processed_path)
    res = pd.read_csv(anomaly_results_path)
    # align lengths if needed
    n = min(len(proc), len(res))
    proc = proc.iloc[:n].reset_index(drop=True)
    res = res.iloc[:n].reset_index(drop=True)
    X = proc.drop(columns=["label", "timestamp"], errors="ignore").fillna(0)
    # sample to keep plot readable
    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X = X.iloc[idx]
        y_true = proc["label"].iloc[idx].astype(int)
        y_pred = res["predicted_label"].iloc[idx].astype(int) if "predicted_label" in res.columns else np.zeros(len(idx), dtype=int)
    else:
        y_true = proc["label"].astype(int)
        y_pred = res["predicted_label"].astype(int) if "predicted_label" in res.columns else np.zeros(len(X), dtype=int)

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X.values)
    dfp = pd.DataFrame({"pc1": comps[:, 0], "pc2": comps[:, 1], "true": y_true.values, "pred": y_pred.values})
    plt.figure(figsize=(8,6))
    # color by true label, edgecolor/marker style by predicted label
    cmap = {0: "tab:green", 1: "tab:red"}
    markers = {0: "o", 1: "X"}
    for t in sorted(dfp["true"].unique()):
        subset = dfp[dfp["true"] == t]
        for pval in sorted(subset["pred"].unique()):
            s2 = subset[subset["pred"] == pval]
            plt.scatter(s2["pc1"], s2["pc2"], c=cmap.get(t, "gray"), marker=markers.get(pval, "o"),
                        label=f"true={t}, pred={pval}", s=10, alpha=0.6)
    plt.title("PCA projection (color=true label, marker=predicted)")
    plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "pca_true_vs_pred.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")


def plot_cumulative_gain(model, hold):
    """Plot cumulative gains (lift) curve on holdout."""
    if hold is None or "label" not in hold.columns:
        logger.warning("Holdout missing for cumulative gains plot.")
        return
    if not hasattr(model, "predict_proba"):
        logger.warning("Model has no predict_proba; skipping cumulative gains.")
        return
    X_hold = hold.drop(columns=["label", "timestamp"], errors="ignore")
    y_hold = hold["label"].astype(int)
    probs = model.predict_proba(X_hold)[:, 1]
    df = pd.DataFrame({"y": y_hold, "proba": probs})
    df = df.sort_values("proba", ascending=False).reset_index(drop=True)
    df["cum_positive"] = df["y"].cumsum()
    df["cum_pct_positive"] = df["cum_positive"] / df["y"].sum()
    df["pct_population"] = (np.arange(1, len(df) + 1) / len(df))
    plt.figure(figsize=(7,6))
    plt.plot(df["pct_population"], df["cum_pct_positive"], label="Model")
    plt.plot([0,1], [0,1], '--', color='gray', label="Random")
    plt.xlabel("Proportion of Population (sorted by score)")
    plt.ylabel("Cumulative Proportion of Positives (Recall)")
    plt.title("Cumulative Gains (Lift) Curve")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "cumulative_gains.png"); plt.savefig(p); plt.close(); logger.info(f"Saved {p}")


if __name__ == "__main__":
    model, hold, features = safe_load_model_and_holdout()
    # produce the 7 visuals
    plot_holdout_roc_pr(model, hold)
    plot_confusion_matrix_holdout(model, hold)
    plot_feature_importance_top20(model, features)
    plot_calibration_curve(model, hold)
    plot_score_distribution_by_true_class(model, hold)
    plot_pca_true_vs_pred(PROCESSED_DATA_PATH, ANOMALY_RESULTS_PATH)
    plot_cumulative_gain(model, hold)
    logger.info("All requested visuals generated and saved to: " + FIG_DIR)
