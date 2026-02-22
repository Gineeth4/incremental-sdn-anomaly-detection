# src/models/anomaly_detector.py
import numpy as np

class AnomalyDetector:
    def __init__(self, model, threshold=0.5):
        """
        model: sklearn-like model
        threshold: for probability-based models; for other models it's a fallback
        """
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        mdl = self.model
        # probabilistic classifier (binary proba)
        if hasattr(mdl, "predict_proba"):
            probs = mdl.predict_proba(X)[:, 1]
            preds = (probs >= self.threshold).astype(int)
            return preds, probs

        # IsolationForest: predict returns -1 for anomaly
        cls_name = mdl.__class__.__name__.lower()
        if "isolationforest" in cls_name:
            raw = mdl.predict(X)  # -1 anomaly, 1 normal
            preds = (raw == -1).astype(int)
            # use negative score_samples so higher -> more anomalous
            scores = None
            if hasattr(mdl, "score_samples"):
                scores = -mdl.score_samples(X)
            return preds, scores

        # If model has decision_function -> use it (invert so higher => more anomalous)
        if hasattr(mdl, "decision_function"):
            df = mdl.decision_function(X)
            # many models have larger=more positive for normal; we invert to make higher=more anomalous
            scores = -df
            preds = (scores >= self.threshold).astype(int)
            return preds, scores

        # fallback: model.predict
        preds = mdl.predict(X)
        preds_arr = np.array(preds)
        unique = set(np.unique(preds_arr))
        if unique <= {-1, 1}:
            # -1 anomaly
            preds_final = (preds_arr == -1).astype(int)
            return preds_final, None
        # else assume binary 0/1 already
        return preds_arr.astype(int), None

    def update(self, X, y):
        """
        Incremental update: call partial_fit if available; otherwise fit() (retrain on provided batch)
        For unsupervised models like IsolationForest, fit() will retrain on new batch.
        """
        mdl = self.model
        if hasattr(mdl, "partial_fit"):
            if len(y) > 0:
                mdl.partial_fit(X, y)
        else:
            # fallback: fit on provided X (for unsupervised adaptation)
            try:
                mdl.fit(X)
            except Exception:
                # If this fails, ignore update
                pass
