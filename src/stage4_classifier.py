"""
Stage 4 – Anomaly Scoring and Decision.

Aggregates s1, s2, s3 into a final backdoor probability via
logistic regression (or hand-tuned weights when training data is small).

Also handles:
  - Training on BackdoorLLM labeled pairs
  - Threshold selection (FPR@95%TPR)
  - AUROC / accuracy reporting
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class BackdoorClassifier:
    """
    Logistic classifier over (s1, s2, s3) feature vector.
    Falls back to hand-tuned weights when fewer than 4 training samples.
    """

    # Hand-tuned weights: [s1, s2, s3, kurtosis, norm_trig_ratio, norm_style_ratio]
    _DEFAULT_WEIGHTS = np.array([0.5, 0.3, 0.4, 0.1, 0.3, 0.2])
    _DEFAULT_BIAS = -0.5

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(random_state=random_seed, max_iter=1000, C=1.0)
        self._trained = False
        self._threshold = 0.5

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BackdoorClassifier":
        """
        features: (n_models, 3)  — columns are [s1, s2, s3]
        labels:   (n_models,)    — 0=clean, 1=backdoored
        """
        if len(features) < 4 or len(np.unique(labels)) < 2:
            logger.warning("Too few labeled samples; using hand-tuned weights.")
            self._trained = False
            return self

        X = self.scaler.fit_transform(features)
        self.clf.fit(X, labels)
        self._trained = True
        logger.info(f"Classifier trained on {len(features)} samples.")
        logger.info(f"  Coefficients: {self.clf.coef_[0]}")
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Returns P(backdoored) for each row in features."""
        if self._trained:
            X = self.scaler.transform(features)
            return self.clf.predict_proba(X)[:, 1]
        # Hand-tuned fallback
        scores = features @ self._DEFAULT_WEIGHTS + self._DEFAULT_BIAS
        return np.array([sigmoid(s) for s in scores])

    def predict(self, features: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(features)
        return (proba >= self._threshold).astype(int)

    def set_threshold_at_tpr(self, features: np.ndarray, labels: np.ndarray, tpr: float = 0.95):
        """Tune decision threshold to achieve target TPR."""
        proba = self.predict_proba(features)
        fpr_arr, tpr_arr, thresholds = roc_curve(labels, proba)
        idx = np.searchsorted(tpr_arr, tpr)
        if idx < len(thresholds):
            self._threshold = float(thresholds[idx])
        logger.info(f"Threshold set to {self._threshold:.4f} for TPR={tpr}")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "scaler": self.scaler,
                         "trained": self._trained, "threshold": self._threshold}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.clf = d["clf"]
        self.scaler = d["scaler"]
        self._trained = d["trained"]
        self._threshold = d["threshold"]


def evaluate_classifier(
    classifier: BackdoorClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute AUROC, FPR@95%TPR, accuracy, F1."""
    proba = classifier.predict_proba(features)
    preds = classifier.predict(features)

    auroc = roc_auc_score(labels, proba) if len(np.unique(labels)) > 1 else float("nan")
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    # FPR @ 95% TPR
    fpr_at_95 = float("nan")
    if len(np.unique(labels)) > 1:
        fpr_arr, tpr_arr, _ = roc_curve(labels, proba)
        idx = np.searchsorted(tpr_arr, 0.95)
        if idx < len(fpr_arr):
            fpr_at_95 = float(fpr_arr[idx])

    results = {
        "auroc": float(auroc),
        "fpr_at_95tpr": float(fpr_at_95),
        "accuracy": float(acc),
        "f1": float(f1),
        "n_samples": len(labels),
        "n_backdoored": int(labels.sum()),
        "n_clean": int((labels == 0).sum()),
    }
    logger.info(
        f"Evaluation | AUROC={auroc:.4f} FPR@95TPR={fpr_at_95:.4f} Acc={acc:.4f} F1={f1:.4f}"
    )
    return results


def cross_validate(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """K-fold cross-validation for AUROC estimation."""
    if len(np.unique(labels)) < 2:
        logger.warning("Cannot cross-validate: only one class present.")
        return {}

    # Need at least n_folds samples of each class so every fold has both classes
    min_class_count = int(np.min(np.bincount(labels)))
    if min_class_count < 2:
        logger.warning(f"Cannot cross-validate: minority class has only {min_class_count} sample(s).")
        return {}
    n_folds = min(n_folds, min_class_count)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    clf = LogisticRegression(random_state=random_seed, max_iter=1000)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    try:
        proba = cross_val_predict(clf, X, labels, cv=skf, method="predict_proba")[:, 1]
    except ValueError as e:
        logger.warning(f"Cross-validation failed: {e}")
        return {}

    auroc = roc_auc_score(labels, proba)
    fpr_arr, tpr_arr, _ = roc_curve(labels, proba)
    idx = np.searchsorted(tpr_arr, 0.95)
    fpr_at_95 = float(fpr_arr[idx]) if idx < len(fpr_arr) else float("nan")

    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)

    result = {
        "cv_auroc": float(auroc),
        "cv_fpr_at_95tpr": float(fpr_at_95),
        "cv_accuracy": float(acc),
        "n_folds": n_folds,
    }
    logger.info(f"CV ({n_folds}-fold) | AUROC={auroc:.4f} FPR@95TPR={fpr_at_95:.4f} Acc={acc:.4f}")
    return result


def aggregate_scores(
    model_results: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert list of per-model result dicts into feature matrix and label vector.
    Features: [s1, s2, s3, kurtosis, norm_trig_ratio, norm_style_ratio]
    Falls back to [s1, s2, s3] if diagnostics unavailable.
    """
    rows = []
    for r in model_results:
        d1 = r.get("diagnostics", {}).get("stage1", {})
        row = [
            r["s1"],
            r["s2"],
            r["s3"],
            float(d1.get("kurtosis", 0.0)),
            float(d1.get("norm_trig_ratio", d1.get("trig_ratio", 1.0))),
            float(d1.get("norm_style_ratio", d1.get("style_ratio", 1.0))),
        ]
        rows.append(row)
    features = np.array(rows)
    labels = np.array([r["label"] for r in model_results])
    return features, labels
