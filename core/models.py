"""
POSEIDON ML Models Module
--------------------------
Pipelines for: Random Forest, SVM (RBF), Logistic Regression
Includes: evaluation, bootstrap CI, threshold optimization
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)


# ─────────────────────────────────────────────
# MODEL FACTORIES
# ─────────────────────────────────────────────
def build_rf_pipeline(params=None):
    """Random Forest pipeline with imputer."""
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 10,
            "max_features": 0.5,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(**params)),
    ])


def build_svm_pipeline(params=None):
    """SVM RBF pipeline with imputer + scaler."""
    if params is None:
        params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
        }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(**params)),
    ])


def build_lr_pipeline(params=None):
    """Logistic Regression pipeline with imputer + scaler."""
    if params is None:
        params = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
        }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])


MODEL_REGISTRY = {
    "Random Forest": build_rf_pipeline,
    "SVM (RBF)": build_svm_pipeline,
    "Logistic Regression": build_lr_pipeline,
}


# ─────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────
def best_threshold_f1(y_true, y_prob):
    """Find threshold that maximizes F1 on positive class."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    j = int(np.argmax(f1))
    return float(thr[j]), float(f1[j]), float(prec[:-1][j]), float(rec[:-1][j])


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(y_true, y_prob, threshold=0.5):
    """Comprehensive evaluation with given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "f1_aftershock": report.get("1", {}).get("f1-score", 0),
        "precision_aftershock": report.get("1", {}).get("precision", 0),
        "recall_aftershock": report.get("1", {}).get("recall", 0),
        "confusion_matrix": cm,
        "report_dict": report,
        "roc_curve": (fpr, tpr),
        "pr_curve": (recall_curve, precision_curve),
        "threshold": threshold,
    }


# ─────────────────────────────────────────────
# BOOTSTRAP CI
# ─────────────────────────────────────────────
def bootstrap_ci_auc(y_true, y_score, n_boot=1000, seed=42):
    """Bootstrap 95% CI for ROC-AUC and PR-AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    roc_scores = []
    pr_scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        roc_scores.append(roc_auc_score(yt, ys))
        pr_scores.append(average_precision_score(yt, ys))

    roc_scores = np.array(roc_scores)
    pr_scores = np.array(pr_scores)

    roc_ci = np.percentile(roc_scores, [2.5, 97.5]) if len(roc_scores) > 0 else [np.nan, np.nan]
    pr_ci  = np.percentile(pr_scores, [2.5, 97.5]) if len(pr_scores) > 0 else [np.nan, np.nan]

    return {
        "roc_auc_ci": (float(roc_ci[0]), float(roc_ci[1])),
        "pr_auc_ci": (float(pr_ci[0]), float(pr_ci[1])),
    }


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (RF only)
# ─────────────────────────────────────────────
def get_feature_importance(pipe, feature_names):
    """Extract feature importance from trained RF pipeline."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": imp
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return None


# ─────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────
ABLATION_SETS = {
    "FULL": [
        "mag", "log_dt_big_near", "log_dr_big_near",
        "log_n_prev_30d", "log_n_prev_30d_r50",
        "log_n_big_prev_30d", "max_mag_prev_7d"
    ],
    "NO_TEMPORAL": [
        "mag", "log_dr_big_near",
        "log_n_prev_30d", "log_n_prev_30d_r50",
        "log_n_big_prev_30d", "max_mag_prev_7d"
    ],
    "NO_SPATIAL": [
        "mag", "log_dt_big_near",
        "log_n_prev_30d", "log_n_prev_30d_r50",
        "log_n_big_prev_30d", "max_mag_prev_7d"
    ],
    "NO_DENSITY": [
        "mag", "log_dt_big_near", "log_dr_big_near",
        "max_mag_prev_7d"
    ],
    "NO_SHORT_MAG": [
        "mag", "log_dt_big_near", "log_dr_big_near",
        "log_n_prev_30d", "log_n_prev_30d_r50",
        "log_n_big_prev_30d"
    ],
}

ABLATION_LABELS = {
    "FULL": "Full Model (All Features)",
    "NO_TEMPORAL": "Remove Temporal Decay (Δt)",
    "NO_SPATIAL": "Remove Spatial Decay (Δr)",
    "NO_DENSITY": "Remove Activity/Density Features",
    "NO_SHORT_MAG": "Remove Short-term Max Magnitude",
}
