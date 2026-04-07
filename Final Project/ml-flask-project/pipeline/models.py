"""
Train three classifiers and log everything with MLflow.
"""

import time
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier


# ─── Metrics helper ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    prob_col = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1_score":  float(f1_score(y_true, y_pred, average="weighted")),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, prob_col)),
    }


def confusion_matrix_data(model_name, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "modelName":     model_name,
        "labels":        ["Human", "AI"],
        "matrix":        cm.tolist(),
        "truePositive":  int(tp),
        "trueNegative":  int(tn),
        "falsePositive": int(fp),
        "falseNegative": int(fn),
    }


def roc_curve_data(model_name, y_true, y_prob, sample_size=200) -> dict:
    prob_col = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    fpr, tpr, thr = roc_curve(y_true, prob_col)
    auc = float(roc_auc_score(y_true, prob_col))

    # Downsample to keep JSON compact
    total = len(fpr)
    idx = [int(i * total / sample_size) for i in range(sample_size)]
    return {
        "modelName":  model_name,
        "fpr":        [fpr[i] for i in idx],
        "tpr":        [tpr[i] for i in idx],
        "thresholds": [float(thr[min(i, len(thr)-1)]) for i in idx],
        "auc":        auc,
    }


def feature_importance_data(feature_names, importances) -> list:
    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return [
        {"feature": feat, "importance": float(imp), "rank": rank}
        for rank, (feat, imp) in enumerate(pairs, start=1)
    ]


# ─── Individual trainers ─────────────────────────────────────────────────────

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Logistic Regression with StandardScaler."""
    params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    with mlflow.start_run(run_name="Logistic Regression") as run:
        mlflow.log_params(params)
        t0 = time.time()
        model = LogisticRegression(**params, random_state=42)
        model.fit(Xtr, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics({k: round(v, 6) for k, v in metrics.items()})
        mlflow.log_metric("training_time_seconds", round(elapsed, 3))
        mlflow.sklearn.log_model(model, artifact_path="model")

        importances = np.abs(model.coef_[0])

        return {
            "run_id":          run.info.run_id,
            "metrics":         metrics,
            "params":          params,
            "training_time":   elapsed,
            "y_pred":          y_pred,
            "y_prob":          y_prob,
            "importances":     importances,
        }


def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Random Forest classifier."""
    params = {
        "n_estimators": 100,
        "max_depth":     15,
        "min_samples_split": 5,
        "min_samples_leaf":  2,
        "random_state":  42,
    }

    with mlflow.start_run(run_name="Random Forest") as run:
        mlflow.log_params(params)
        t0 = time.time()
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics({k: round(v, 6) for k, v in metrics.items()})
        mlflow.log_metric("training_time_seconds", round(elapsed, 3))
        mlflow.sklearn.log_model(model, artifact_path="model")

        return {
            "run_id":          run.info.run_id,
            "metrics":         metrics,
            "params":          params,
            "training_time":   elapsed,
            "y_pred":          y_pred,
            "y_prob":          y_prob,
            "importances":     model.feature_importances_,
        }


def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """XGBoost classifier."""
    params = {
        "n_estimators":    200,
        "max_depth":       6,
        "learning_rate":   0.1,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "eval_metric":     "logloss",
        "random_state":    42,
        "verbosity":       0,
    }

    with mlflow.start_run(run_name="XGBoost") as run:
        log_params = {k: v for k, v in params.items() if k != "verbosity"}
        mlflow.log_params(log_params)
        t0 = time.time()
        model = XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics({k: round(v, 6) for k, v in metrics.items()})
        mlflow.log_metric("training_time_seconds", round(elapsed, 3))
        mlflow.xgboost.log_model(model, artifact_path="model")

        return {
            "run_id":          run.info.run_id,
            "metrics":         metrics,
            "params":          log_params,
            "training_time":   elapsed,
            "y_pred":          y_pred,
            "y_prob":          y_prob,
            "importances":     model.feature_importances_,
        }
