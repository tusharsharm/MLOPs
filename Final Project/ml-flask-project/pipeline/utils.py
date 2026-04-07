"""
Read result JSON files produced by the pipeline and shape them for the API.
"""

import json
from pathlib import Path
from typing import Optional

MODEL_DIRS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
MODEL_ORDER = list(MODEL_DIRS.keys())


def _read(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


# ─── Pipeline status ──────────────────────────────────────────────────────────

def get_pipeline_status(results_dir: Path) -> dict:
    data = _read(results_dir / "pipeline_status.json")
    if not data:
        return {
            "status":      "idle",
            "message":     "No pipeline run yet. POST /api/pipeline/run to start.",
            "progress":    None,
            "currentStep": None,
            "startedAt":   None,
            "completedAt": None,
            "error":       None,
        }
    return data


# ─── Dataset info ─────────────────────────────────────────────────────────────

def get_dataset_info(results_dir: Path) -> Optional[dict]:
    return _read(results_dir / "dataset_info.json")


# ─── Summary ─────────────────────────────────────────────────────────────────

def get_summary(results_dir: Path) -> dict:
    data = _read(results_dir / "summary.json")
    return data or {}


# ─── Model results ────────────────────────────────────────────────────────────

def get_all_model_results(results_dir: Path) -> list:
    summary = get_summary(results_dir)
    best_name = summary.get("bestModel")
    results = []

    for name in MODEL_ORDER:
        dir_name = MODEL_DIRS[name]
        data = _read(results_dir / dir_name / "result.json")
        if data:
            m = data.get("metrics", {})
            results.append({
                "name":         name,
                "modelType":    name,
                "accuracy":     m.get("accuracy"),
                "f1Score":      m.get("f1_score"),
                "precision":    m.get("precision"),
                "recall":       m.get("recall"),
                "aucRoc":       m.get("auc_roc"),
                "trainingTime": data.get("trainingTime"),
                "mlflowRunId":  data.get("mlflowRunId"),
                "parameters":   data.get("parameters", {}),
                "isBest":       (name == best_name),
            })

    return results


def get_model_result(results_dir: Path, model_name: str) -> Optional[dict]:
    dir_name = MODEL_DIRS.get(model_name)
    if not dir_name:
        return None
    return _read(results_dir / dir_name / "result.json")


# ─── Per-model artifacts ──────────────────────────────────────────────────────

def get_confusion_matrix(results_dir: Path, model_name: str) -> Optional[dict]:
    dir_name = MODEL_DIRS.get(model_name)
    if not dir_name:
        return None
    return _read(results_dir / dir_name / "confusion_matrix.json")


def get_roc_curve(results_dir: Path, model_name: str) -> Optional[dict]:
    dir_name = MODEL_DIRS.get(model_name)
    if not dir_name:
        return None
    return _read(results_dir / dir_name / "roc_curve.json")


def get_feature_importance(results_dir: Path, model_name: str, top_n: int = 15) -> list:
    dir_name = MODEL_DIRS.get(model_name)
    if not dir_name:
        return []
    data = _read(results_dir / dir_name / "feature_importance.json")
    if not data:
        return []
    return data[:top_n]
