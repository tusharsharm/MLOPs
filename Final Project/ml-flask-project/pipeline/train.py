"""
Orchestrator: load data → preprocess → train 3 models → save results.
Called by Flask POST /api/pipeline/run  or  python pipeline/train.py
"""

import json
import mlflow
from datetime import datetime
from pathlib import Path

from pipeline.dataset import load_or_generate_dataset
from pipeline.preprocess import preprocess
from pipeline.models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    confusion_matrix_data,
    roc_curve_data,
    feature_importance_data,
)

MLFLOW_DIR = Path(__file__).parent.parent / "ml-pipeline" / "mlruns"
DATA_DIR   = Path(__file__).parent.parent / "ml-pipeline" / "data"


def _write_status(results_dir, status, message, progress=None, step=None, error=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "status":      status,
        "message":     message,
        "progress":    progress,
        "currentStep": step,
        "startedAt":   datetime.now().isoformat(),
        "completedAt": datetime.now().isoformat() if status in ("completed", "failed") else None,
        "error":       error,
    }
    (results_dir / "pipeline_status.json").write_text(json.dumps(data, indent=2))
    print(f"[{status.upper()}] {message}")


def _save_model_files(results_dir, dir_name, display_name, result, feature_names):
    model_dir = results_dir / dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # result.json
    (model_dir / "result.json").write_text(json.dumps({
        "name":         display_name,
        "modelType":    display_name,
        "metrics":      result["metrics"],
        "parameters":   result["params"],
        "trainingTime": result["training_time"],
        "mlflowRunId":  result["run_id"],
        "completedAt":  datetime.now().isoformat(),
    }, indent=2))

    # confusion_matrix.json
    cm = confusion_matrix_data(display_name, result["_y_test"], result["y_pred"])
    (model_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2))

    # roc_curve.json
    roc = roc_curve_data(display_name, result["_y_test"], result["y_prob"])
    (model_dir / "roc_curve.json").write_text(json.dumps(roc, indent=2))

    # feature_importance.json
    fi = feature_importance_data(feature_names, result["importances"])
    (model_dir / "feature_importance.json").write_text(json.dumps(fi, indent=2))

    print(f"  Saved: {display_name}  accuracy={result['metrics']['accuracy']:.4f}")


MODELS = [
    ("logistic_regression", "Logistic Regression", train_logistic_regression),
    ("random_forest",       "Random Forest",       train_random_forest),
    ("xgboost",             "XGBoost",             train_xgboost),
]


def run_pipeline(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use a file URI that is valid on Windows (e.g. file:///D:/path)
    # Use SQLite backend for tracking (mlflow.db) and keep artifacts in mlruns
    # Use the file-based `mlruns` tracking store so the UI (file-store) and
    # training code operate on the same run folders when using the host bind mount.
    # e.g. file:///D:/.../ml-pipeline/mlruns
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    # Older/newer MLflow versions may not expose `set_artifact_uri` as a top-level
    # helper; guard against AttributeError and fall back to default artifact root.
    try:
        mlflow.set_artifact_uri(MLFLOW_DIR.as_uri())
    except AttributeError:
        print('mlflow.set_artifact_uri not available in this MLflow version; using default artifact root')
    mlflow.set_experiment("Global AI vs Human Content 2026")

    _write_status(results_dir, "running", "Loading dataset...", progress=0.05, step="load_data")
    df = load_or_generate_dataset(DATA_DIR)

    _write_status(results_dir, "running", "Preprocessing...", progress=0.10, step="preprocessing")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    # Dataset info
    (results_dir / "dataset_info.json").write_text(json.dumps({
        "totalSamples":       len(df),
        "trainSamples":       len(X_train),
        "testSamples":        len(X_test),
        "numFeatures":        len(feature_names),
        "classDistribution":  df["label"].value_counts().to_dict(),
        "featureNames":       feature_names,
        "missingValues":      int(df.isnull().sum().sum()),
    }, indent=2))

    all_results = {}
    progress_steps = [0.2, 0.5, 0.75]

    for (dir_name, display_name, trainer), progress in zip(MODELS, progress_steps):
        _write_status(results_dir, "running",
                      f"Training {display_name}...",
                      progress=progress, step=dir_name)
        try:
            result = trainer(X_train, X_test, y_train, y_test, feature_names)
            result["_y_test"] = y_test          # attach for metric helpers
            _save_model_files(results_dir, dir_name, display_name, result, feature_names)
            all_results[display_name] = result
        except Exception as exc:
            _write_status(results_dir, "failed",
                          f"{display_name} failed: {exc}", error=str(exc))
            raise

    best = max(all_results, key=lambda n: all_results[n]["metrics"]["accuracy"])
    (results_dir / "summary.json").write_text(json.dumps({
        "models": {
            name: {**data["metrics"], "run_id": data["run_id"]}
            for name, data in all_results.items()
        },
        "bestModel":   best,
        "completedAt": datetime.now().isoformat(),
    }, indent=2))

    # Tag runs in MLflow so the UI can highlight the best run.
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW_DIR.as_uri())
        for name, data in all_results.items():
            run_id = data["run_id"]
            is_best = "true" if name == best else "false"
            client.set_tag(run_id, "is_best", is_best)
    except Exception as exc:
        print(f"Warning: couldn't set best-run tag in MLflow: {exc}")

    _write_status(results_dir, "completed",
                  f"Done! Best model: {best} "
                  f"(accuracy={all_results[best]['metrics']['accuracy']:.4f})",
                  progress=1.0, step="completed")

    print("\n=== FINAL RESULTS ===")
    for name, data in all_results.items():
        m = data["metrics"]
        print(f"  {name}: accuracy={m['accuracy']:.4f}  "
              f"f1={m['f1_score']:.4f}  auc={m['auc_roc']:.4f}")
    print(f"  Best: {best}\n")


# Allow running directly: python pipeline/train.py
if __name__ == "__main__":
    default_results = Path(__file__).parent.parent / "ml-pipeline" / "results"
    run_pipeline(default_results)
