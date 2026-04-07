"""
Flask API for Global AI vs Human Content ML Experiment Tracker
Endpoints expose training pipeline control, model results, and MLflow tracking.
Run: python app.py
"""

from flask import Flask, jsonify, request, redirect
from flask_cors import CORS
import os
import json
import subprocess
import threading
from pathlib import Path

from pipeline.train import run_pipeline
from pipeline.utils import (
    get_pipeline_status,
    get_dataset_info,
    get_all_model_results,
    get_model_result,
    get_confusion_matrix,
    get_roc_curve,
    get_feature_importance,
    get_summary,
)

app = Flask(__name__)
CORS(app)

RESULTS_DIR = Path(__file__).parent / "ml-pipeline" / "results"


# Root: user-friendly landing page (avoid 404 at '/').
@app.route("/", methods=["GET"])
def index():
    return (
        "<html><head><title>ML Flask Project</title></head><body>"
        "<h1>ML Flask Project API</h1>"
        "<p>API is available under <a href=\"/api/health\">/api/health</a></p>"
        "<ul>"
        "<li><a href=\"/api/health\">/api/health</a></li>"
        "<li><a href=\"/api/experiments\">/api/experiments</a></li>"
        "<li><a href=\"/api/models\">/api/models</a></li>"
        "</ul>"
        "</body></html>"
    )


# ─── Health ──────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ─── Pipeline ────────────────────────────────────────────────────────────────

_pipeline_running = False
_pipeline_lock = threading.Lock()


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    return jsonify(get_pipeline_status(RESULTS_DIR))


@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline_endpoint():
    global _pipeline_running

    with _pipeline_lock:
        if _pipeline_running:
            return jsonify({
                "status": "running",
                "message": "Pipeline is already running...",
            }), 200

        _pipeline_running = True

    def _run():
        global _pipeline_running
        try:
            run_pipeline(RESULTS_DIR)
        finally:
            with _pipeline_lock:
                _pipeline_running = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({
        "status": "running",
        "message": "Pipeline started! Training 3 models: Logistic Regression, Random Forest, XGBoost.",
    }), 200


@app.route("/api/pipeline/dataset-info", methods=["GET"])
def dataset_info():
    info = get_dataset_info(RESULTS_DIR)
    if not info:
        return jsonify({"error": "No dataset info yet. Run the pipeline first."}), 404
    return jsonify(info)


# ─── Experiments ─────────────────────────────────────────────────────────────

@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    models = get_all_model_results(RESULTS_DIR)
    status = get_pipeline_status(RESULTS_DIR)
    experiments = []
    for i, m in enumerate(models, start=1):
        experiments.append({
            "id": i,
            "name": m["name"],
            "modelType": m["modelType"],
            "status": status.get("status", "idle"),
            "accuracy": m.get("accuracy"),
            "f1Score": m.get("f1Score"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "aucRoc": m.get("aucRoc"),
            "trainingTime": m.get("trainingTime"),
            "mlflowRunId": m.get("mlflowRunId"),
        })
    return jsonify(experiments)


@app.route("/api/experiments/summary", methods=["GET"])
def experiments_summary():
    summary = get_summary(RESULTS_DIR)
    return jsonify(summary)


@app.route("/api/experiments/<int:experiment_id>", methods=["GET"])
def get_experiment(experiment_id):
    models = get_all_model_results(RESULTS_DIR)
    if experiment_id < 1 or experiment_id > len(models):
        return jsonify({"error": "Experiment not found"}), 404
    m = models[experiment_id - 1]
    return jsonify({**m, "id": experiment_id})


# ─── Models ──────────────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def list_models():
    models = get_all_model_results(RESULTS_DIR)
    summary = get_summary(RESULTS_DIR)
    best = summary.get("bestModel")
    for i, m in enumerate(models, start=1):
        m["id"] = i
        m["isBest"] = (m["name"] == best)
    return jsonify(models)


@app.route("/api/models/best", methods=["GET"])
def best_model():
    models = get_all_model_results(RESULTS_DIR)
    summary = get_summary(RESULTS_DIR)
    best_name = summary.get("bestModel")
    for i, m in enumerate(models, start=1):
        if m["name"] == best_name:
            m["id"] = i
            m["isBest"] = True
            return jsonify(m)
    if models:
        models[0]["id"] = 1
        models[0]["isBest"] = True
        return jsonify(models[0])
    return jsonify({"error": "No trained models found"}), 404


@app.route("/api/models/<int:model_id>/confusion-matrix", methods=["GET"])
def confusion_matrix_endpoint(model_id):
    models = get_all_model_results(RESULTS_DIR)
    if model_id < 1 or model_id > len(models):
        return jsonify({"error": "Model not found"}), 404
    model_name = models[model_id - 1]["name"]
    data = get_confusion_matrix(RESULTS_DIR, model_name)
    if not data:
        return jsonify({"error": "Confusion matrix not found. Run the pipeline first."}), 404
    data["modelId"] = model_id
    return jsonify(data)


@app.route("/api/models/<int:model_id>/roc-curve", methods=["GET"])
def roc_curve_endpoint(model_id):
    models = get_all_model_results(RESULTS_DIR)
    if model_id < 1 or model_id > len(models):
        return jsonify({"error": "Model not found"}), 404
    model_name = models[model_id - 1]["name"]
    data = get_roc_curve(RESULTS_DIR, model_name)
    if not data:
        return jsonify({"error": "ROC curve not found. Run the pipeline first."}), 404
    data["modelId"] = model_id
    return jsonify(data)


@app.route("/api/models/<int:model_id>/feature-importance", methods=["GET"])
def feature_importance_endpoint(model_id):
    models = get_all_model_results(RESULTS_DIR)
    if model_id < 1 or model_id > len(models):
        return jsonify({"error": "Model not found"}), 404
    model_name = models[model_id - 1]["name"]
    data = get_feature_importance(RESULTS_DIR, model_name, top_n=15)
    return jsonify(data)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
