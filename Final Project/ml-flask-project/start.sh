#!/bin/sh
# Start MLflow UI in background, then start the Flask app
# MLflow will use the ml-pipeline/mlruns directory inside the container.

echo "Starting MLflow UI on port 5001 using file-based mlruns..."
# Let MLflow UI read the existing file-based `mlruns` directory (artifact + run metadata)
# This ensures the UI shows experiments created as file-based runs in /app/ml-pipeline/mlruns
mlflow ui --host 0.0.0.0 --port 5001 \
	--backend-store-uri file:///app/ml-pipeline/mlruns \
	--default-artifact-root /app/ml-pipeline/mlruns &

echo "Starting Flask app on port 5000..."
# Kick off an initial training run in the background so every container start
# produces a new MLflow run visible in the UI and in the host-mounted `mlruns`.
echo "Launching initial training run in background (this may take a few minutes)..."
python - <<'PY' &
from pathlib import Path
from pipeline.train import run_pipeline
try:
	run_pipeline(Path('/app/ml-pipeline/results'))
except Exception as e:
	print('Initial training run failed:', e)
PY

# Start the Flask app in the foreground
python app.py
