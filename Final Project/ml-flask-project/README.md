# ml-flask-project

Flask API + training pipeline with MLflow UI for tracking experiments.

Quick start (Docker):

- Build the image:

```powershell
docker build -t ml-flask-app ./ml-flask-project
```

- Run the container and bind-mount the host `ml-pipeline` folder so `mlruns`, `results` and `data` persist:

```powershell
docker run -d --name ml-flask-container \
    -p 5000:5000 -p 5001:5001 \
    --mount type=bind,source="${PWD}/ml-flask-project/ml-pipeline",target=/app/ml-pipeline \
    ml-flask-app
```

- Endpoints:
    - Flask API: http://localhost:5000 (health, pipeline control, models)
    - MLflow UI: http://localhost:5001 (experiments and runs)

Behaviour notes:
- The container will by default launch an initial training run on start and write run artifacts into the host-mounted `ml-pipeline/mlruns` directory so they are visible in the MLflow UI.
- To retrain manually, call the pipeline endpoint:

```bash
curl -X POST http://localhost:5000/api/pipeline/run
```

- If you prefer to disable the automatic training on container start, edit `start.sh` and comment out or remove the block that calls the pipeline at startup.

Where files are stored:
- Host `ml-pipeline/mlruns` — MLflow run folders and artifacts (persisted)
- Host `ml-pipeline/results` — pipeline JSON results and model summaries
- Host `ml-pipeline/data` — dataset files

Troubleshooting:
- If experiments do not appear in the UI, ensure the container was started with the `ml-pipeline` bind mount and refresh the MLflow UI.
- For a more feature-complete MLflow server (jobs, search), consider switching to a SQL backend (SQLite or PostgreSQL) and setting `--backend-store-uri` when launching the MLflow server.

Development:
- Run the API locally: `python app.py` (requires the virtualenv and `requirements.txt`)

License: none
# Global AI vs Human Content — ML Experiment API



## Project Layout

```
ml-flask-project/
├── app.py                        # Flask application (all endpoints)
├── requirements.txt
├── test_api.py                   # Smoke-test every endpoint
│
├── pipeline/
│   ├── __init__.py
│   ├── dataset.py                # Load CSV or generate synthetic data
│   ├── preprocess.py             # Encode + split
│   ├── models.py                 # LR / RF / XGBoost trainers + MLflow logging
│   ├── train.py                  # Orchestrator (called by Flask or directly)
│   └── utils.py                  # Read result JSON → shape for API
│
└── ml-pipeline/                  # Created at runtime
    ├── data/
    │   └── ai_human_content.csv  #  Kaggle CSV 
    ├── results/
    │   ├── pipeline_status.json
    │   ├── dataset_info.json
    │   ├── summary.json
    │   ├── logistic_regression/  # result.json, confusion_matrix.json, roc_curve.json, feature_importance.json
    │   ├── random_forest/
    │   └── xgboost/
    └── mlruns/                   # MLflow tracking store
```

---

## Quick Start (VS Code)

### 1. Clone / open the folder in VS Code

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add the real Kaggle dataset

Download from:
https://www.kaggle.com/datasets/asifxzaman/global-ai-vs-human-content-dataset-2026

Place the CSV at:
```
ml-pipeline/data/ai_human_content.csv
```

If the file is absent, a 10 000-sample synthetic dataset
with the same feature distribution is generated automatically.

### 5. Start the Flask server

```bash
python app.py
```

Server runs at `http://localhost:5000`.

### 6. Trigger training

```bash
curl -X POST http://localhost:5000/api/pipeline/run
```

Training runs in a background thread (~30-60 seconds).
Poll status with:

```bash
curl http://localhost:5000/api/pipeline/status
```

### 7. Explore results

See **All Endpoints** below, or run the smoke-test:

```bash
python test_api.py
```

### 8. View MLflow UI

```bash
mlflow ui --backend-store-uri ml-pipeline/mlruns
```
Open `http://localhost:5001` to compare runs.

---

## All Endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server health check |

### Pipeline

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pipeline/status` | Current training status + progress |
| POST | `/api/pipeline/run` | Trigger training (non-blocking) |
| GET | `/api/pipeline/dataset-info` | Dataset stats (samples, features, class distribution) |

### Experiments

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/experiments` | List all 3 experiment runs |
| GET | `/api/experiments/summary` | Best model, avg training time, accuracy |
| GET | `/api/experiments/<id>` | Single experiment (id = 1, 2, or 3) |

### Models

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models` | All 3 models with metrics + isBest flag |
| GET | `/api/models/best` | Best performing model |
| GET | `/api/models/<id>/confusion-matrix` | 2×2 confusion matrix (TP/TN/FP/FN) |
| GET | `/api/models/<id>/roc-curve` | ROC curve data (fpr, tpr, thresholds, auc) |
| GET | `/api/models/<id>/feature-importance` | Top 15 feature importances |

Model IDs: `1` = Logistic Regression, `2` = Random Forest, `3` = XGBoost

