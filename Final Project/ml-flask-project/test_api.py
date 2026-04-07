"""
Quick smoke-test for every endpoint.
Run AFTER starting the Flask server:   python app.py
Then:                                  python test_api.py
"""

import json
import time
import requests

BASE = "http://localhost:5000/api"


def get(path):
    r = requests.get(f"{BASE}{path}")
    print(f"GET  {path}  →  {r.status_code}")
    return r


def post(path, body=None):
    r = requests.post(f"{BASE}{path}", json=body or {})
    print(f"POST {path}  →  {r.status_code}")
    return r


def pprint(data):
    print(json.dumps(data, indent=2, default=str)[:600])
    print()


# ── Health ────────────────────────────────────────────────────────────────────
r = get("/health")
pprint(r.json())

# ── Pipeline status ───────────────────────────────────────────────────────────
r = get("/pipeline/status")
pprint(r.json())

# ── Trigger training ──────────────────────────────────────────────────────────
print("Starting training pipeline...")
r = post("/pipeline/run")
pprint(r.json())

# Poll until done
for _ in range(120):          # up to ~2 minutes
    time.sleep(3)
    r = get("/pipeline/status")
    status = r.json()
    print(f"  status={status['status']}  progress={status.get('progress')}  step={status.get('currentStep')}")
    if status["status"] in ("completed", "failed"):
        break

print()

# ── Experiments ───────────────────────────────────────────────────────────────
r = get("/experiments")
experiments = r.json()
pprint(experiments)

r = get("/experiments/summary")
pprint(r.json())

# ── Models ────────────────────────────────────────────────────────────────────
r = get("/models")
models = r.json()
pprint(models)

r = get("/models/best")
pprint(r.json())

# ── Per-model detail (model id 1 = Logistic Regression) ──────────────────────
for model_id in [1, 2, 3]:
    r = get(f"/models/{model_id}/confusion-matrix")
    pprint(r.json())

    r = get(f"/models/{model_id}/roc-curve")
    data = r.json()
    print(f"  ROC AUC for model {model_id}: {data.get('auc')}")
    print()

    r = get(f"/models/{model_id}/feature-importance")
    fi = r.json()
    print(f"  Top 5 features for model {model_id}:")
    for item in fi[:5]:
        print(f"    {item['rank']:2d}. {item['feature']:35s} {item['importance']:.6f}")
    print()

# ── Dataset info ──────────────────────────────────────────────────────────────
r = get("/pipeline/dataset-info")
pprint(r.json())

print("All checks done.")
