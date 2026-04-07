import requests
import json
url = 'http://127.0.0.1:5001/api/2.0/mlflow/experiments/search'
body = {"filter": "", "view_type": 0, "max_results": 100}
resp = requests.post(url, json=body)
print('status', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
