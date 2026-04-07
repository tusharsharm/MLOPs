import sqlite3
from pathlib import Path
p = Path('ml-flask-project') / 'ml-pipeline' / 'mlruns' / 'mlflow_from_container.db'
print('checking', p)
print('exists', p.exists())
if not p.exists():
    raise SystemExit('db missing')
conn = sqlite3.connect(str(p))
cur = conn.cursor()
cur.execute('SELECT experiment_id, name FROM experiments')
print('experiments', cur.fetchall())
cur.execute('SELECT count(*) FROM runs')
print('runs', cur.fetchone()[0])
conn.close()
