import sqlite3
from pathlib import Path

base = Path(__file__).resolve().parent.parent / 'ml-pipeline'
paths = [base / 'mlruns' / 'mlflow.db', base / 'mlflow.db']
for p in paths:
    print('\nDB:', p)
    if not p.exists():
        print('  (not found)')
        continue
    conn = sqlite3.connect(str(p))
    cur = conn.cursor()
    try:
        cur.execute('SELECT experiment_id, name FROM experiments')
        print(' Experiments:', cur.fetchall())
    except Exception as e:
        print('  error reading experiments:', e)
    try:
        cur.execute('SELECT count(*) FROM runs')
        print(' Runs count:', cur.fetchone()[0])
    except Exception as e:
        print('  error reading runs:', e)
    conn.close()
