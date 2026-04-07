import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "ml-pipeline" / "mlflow.db"
print("DB path:", DB)
if not DB.exists():
    print("DB not found")
    raise SystemExit(1)

conn = sqlite3.connect(str(DB))
cur = conn.cursor()

print('\nExperiments:')
try:
    cur.execute('SELECT experiment_id, name, lifecycle_stage FROM experiments')
    rows = cur.fetchall()
    for r in rows:
        print(' ', r)
except Exception as e:
    print('  Error reading experiments:', e)

print('\nRuns:')
try:
    cur.execute('SELECT run_uuid, experiment_id, lifecycle_stage FROM runs')
    rows = cur.fetchall()
    for r in rows[:20]:
        print(' ', r)
    print('  total runs:', len(rows))
except Exception as e:
    print('  Error reading runs:', e)

conn.close()
