from pathlib import Path
import sqlite3
import sys

paths = [
    Path(r"D:\II MSc\IV SEM\mlops\MLOPs LAB\lab\Final Project\ml-flask-project\ml-pipeline\mlruns\mlflow.db"),
    Path(r"D:\II MSc\IV SEM\mlops\MLOPs LAB\lab\Final Project\ml-flask-project\ml-pipeline\mlflow.db"),
]

for p in paths:
    print('\nchecking', p)
    print('exists', p.exists())
    if not p.exists():
        continue
    conn = sqlite3.connect(str(p))
    cur = conn.cursor()
    cur.execute('SELECT experiment_id,name FROM experiments')
    print('experiments', cur.fetchall())
    cur.execute('SELECT count(*) FROM runs')
    print('runs', cur.fetchone()[0])
    conn.close()
