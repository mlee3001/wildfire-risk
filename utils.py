from pathlib import Path
import pandas as pd
import numpy as np
"""
Shared utilities for the project
Provides: SOCAL_BBOX, ensure_dirs(), parse_utc(), build_grid(), assign_cell_ids().
Used by other scripts to handle directory setup, timestamp parsing, and 2-km grid logic.
"""

SOCAL_BBOX = (-121.0, 32.0, -114.0, 37.0)

def ensure_dirs():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

def parse_utc(date_val, time_val):
    s = str(time_val)
    if s.isdigit():
        s = f"{int(s):04d}"
    else:
        s = "".join([ch for ch in s if ch.isdigit()])
        s = (s + "0000")[:4]
    return pd.to_datetime(str(date_val) + " " + s[:2] + ":" + s[2:] + ":00", utc=True, errors="coerce")

def build_grid(xmin, ymin, xmax, ymax, cell_km=2.0):
    deg = cell_km / 111.0
    nx = int(np.ceil((xmax - xmin) / deg))
    ny = int(np.ceil((ymax - ymin) / deg))
    rows = []
    for j in range(ny):
        y0 = ymin + j * deg
        for i in range(nx):
            x0 = xmin + i * deg
            cid = i + j * nx
            rows.append((cid, x0, y0, x0 + deg, y0 + deg, x0 + deg / 2, y0 + deg / 2))
    return pd.DataFrame(rows, columns=["cell_id","xmin","ymin","xmax","ymax","cx","cy"])

def assign_cell_ids(df, xmin, ymin, xmax, ymax, cell_km=2.0):
    deg = cell_km / 111.0
    nx = int(np.ceil((xmax - xmin) / deg))
    i = np.floor((df["longitude"].values - xmin) / deg).astype(int)
    j = np.floor((df["latitude"].values - ymin) / deg).astype(int)
    return i + j * nx