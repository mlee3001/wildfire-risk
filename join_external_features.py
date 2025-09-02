import pandas as pd
from pathlib import Path
"""
Joins external features (weather, NDVI) and add rolling windows.
"""

base = pd.read_csv("data/cell_day_viirs_features.csv", parse_dates=["date"])
features = base.copy()
wpath = Path("data/era5_cell_day.csv")
vpath = Path("data/ndvi_cell_day.csv")
if wpath.exists():
    w = pd.read_csv(wpath, parse_dates=["date"])
    features = features.merge(w, on=["cell_id","date"], how="left")
if vpath.exists():
    v = pd.read_csv(vpath, parse_dates=["date"])
    features = features.merge(v, on=["cell_id","date"], how="left")
features = features.sort_values(["cell_id","date"])
num_cols = [c for c in features.columns if c not in ["cell_id","date","y"]]
for c in num_cols:
    for w in (3,7,30):
        features[f"{c}_l{w}"] = features.groupby("cell_id")[c].shift(1).rolling(w, min_periods=1).mean()
features = features.dropna()
features.to_csv("data/cell_day_model_features.csv", index=False)