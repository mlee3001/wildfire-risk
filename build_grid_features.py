import pandas as pd
from utils import SOCAL_BBOX, build_grid, assign_cell_ids
"""
Build a 2-km grid and aggregate VIIRS detections into cell-day features.
Generates next-day ignition labels and short lag features.
"""

viirs = pd.read_csv("data/viirs_socall_clean.csv", parse_dates=["acq_datetime_utc"])
viirs["date"] = viirs["acq_datetime_utc"].dt.date
xmin, ymin, xmax, ymax = SOCAL_BBOX
grid = build_grid(xmin, ymin, xmax, ymax, cell_km=2.0)
viirs = viirs[(viirs["longitude"].between(xmin, xmax)) & (viirs["latitude"].between(ymin, ymax))]
viirs["cell_id"] = assign_cell_ids(viirs, xmin, ymin, xmax, ymax, cell_km=2.0)
hits = viirs.groupby(["cell_id","date"]).agg(frp_sum=("frp","sum"), frp_max=("frp","max"), n_det=("frp","size")).reset_index()
tmp = hits.copy()
tmp["date"] = pd.to_datetime(tmp["date"])
hits["date"] = pd.to_datetime(hits["date"])
tmp["date_next"] = tmp["date"] + pd.Timedelta(days=1)
fires_tomorrow = tmp[["cell_id","date_next"]].assign(y=1).rename(columns={"date_next":"date"})
tab = hits.merge(fires_tomorrow, on=["cell_id","date"], how="left").fillna({"y":0})
tab = tab.sort_values(["cell_id","date"])
for k in [1,3,7]:
    tab[f"frp_sum_l{k}"] = tab.groupby("cell_id")["frp_sum"].shift(1).rolling(k, min_periods=1).sum()
    tab[f"n_det_l{k}"] = tab.groupby("cell_id")["n_det"].shift(1).rolling(k, min_periods=1).sum()
tab = tab.dropna()
tab.to_csv("data/cell_day_viirs_features.csv", index=False)
grid.to_csv("data/grid_2km.csv", index=False)