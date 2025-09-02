# wildfire-risk
Wildfire Nowcasting and Risk Prediction Utilizing Satellite Data (SoCal)

## Features
latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight

## Quick start
```bash
python data_ingest.py
python make_figures.py
python build_grid_features.py
python join_external_features.py  # place era5_cell_day.csv and ndvi_cell_day.csv in data/ if available
python train_model.py
python score_and_map.py
python simulate_spread.py