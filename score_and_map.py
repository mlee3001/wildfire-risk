import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import TimestampedGeoJson
"""
Score daily wildfire risk and build a time-slider optical web map.
"""

feat = pd.read_csv("data/cell_day_model_features.csv", parse_dates=["date"])
grid = pd.read_csv("data/grid_2km.csv")
X_cols = [c for c in feat.columns if c not in ["cell_id","date","y","cx","cy"]]
model = joblib.load("models/xgb_nextday.pkl")
feat = feat.merge(grid[["cell_id","xmin","ymin","xmax","ymax","cx","cy"]], on="cell_id", how="left")
feat["risk"] = model.predict_proba(feat[X_cols].values)[:,1]
feat[["cell_id","date","risk"]].to_csv("data/risk_scores.csv", index=False)
start = feat["date"].min()
end = feat["date"].max()
dates = pd.date_range(start, end, freq="D")
features = []
for d in dates:
    df = feat[feat["date"] == d][["risk","xmin","ymin","xmax","ymax"]]
    for _, r in df.iterrows():
        coords = [[r["xmin"], r["ymin"]],[r["xmax"], r["ymin"]],[r["xmax"], r["ymax"]],[r["xmin"], r["ymax"]],[r["xmin"], r["ymin"]]]
        features.append({"type":"Feature","geometry":{"type":"Polygon","coordinates":[coords]},"properties":{"time": d.isoformat(),"risk": float(r["risk"])}})
gj = {"type":"FeatureCollection","features":features}
m = folium.Map(location=[34.5,-118.5], zoom_start=7, tiles=None)
folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Tiles © Esri", name="Esri WorldImagery").add_to(m)
def style_function(f):
    r = f["properties"]["risk"]
    return {"fillColor":"#000000","color":None,"fillOpacity":float(min(max(r,0.05),0.9))}
TimestampedGeoJson(data=gj, period="P1D", add_last_point=False, transition_time=200, loop=False, auto_play=False, time_slider_drag_update=True, style_function=style_function).add_to(m)
folium.LayerControl().add_to(m)
m.save("figures/map_risk_timeseries.html")