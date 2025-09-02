import pandas as pd
from pathlib import Path
from utils import ensure_dirs, parse_utc, SOCAL_BBOX
"""
Ingest and normalize FIRMS VIIRS CSVs
Merges inputs, standardizes fields, parses UTC timestamps, 
filters to location, and writes cleaned detections
"""

ensure_dirs()
paths = sorted(Path("data").glob("fire_nrt_*july25.csv"))
dfs = []
for p in paths:
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df["acq_datetime_utc"] = df.apply(lambda r: parse_utc(r.get("acq_date"), r.get("acq_time")), axis=1)
for c in ["latitude","longitude","frp"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["daynight"] = df.get("daynight", pd.Series(index=df.index)).astype(str).str.upper()
df["satellite"] = df.get("satellite", pd.Series(index=df.index)).astype(str)
if "confidence" in df.columns:
    conf_raw = df["confidence"].astype(str).str.strip().str.lower()
    letter_map = {"l":"low","n":"nominal","h":"high"}
    df["confidence_cat"] = conf_raw.map(lambda x: letter_map.get(x, x))
    df["confidence_num"] = pd.to_numeric(df["confidence"], errors="coerce")
xmin, ymin, xmax, ymax = SOCAL_BBOX
mask = df["longitude"].between(xmin, xmax) & df["latitude"].between(ymin, ymax)
df = df.loc[mask].copy()
df = df.dropna(subset=["latitude","longitude","acq_datetime_utc"])
if "frp" in df.columns:
    df = df[df["frp"].notna() & (df["frp"] >= 0)]
df["date"] = df["acq_datetime_utc"].dt.date
df.to_csv("data/viirs_socall_clean.csv", index=False)