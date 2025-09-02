import pandas as pd
import matplotlib.pyplot as plt
"""
Creates EDA figures from clean data
Outputs daily trends, by-satellite trends, confidence, FRP, day/night split,
spatial hexbin, and local-time histogram
"""

df = pd.read_csv("data/viirs_socall_clean.csv", parse_dates=["acq_datetime_utc"])
daily = df.groupby("date").size().rename("count").reset_index()
plt.figure(figsize=(8,4))
plt.plot(pd.to_datetime(daily["date"]), daily["count"])
plt.title("VIIRS Detections per Day — Southern California")
plt.xlabel("Date")
plt.ylabel("Detections")
plt.tight_layout()
plt.savefig("figures/daily_counts.png", dpi=160)
plt.close()

plt.figure(figsize=(8,4))
tmp = df.groupby(["date","satellite"]).size().unstack(fill_value=0)
for col in tmp.columns:
    plt.plot(pd.to_datetime(tmp.index), tmp[col], label=str(col))
plt.title("Detections per Day by Satellite — Southern California")
plt.xlabel("Date")
plt.ylabel("Detections")
plt.legend()
plt.tight_layout()
plt.savefig("figures/daily_by_satellite.png", dpi=160)
plt.close()

plt.figure(figsize=(6,4))
if "confidence_num" in df.columns and df["confidence_num"].notna().any():
    df["confidence_num"].plot(kind="hist", bins=20)
    plt.title("Confidence Distribution (Numeric) — Southern California")
    plt.xlabel("Confidence (%)")
    plt.tight_layout()
    plt.savefig("figures/confidence_hist.png", dpi=160)
    plt.close()
else:
    vc = df.get("confidence_cat", pd.Series(dtype=object)).value_counts(dropna=False)
    vc.plot(kind="bar")
    plt.title("Confidence Classes — Southern California")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figures/confidence_bar.png", dpi=160)
    plt.close()

plt.figure(figsize=(6,4))
cap = df["frp"].quantile(0.99)
df["frp"].clip(upper=cap).plot(kind="hist", bins=20)
plt.title("FRP Distribution (99th Percentile Capped) — Southern California")
plt.xlabel("FRP (MW)")
plt.tight_layout()
plt.savefig("figures/frp_hist.png", dpi=160)
plt.close()

plt.figure(figsize=(5,4))
df["daynight"].value_counts(dropna=False).plot(kind="bar")
plt.title("Detections by Day/Night — Southern California")
plt.xlabel("Day (D) vs Night (N)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/daynight_bar.png", dpi=160)
plt.close()

plt.figure(figsize=(6,5))
plt.hexbin(df["longitude"], df["latitude"], gridsize=60, bins="log")
plt.title("Spatial Density (Hexbin) — Southern California")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("figures/spatial_hexbin.png", dpi=160)
plt.close()

plt.figure(figsize=(6,4))
dt_local = df["acq_datetime_utc"].dt.tz_convert("America/Los_Angeles")
dt_local.dt.hour.dropna().plot(kind="hist", bins=24)
plt.title("Local-Time Detection Histogram — Southern California")
plt.xlabel("Hour of Day (Local)")
plt.tight_layout()
plt.savefig("figures/local_hour_hist.png", dpi=160)
plt.close()