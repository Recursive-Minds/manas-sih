import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(120.0 * 1e9)
bo_end_ns = t0_ns + int(150.0 * 1e9)

samples = [s for s in trip.imu_samples if bo_start_ns <= s.timestamp_ns <= bo_end_ns]
df_s = pd.DataFrame([{
    "t_s": (s.timestamp_ns - t0_ns) * 1e-9,
    "ax": s.accel[0], "ay": s.accel[1], "az": s.accel[2],
    "gx": s.gyro[0], "gy": s.gyro[1], "gz": s.gyro[2],
} for s in samples])

print("Gyro stats during blackout 120s-150s:")
print(df_s[["gx", "gy", "gz"]].describe())

# Check raw dataframe columns directly from CSV
raw_csv = download_iovnbd_trip("S-S1")
df_raw = pd.read_csv(raw_csv, encoding="latin-1")
print("\nRaw columns:")
for c in df_raw.columns:
    if "gyro" in c.lower() or "speed" in c.lower() or "lat" in c.lower() or "long" in c.lower():
        print(f"  {c}")

# Let's inspect raw gyro columns around 120s to 150s
t_col = [c for c in df_raw.columns if "time" in c.lower() or "sec" in c.lower() or "index" in c.lower() or "ms" in c.lower()][0]
t_raw = df_raw[t_col].values
if t_raw.max() > 1e15: # ns
    t_s = (t_raw - t_raw[0]) * 1e-9
elif t_raw.max() > 1e10: # us
    t_s = (t_raw - t_raw[0]) * 1e-6
elif t_raw.max() > 1e7: # ms
    t_s = (t_raw - t_raw[0]) * 1e-3
else:
    t_s = t_raw - t_raw[0]

df_raw["_ts_calc"] = t_s
df_bo_raw = df_raw[(df_raw["_ts_calc"] >= 120.0) & (df_raw["_ts_calc"] <= 150.0)]
gyro_cols = [c for c in df_raw.columns if "gyro" in c.lower()]
print(f"\nRaw gyro columns during blackout: {gyro_cols}")
print(df_bo_raw[gyro_cols].describe())
