import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))
raw_csv_s2 = download_iovnbd_trip("S-S2")
df_s2 = pd.read_csv(raw_csv_s2, encoding="latin-1")

print("S-S2 raw orientations & accelerations:")
cols = [c for c in df_s2.columns if any(k in c.lower() for k in ["orient", "accel", "gyro", "speed", "lat", "lon"])]
print(df_s2[cols].head(3).to_string())

# Check S-S2 centripetal correlations
accels_s2 = np.array([s.accel for s in trip_s2.imu_samples[:2000]])
gyros_s2 = np.array([s.gyro for s in trip_s2.imu_samples[:2000]])

print("\nS-S2 Cross-correlations between horizontal accels and gyros:")
for g_ax in [0, 1, 2]:
    for a_ax in [0, 1]:
        c = np.corrcoef(accels_s2[:, a_ax], gyros_s2[:, g_ax])[0, 1]
        print(f"  corr(a[{a_ax}], gyro[{g_ax}]): {c:.4f}")
