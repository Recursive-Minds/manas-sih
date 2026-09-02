import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))
t0 = trip_s2.imu_samples[0].timestamp_ns

g_df = pd.DataFrame([{
    "t_s": (g.timestamp_ns - t0)*1e-9,
    "speed": g.speed_mps,
    "bearing": g.bearing_deg,
    "lat": g.latitude_deg,
    "lon": g.longitude_deg
} for g in trip_s2.gnss_samples])

# Find 30s and 60s windows in S-S2 with high driving distance and sharp turns
best_windows = []
for start_t in range(100, int(g_df["t_s"].max()) - 120, 60):
    sub = g_df[(g_df["t_s"] >= start_t) & (g_df["t_s"] <= start_t + 60)]
    if len(sub) >= 4:
        # Distance
        lats = sub["lat"].values
        lons = sub["lon"].values
        d_lat = np.diff(lats) * 111139.0
        d_lon = np.diff(lons) * (111139.0 * np.cos(np.radians(np.mean(lats))))
        dist = np.sum(np.sqrt(d_lat**2 + d_lon**2))
        
        # Turn angle
        brgs = sub["bearing"].dropna().values
        if len(brgs) >= 2:
            turn_deg = abs(np.degrees(np.unwrap(np.radians(brgs))[-1] - np.unwrap(np.radians(brgs))[0]))
        else:
            turn_deg = 0.0
            
        best_windows.append({"start_t": start_t, "dist_60s": dist, "turn_deg": turn_deg})

df_win = pd.DataFrame(best_windows)
print("Top 10 most dynamic driving + turning segments in S-S2:")
print(df_win[(df_win["dist_60s"] > 500) & (df_win["turn_deg"] > 30)].sort_values("dist_60s", ascending=False).head(10).to_string())
