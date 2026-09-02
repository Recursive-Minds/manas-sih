import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(300.0 * 1e9)
bo_end_ns = t0_ns + int(360.0 * 1e9)

g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
g_brg = np.array([g.bearing_deg for g in trip.gnss_samples])
g_spd = np.array([g.speed_mps if g.speed_mps is not None else 0.0 for g in trip.gnss_samples])
g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
g_alts = np.array([g.altitude_m for g in trip.gnss_samples])

enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])

# Inspect GNSS fixes around 300s to 360s
df_g_bo = pd.DataFrame([{
    "t_s": (g.timestamp_ns - t0_ns) * 1e-9,
    "lat": g.latitude_deg, "lon": g.longitude_deg,
    "spd": g.speed_mps, "brg": g.bearing_deg,
    "e": enu_all[i, 0], "n": enu_all[i, 1]
} for i, g in enumerate(trip.gnss_samples) if 290.0 <= (g.timestamp_ns - t0_ns)*1e-9 <= 370.0])

print("GNSS fixes between 290s and 370s:")
print(df_g_bo.to_string())
