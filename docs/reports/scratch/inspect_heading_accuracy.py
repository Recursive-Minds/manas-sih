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

g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
g_brg = np.array([g.bearing_deg for g in trip.gnss_samples])
g_spd = np.array([g.speed_mps for g in trip.gnss_samples])

enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])

# Inspect 30s blackout at 120s
print("--- SCENARIO 1: 30s blackout at 120s ---")
pre_120 = [g for g in trip.gnss_samples if (g.timestamp_ns - t0_ns)*1e-9 <= 120.0]
post_120 = [g for g in trip.gnss_samples if 120.0 <= (g.timestamp_ns - t0_ns)*1e-9 <= 150.0]
last_g_120 = pre_120[-1]
t_last_120 = (last_g_120.timestamp_ns - t0_ns)*1e-9
print(f"Last GNSS fix before 120s: t={t_last_120:.2f}s, Bearing={last_g_120.bearing_deg:.2f}°, Speed={last_g_120.speed_mps:.2f} m/s")

# Compute actual ground truth trajectory direction between 120s and 125s from ENU coordinates
e_120 = np.interp(t0_ns + 120e9, g_ts, enu_all[:, 0])
n_120 = np.interp(t0_ns + 120e9, g_ts, enu_all[:, 1])
e_125 = np.interp(t0_ns + 125e9, g_ts, enu_all[:, 0])
n_125 = np.interp(t0_ns + 125e9, g_ts, enu_all[:, 1])
actual_brg_120 = np.degrees(np.arctan2(e_125 - e_120, n_125 - n_120)) % 360.0
print(f"Actual Ground Track COG at 120s: {actual_brg_120:.2f}° (Difference from last GNSS fix: {actual_brg_120 - last_g_120.bearing_deg:.2f}°)")

# Inspect 60s blackout at 300s
print("\n--- SCENARIO 2: 60s blackout at 300s ---")
pre_300 = [g for g in trip.gnss_samples if (g.timestamp_ns - t0_ns)*1e-9 <= 300.0]
post_300 = [g for g in trip.gnss_samples if 300.0 <= (g.timestamp_ns - t0_ns)*1e-9 <= 360.0]
last_g_300 = pre_300[-1]
t_last_300 = (last_g_300.timestamp_ns - t0_ns)*1e-9
print(f"Last GNSS fix before 300s: t={t_last_300:.2f}s, Bearing={last_g_300.bearing_deg:.2f}°, Speed={last_g_300.speed_mps:.2f} m/s")

e_300 = np.interp(t0_ns + 300e9, g_ts, enu_all[:, 0])
n_300 = np.interp(t0_ns + 300e9, g_ts, enu_all[:, 1])
e_305 = np.interp(t0_ns + 305e9, g_ts, enu_all[:, 0])
n_305 = np.interp(t0_ns + 305e9, g_ts, enu_all[:, 1])
actual_brg_300 = np.degrees(np.arctan2(e_305 - e_300, n_305 - n_300)) % 360.0
print(f"Actual Ground Track COG at 300s: {actual_brg_300:.2f}° (Difference from last GNSS fix: {actual_brg_300 - last_g_300.bearing_deg:.2f}°)")
