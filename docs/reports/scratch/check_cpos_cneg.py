import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

g_samples = [g for g in trip.gnss_samples if (g.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 120.0 and (g.speed_mps or 0.0) >= 2.0]
g_ts = np.array([g.timestamp_ns for g in g_samples])
g_brg = np.array([g.bearing_deg for g in g_samples])
dt_g = np.diff(g_ts) * 1e-9
valid_diff = np.where(dt_g > 0.5)[0]

g_brg_unwrap = np.unwrap(np.radians(g_brg))
g_rates = np.diff(g_brg_unwrap)[valid_diff] / dt_g[valid_diff]
g_mids = (g_ts[valid_diff] + g_ts[valid_diff + 1]) // 2

imu_samples_120 = [s for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 120.0]
imu_ts_arr = np.array([s.timestamp_ns for s in imu_samples_120])
gyros_arr = np.array([s.gyro for s in imu_samples_120])

c_pos = np.corrcoef(g_rates, np.interp(g_mids, imu_ts_arr, gyros_arr[:, 1]))[0, 1]
c_neg = np.corrcoef(g_rates, -np.interp(g_mids, imu_ts_arr, gyros_arr[:, 1]))[0, 1]

print(f"At 120s: c_pos = {c_pos:.4f}, c_neg = {c_neg:.4f}")
