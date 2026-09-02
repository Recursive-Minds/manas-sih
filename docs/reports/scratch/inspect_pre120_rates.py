import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

t0 = trip.imu_samples[0].timestamp_ns
g_samples = [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= 120.0 and (g.speed_mps or 0) >= 1.5]

g_ts = np.array([g.timestamp_ns for g in g_samples])
g_brg = np.array([g.bearing_deg for g in g_samples])
dt_g = np.diff(g_ts) * 1e-9
valid_diff = np.where(dt_g > 0.5)[0]

g_brg_unwrap = np.unwrap(np.radians(g_brg))
g_rates = np.diff(g_brg_unwrap)[valid_diff] / dt_g[valid_diff]
g_mids = (g_ts[valid_diff] + g_ts[valid_diff + 1]) // 2

imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])
gy = np.array([s.gyro[1] for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])
gy_mids = np.interp(g_mids, imu_ts, gy)

print("Pre-120s GNSS turning samples:")
for i in range(len(valid_diff)):
    t_sec = (g_mids[i] - t0)*1e-9
    print(f"  t={t_sec:.1f}s | GNSS d(brg)/dt={np.degrees(g_rates[i]):.2f}°/s | Gyro_y={gy_mids[i]:.4f} rad/s ({np.degrees(gy_mids[i]):.2f}°/s)")
