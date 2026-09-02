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
g_brg_unwrap = np.unwrap(np.radians(g_brg))
d_theta_gnss = np.diff(g_brg_unwrap) # angular displacement over each GNSS fix

imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])
gy = np.array([s.gyro[1] for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])
gx = np.array([s.gyro[0] for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])
gz = np.array([s.gyro[2] for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])

d_theta_gy = []
d_theta_gx = []
d_theta_gz = []

for i in range(len(g_ts) - 1):
    t_start = g_ts[i]
    t_end = g_ts[i+1]
    mask = (imu_ts >= t_start) & (imu_ts <= t_end)
    if np.sum(mask) > 1:
        # Integrated angular change (trapz)
        dt_imu = np.diff(imu_ts[mask]) * 1e-9
        int_y = np.sum(0.5 * (gy[mask][:-1] + gy[mask][1:]) * dt_imu)
        int_x = np.sum(0.5 * (gx[mask][:-1] + gx[mask][1:]) * dt_imu)
        int_z = np.sum(0.5 * (gz[mask][:-1] + gz[mask][1:]) * dt_imu)
        d_theta_gy.append(int_y)
        d_theta_gx.append(int_x)
        d_theta_gz.append(int_z)
    else:
        d_theta_gy.append(0.0)
        d_theta_gx.append(0.0)
        d_theta_gz.append(0.0)

d_theta_gy = np.array(d_theta_gy)
d_theta_gx = np.array(d_theta_gx)
d_theta_gz = np.array(d_theta_gz)

print("Correlation of Integrated GNSS delta_theta with Integrated Gyro delta_theta:")
print(f"  Corr(d_theta_gnss, -d_theta_gy): {np.corrcoef(d_theta_gnss, -d_theta_gy)[0, 1]:.4f}")
print(f"  Corr(d_theta_gnss, +d_theta_gy): {np.corrcoef(d_theta_gnss, +d_theta_gy)[0, 1]:.4f}")
print(f"  Corr(d_theta_gnss, -d_theta_gz): {np.corrcoef(d_theta_gnss, -d_theta_gz)[0, 1]:.4f}")
print(f"  Corr(d_theta_gnss, +d_theta_gz): {np.corrcoef(d_theta_gnss, +d_theta_gz)[0, 1]:.4f}")
