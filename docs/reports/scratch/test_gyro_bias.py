import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

# 1. Pre-blackout GNSS and IMU (t <= 120s)
pre_g = [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= 120.0 and (g.speed_mps or 0) > 3.0]
g_ts = np.array([g.timestamp_ns for g in pre_g])
g_brg = np.unwrap(np.radians(np.array([g.bearing_deg for g in pre_g])))
d_theta_g = np.diff(g_brg)
dt_g = np.diff(g_ts) * 1e-9

imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])
gy = np.array([s.gyro[1] for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9]) # Y-gyro

biases = []
for i in range(len(d_theta_g)):
    t1 = g_ts[i]
    t2 = g_ts[i+1]
    mask = (imu_ts >= t1) & (imu_ts <= t2)
    if np.sum(mask) > 1:
        dt_imu = np.diff(imu_ts[mask]) * 1e-9
        int_gyro = np.sum(0.5 * (gy[mask][:-1] + gy[mask][1:]) * dt_imu)
        # d_theta = -w_z * dt => w_z = -d_theta / dt
        # gyro_measured = w_z + bg = -d_theta_g / dt + bg
        bg_sample = (int_gyro / dt_g[i]) - (-d_theta_g[i] / dt_g[i])
        biases.append(bg_sample)

mean_bg = float(np.mean(biases)) if len(biases) > 0 else 0.0
print(f"Estimated empirical gyro bias before blackout: {mean_bg:.6f} rad/s ({np.degrees(mean_bg):.4f}°/s)")
