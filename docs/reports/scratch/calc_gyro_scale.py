import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

# Extract all GNSS turns where speed > 5 m/s
g_samples = [g for g in trip.gnss_samples if (g.speed_mps or 0) > 5.0]
g_ts = np.array([g.timestamp_ns for g in g_samples])
g_brg = np.unwrap(np.radians(np.array([g.bearing_deg for g in g_samples])))
dt_g = np.diff(g_ts) * 1e-9

d_theta_gnss = np.diff(g_brg) # rad

imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples])
gy = np.array([s.gyro[1] for s in trip.imu_samples])

d_theta_gyro = []
for i in range(len(d_theta_gnss)):
    t1 = g_ts[i]
    t2 = g_ts[i+1]
    mask = (imu_ts >= t1) & (imu_ts <= t2)
    if np.sum(mask) > 1:
        dt_imu = np.diff(imu_ts[mask]) * 1e-9
        int_y = np.sum(0.5 * (gy[mask][:-1] + gy[mask][1:]) * dt_imu)
        d_theta_gyro.append(int_y)
    else:
        d_theta_gyro.append(0.0)

d_theta_gyro = np.array(d_theta_gyro)

# Select major turns where |d_theta_gnss| > 5 degrees (0.087 rad)
major_turns = np.abs(d_theta_gnss) > np.radians(5.0)
ratio = -d_theta_gnss[major_turns] / d_theta_gyro[major_turns]

print(f"Across {np.sum(major_turns)} major turns in S-S1:")
print(f"  Mean Turn Ratio (-d_theta_GNSS / d_theta_gyro): {np.mean(ratio):.4f}")
print(f"  Median Turn Ratio:                              {np.median(ratio):.4f}")
print(f"  25th percentile:                                {np.percentile(ratio, 25):.4f}")
print(f"  75th percentile:                                {np.percentile(ratio, 75):.4f}")
