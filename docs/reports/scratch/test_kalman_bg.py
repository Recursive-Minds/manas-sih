import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.config import PipelineConfig, FusionFilterConfig, VelocityEstimatorConfig, CalibrationConfig
from sih.eval.benchmark import BlackoutConfig, BenchmarkRunner

# Let's inspect what the gyro bias is in S-S1 before 120s and 300s
trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

# Extract GNSS and IMU before 120s
g_sub = [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= 120.0 and (g.speed_mps or 0) > 3.0]
g_ts = np.array([g.timestamp_ns for g in g_sub])
g_brg = np.unwrap(np.radians(np.array([g.bearing_deg for g in g_sub])))

imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])
gy = np.array([s.gyro[1] for s in trip.imu_samples if s.timestamp_ns <= t0 + 120e9])

bg_estimates = []
for i in range(len(g_ts) - 1):
    t1 = g_ts[i]
    t2 = g_ts[i+1]
    dt_g = (t2 - t1) * 1e-9
    if dt_g > 0.1:
        mask = (imu_ts >= t1) & (imu_ts <= t2)
        if np.sum(mask) > 1:
            dt_i = np.diff(imu_ts[mask]) * 1e-9
            # heading integration: d_hdg = - (w_z - bg) * dt => d_hdg = - w_z * dt + bg * dt
            # Therefore: bg * dt = d_hdg + w_z * dt
            d_hdg_gnss = g_brg[i+1] - g_brg[i]
            int_w = np.sum(0.5 * (gy[mask][:-1] + gy[mask][1:]) * dt_i)
            bg_k = (d_hdg_gnss + int_w) / dt_g
            bg_estimates.append(bg_k)

bg_estimates = np.array(bg_estimates)
print(f"Mean Gyro Bias before 120s:   {np.mean(bg_estimates):.6f} rad/s ({np.degrees(np.mean(bg_estimates)):.4f}°/s)")
print(f"Median Gyro Bias before 120s: {np.median(bg_estimates):.6f} rad/s ({np.degrees(np.median(bg_estimates)):.4f}°/s)")
print(f"Std of Gyro Bias before 120s: {np.std(bg_estimates):.6f} rad/s")
