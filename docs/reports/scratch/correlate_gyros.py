import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

# Extract GNSS yaw rate (d(bearing)/dt) for all moving points
g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
g_brg = np.array([g.bearing_deg for g in trip.gnss_samples])
g_spd = np.array([g.speed_mps if g.speed_mps is not None else 0.0 for g in trip.gnss_samples])

# Unwrapped bearing in radians
g_brg_unwrapped = np.unwrap(np.radians(g_brg))

# Calculate ground truth yaw rate from GNSS: d(bearing)/dt
dt_g = np.diff(g_ts) * 1e-9
# In ENU clockwise from North, a right turn means bearing increases (d(bearing)/dt > 0)
gnss_yaw_rate = np.diff(g_brg_unwrapped) / np.maximum(dt_g, 1.0) # rad/s
g_ts_mid = (g_ts[:-1] + g_ts[1:]) // 2

# Filter moving points
valid_g = np.where((dt_g > 1.0) & (g_spd[:-1] > 3.0) & (g_spd[1:] > 3.0))[0]

# Interpolate IMU gyros to GNSS timestamps
imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples])
gx = np.array([s.gyro[0] for s in trip.imu_samples])
gy = np.array([s.gyro[1] for s in trip.imu_samples])
gz = np.array([s.gyro[2] for s in trip.imu_samples])

gx_interp = np.interp(g_ts_mid[valid_g], imu_ts, gx)
gy_interp = np.interp(g_ts_mid[valid_g], imu_ts, gy)
gz_interp = np.interp(g_ts_mid[valid_g], imu_ts, gz)
gt_rates = gnss_yaw_rate[valid_g]

print("Correlation with GNSS Yaw Rate (d(bearing)/dt):")
print(f"  Corr(gt_rate, gx): {np.corrcoef(gt_rates, gx_interp)[0, 1]:.4f}")
print(f"  Corr(gt_rate, gy): {np.corrcoef(gt_rates, gy_interp)[0, 1]:.4f}")
print(f"  Corr(gt_rate, gz): {np.corrcoef(gt_rates, gz_interp)[0, 1]:.4f}")

print("\nNegative correlation check:")
print(f"  Corr(gt_rate, -gx): {np.corrcoef(gt_rates, -gx_interp)[0, 1]:.4f}")
print(f"  Corr(gt_rate, -gy): {np.corrcoef(gt_rates, -gy_interp)[0, 1]:.4f}")
print(f"  Corr(gt_rate, -gz): {np.corrcoef(gt_rates, -gz_interp)[0, 1]:.4f}")
