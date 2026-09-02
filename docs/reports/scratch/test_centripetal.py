import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

# Get all IMU samples before 120s
samples_120 = [s for s in trip.imu_samples if (s.timestamp_ns - trip.imu_samples[0].timestamp_ns)*1e-9 <= 120.0]

ax = np.array([s.accel[0] for s in samples_120])
ay = np.array([s.accel[1] for s in samples_120])
az = np.array([s.accel[2] for s in samples_120])

gx = np.array([s.gyro[0] for s in samples_120])
gy = np.array([s.gyro[1] for s in samples_120])
gz = np.array([s.gyro[2] for s in samples_120])

# In high school physics: centripetal lateral acceleration is a_lat = v * w.
# Since v > 0, a_lat is proportional to w!
# Let's check cross-correlation between horizontal accels (ax, ay) and gyros (gx, gy, gz)
print("Cross-correlations between horizontal accels and gyros:")
print(f"  corr(ax, gx): {np.corrcoef(ax, gx)[0, 1]:.4f}")
print(f"  corr(ax, gy): {np.corrcoef(ax, gy)[0, 1]:.4f}")
print(f"  corr(ax, gz): {np.corrcoef(ax, gz)[0, 1]:.4f}")
print(f"  corr(ay, gx): {np.corrcoef(ay, gx)[0, 1]:.4f}")
print(f"  corr(ay, gy): {np.corrcoef(ay, gy)[0, 1]:.4f}")
print(f"  corr(ay, gz): {np.corrcoef(ay, gz)[0, 1]:.4f}")
