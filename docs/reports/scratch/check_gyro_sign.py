import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

loader = GenericDataLoader()
trip = loader.load_file("data/raw/iovnbd_trips/S-S1.csv")

t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(120 * 1e9)
bo_end = bo_start + int(30 * 1e9)

# Get IMU samples during blackout
imus = [s for s in trip.imu_samples if bo_start <= s.timestamp_ns <= bo_end]
gnss = [g for g in trip.gnss_samples if bo_start - 10e9 <= g.timestamp_ns <= bo_end + 10e9]

print("GNSS Bearings during blackout:")
for g in gnss:
    print(f"  t={(g.timestamp_ns - t0)*1e-9:5.1f}s | bearing={g.bearing_deg:5.1f} deg")

print("\nIMU Gyro Z during blackout (every 2 seconds):")
for i in range(0, len(imus), 20):
    s = imus[i]
    print(f"  t={(s.timestamp_ns - t0)*1e-9:5.1f}s | Gyro: roll={s.gyro[0]:6.3f}, pitch={s.gyro[1]:6.3f}, yaw={s.gyro[2]:6.3f} rad/s")

# Let's integrate Gyro Z with both + and - signs
dt = 0.1
# Starting heading at t=120s is 348.9 deg (in ENU yaw: 90 - 348.9 = -258.9 = +101.1 deg)
yaw_pos = 101.1
yaw_neg = 101.1

for s in imus:
    gz = s.gyro[2]
    yaw_pos += np.degrees(gz * dt)
    yaw_neg -= np.degrees(gz * dt)

bearing_pos = (90.0 - yaw_pos) % 360.0
bearing_neg = (90.0 - yaw_neg) % 360.0
print(f"\nFinal Bearing after 30s integration:")
print(f"  Ground Truth Target Bearing: ~314.2 deg")
print(f"  With + Gyro Z: {bearing_pos:5.1f} deg")
print(f"  With - Gyro Z: {bearing_neg:5.1f} deg")
