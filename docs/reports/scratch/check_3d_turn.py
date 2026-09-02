import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

# 60s turn between 300s and 360s
bo_imu = [s for s in trip.imu_samples if 300.0 <= (s.timestamp_ns - t0)*1e-9 <= 360.0]
ts = np.array([s.timestamp_ns for s in bo_imu])
dt = np.diff(ts) * 1e-9

gx = np.array([s.gyro[0] for s in bo_imu])
gy = np.array([s.gyro[1] for s in bo_imu])
gz = np.array([s.gyro[2] for s in bo_imu])

int_gx = np.sum(0.5 * (gx[:-1] + gx[1:]) * dt)
int_gy = np.sum(0.5 * (gy[:-1] + gy[1:]) * dt)
int_gz = np.sum(0.5 * (gz[:-1] + gz[1:]) * dt)

print("60s Blackout (300s-360s) Gyro Integrals:")
print(f"  Integral of Gyro X (Roll):  {np.degrees(int_gx):.2f}°")
print(f"  Integral of Gyro Y (Pitch): {np.degrees(int_gy):.2f}°")
print(f"  Integral of Gyro Z (Yaw):   {np.degrees(int_gz):.2f}°")

# Let's check ground truth turn angle
gt_300 = [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= 300.0][-1]
gt_360 = [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= 360.0][-1]
print(f"  GT Start Bearing (300s): {gt_300.bearing_deg:.2f}°")
print(f"  GT End Bearing (360s):   {gt_360.bearing_deg:.2f}°")
print(f"  GT Turn Angle:           {gt_360.bearing_deg - gt_300.bearing_deg:.2f}°")

# What is the 3D phone tilt angle?
accels = np.array([s.accel for s in trip.imu_samples[:500]])
mean_acc = np.mean(accels, axis=0)
print(f"  Gravity Accel: {mean_acc}")
u_g = mean_acc / np.linalg.norm(mean_acc)
print(f"  Gravity Unit Vector: [{u_g[0]:.4f}, {u_g[1]:.4f}, {u_g[2]:.4f}]")

# 3D projected angular turn around Earth's gravity vector:
w_earth_vert = (gx * u_g[0] + gy * u_g[1] + gz * u_g[2])
int_earth_vert = np.sum(0.5 * (w_earth_vert[:-1] + w_earth_vert[1:]) * dt)
print(f"  Turn around Earth's Vertical Axis: {np.degrees(int_earth_vert):.2f}°")
