import os
import sys
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

raw_csv = download_iovnbd_trip("S-S1")
df_raw = pd.read_csv(raw_csv, encoding="latin-1")
orient_cols = [c for c in df_raw.columns if "orient" in c.lower() and "gps" not in c.lower()]
print("Orientation columns:", orient_cols)

pitch_deg = float(df_raw[orient_cols[1]].iloc[:500].mean())
roll_deg = float(df_raw[orient_cols[2]].iloc[:500].mean())
print(f"Mean Pitch: {pitch_deg:.2f}°, Mean Roll: {roll_deg:.2f}°")

# 3D Leveled Rotation Matrix (Pitch around X/Y and Roll)
# Android coordinate system: Pitch is around X-axis, Roll is around Y-axis
R_mount = R.from_euler("xyz", [pitch_deg, roll_deg, 0.0], degrees=True)

bo_imu = [s for s in trip.imu_samples if 300.0 <= (s.timestamp_ns - t0)*1e-9 <= 360.0]
ts = np.array([s.timestamp_ns for s in bo_imu])
dt = np.diff(ts) * 1e-9

gyros_phone = np.array([s.gyro for s in bo_imu])
gyros_veh = R_mount.apply(gyros_phone)

int_x = np.sum(0.5 * (gyros_veh[:-1, 0] + gyros_veh[1:, 0]) * dt)
int_y = np.sum(0.5 * (gyros_veh[:-1, 1] + gyros_veh[1:, 1]) * dt)
int_z = np.sum(0.5 * (gyros_veh[:-1, 2] + gyros_veh[1:, 2]) * dt)

print(f"Vehicle Frame Integrals: X={np.degrees(int_x):.2f}°, Y={np.degrees(int_y):.2f}°, Z={np.degrees(int_z):.2f}°")
