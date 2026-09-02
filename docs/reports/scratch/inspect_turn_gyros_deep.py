import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(120.0 * 1e9)
bo_end_ns = t0_ns + int(150.0 * 1e9)

# Precompute GT heading
g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
g_brg = np.array([g.bearing_deg for g in trip.gnss_samples])

enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])

# Gravity leveling
accels_all = np.array([s.accel for s in trip.imu_samples[:500]])
g_body = np.mean(accels_all, axis=0)
g_unit = g_body / np.linalg.norm(g_body)
from scipy.spatial.transform import Rotation as R
up = np.array([0.0, 0.0, 1.0])
cross = np.cross(g_unit, up)
dot = float(np.dot(g_unit, up))
cn = np.linalg.norm(cross)
R_level = R.from_rotvec((cross/cn)*np.arctan2(cn, dot)) if cn > 1e-6 else R.identity()

# Check what gyro rates were recorded
samples = [s for s in trip.imu_samples if bo_start_ns <= s.timestamp_ns <= bo_end_ns]
df_turn = pd.DataFrame([{
    "t_s": (s.timestamp_ns - t0_ns) * 1e-9,
    "raw_gx": s.gyro[0], "raw_gy": s.gyro[1], "raw_gz": s.gyro[2],
    "lev_gx": R_level.apply(s.gyro)[0],
    "lev_gy": R_level.apply(s.gyro)[1],
    "lev_gz": R_level.apply(s.gyro)[2],
    "raw_ax": s.accel[0], "raw_ay": s.accel[1], "raw_az": s.accel[2],
    "lev_ax": R_level.apply(s.accel)[0],
    "lev_ay": R_level.apply(s.accel)[1],
    "lev_az": R_level.apply(s.accel)[2],
    "gt_brg": float(np.interp(s.timestamp_ns, g_ts, g_brg)),
    "gt_e": float(np.interp(s.timestamp_ns, g_ts, enu_all[:, 0])),
    "gt_n": float(np.interp(s.timestamp_ns, g_ts, enu_all[:, 1])),
} for s in samples])

print("GT heading at 120s:", df_turn["gt_brg"].iloc[0])
print("GT heading at 150s:", df_turn["gt_brg"].iloc[-1])
print("Change in GT heading:", df_turn["gt_brg"].iloc[-1] - df_turn["gt_brg"].iloc[0], "degrees")

# Integrate each gyro axis
dt = 0.1
int_raw_gz = np.sum(df_turn["raw_gz"]) * dt * (180.0 / np.pi)
int_lev_gz = np.sum(df_turn["lev_gz"]) * dt * (180.0 / np.pi)
int_lev_gx = np.sum(df_turn["lev_gx"]) * dt * (180.0 / np.pi)
int_lev_gy = np.sum(df_turn["lev_gy"]) * dt * (180.0 / np.pi)

print(f"Integrated raw gz: {int_raw_gz:.2f}°")
print(f"Integrated lev gz: {int_lev_gz:.2f}°")
print(f"Integrated lev gx: {int_lev_gx:.2f}°")
print(f"Integrated lev gy: {int_lev_gy:.2f}°")

# Let's inspect turn around 130s-145s
print("\nTurn interval around 135s-145s:")
print(df_turn[(df_turn["t_s"] >= 135.0) & (df_turn["t_s"] <= 145.0)][["t_s", "lev_gx", "lev_gy", "lev_gz", "lev_ay", "gt_brg"]])
