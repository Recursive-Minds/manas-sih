import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import CalibratedSample, VelocityEstimate
from scipy.spatial.transform import Rotation as R

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

ekf = ErrorStateEKF(accel_noise_std=0.2, gyro_noise_std=0.015, enable_nhc=True)
ekf.init_from_gnss(trip.gnss_samples[0])

# Gravity leveling
accels_all = np.array([s.accel for s in trip.imu_samples[:500]])
g_body = np.mean(accels_all, axis=0)
g_unit = g_body / np.linalg.norm(g_body)
up = np.array([0.0, 0.0, 1.0])
cross = np.cross(g_unit, up)
dot = float(np.dot(g_unit, up))
cn = np.linalg.norm(cross)
R_level = R.from_rotvec((cross/cn)*np.arctan2(cn, dot)) if cn > 1e-6 else R.identity()

t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(120.0 * 1e9)
bo_end_ns = t0_ns + int(150.0 * 1e9)

from sih.data.geo import geodetic_to_enu
g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])

gnss_idx = 0
records = []

for imu in trip.imu_samples:
    t_curr = imu.timestamp_ns
    if t_curr > bo_end_ns + int(2e9):
        break
        
    while gnss_idx < len(trip.gnss_samples) and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
        g = trip.gnss_samples[gnss_idx]
        in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
        if not in_bo:
            dt_gap = (g.timestamp_ns - (ekf._last_gnss_ts or g.timestamp_ns)) * 1e-9
            if dt_gap > 0.5:
                ekf._P[0:3, 0:3] += np.eye(3) * (dt_gap * 5.0)**2
                ekf._P[3:6, 3:6] += np.eye(3) * (dt_gap * 1.0)**2
                ekf._P[6:9, 6:9] += np.eye(3) * (0.1)**2
            ekf.update_gnss(g)
        gnss_idx += 1
        
    acc_v = R_level.apply(imu.accel)
    gyro_v = R_level.apply(imu.gyro)
    calib = CalibratedSample(
        timestamp_ns=t_curr,
        accel_vehicle=acc_v,
        gyro_vehicle=gyro_v,
        rotation_body_to_vehicle=R_level.as_matrix(),
        gravity_vehicle=np.array([0.0, 0.0, 9.80665]),
        is_calibrated=True,
    )
    
    st = ekf.predict(calib)
    
    if bo_start_ns <= t_curr <= bo_end_ns:
        gt_e = float(np.interp(t_curr, g_ts, enu_all[:, 0]))
        gt_n = float(np.interp(t_curr, g_ts, enu_all[:, 1]))
        records.append({
            "t_s": (t_curr - t0_ns) * 1e-9,
            "est_e": st.position_enu_m[0],
            "est_n": st.position_enu_m[1],
            "gt_e": gt_e,
            "gt_n": gt_n,
            "v_e": st.velocity_enu_mps[0],
            "v_n": st.velocity_enu_mps[1],
            "hdg": np.degrees(st.heading_rad),
            "bg": ekf._bg * (180.0 / np.pi), # deg/s
        })

df = pd.DataFrame(records)
print("First 5 steps of blackout:")
for idx, row in df.iloc[:5].iterrows():
    print(f"t={row['t_s']:.2f}s | Est=({row['est_e']:.1f}, {row['est_n']:.1f}) | GT=({row['gt_e']:.1f}, {row['gt_n']:.1f}) | Vel=({row['v_e']:.2f}, {row['v_n']:.2f}) | Hdg={row['hdg']:.1f}° | bg_z={row['bg'][2]:.3f}°/s")

print("\nLast 5 steps of blackout:")
for idx, row in df.iloc[-5:].iterrows():
    print(f"t={row['t_s']:.2f}s | Est=({row['est_e']:.1f}, {row['est_n']:.1f}) | GT=({row['gt_e']:.1f}, {row['gt_n']:.1f}) | Vel=({row['v_e']:.2f}, {row['v_n']:.2f}) | Hdg={row['hdg']:.1f}° | bg_z={row['bg'][2]:.3f}°/s")
