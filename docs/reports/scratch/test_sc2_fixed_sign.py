import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu
from sih.calibration.mount import MountCalibrator, MountAlignment
from sih.fusion.es_ekf import ErrorStateEKF
from sih.velocity.ai_estimator import AIVelocityEstimator

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(300.0 * 1e9)
bo_end = t0 + int(360.0 * 1e9)

g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])

calib = MountCalibrator()
ekf = ErrorStateEKF()
vel_est = AIVelocityEstimator(checkpoint_path="models/checkpoints/best_velocity_model.pt", device="cuda:0")

# Pre-calibrate
for g in [g for g in trip.gnss_samples if g.timestamp_ns <= bo_start]:
    calib.observe_gnss(g)
for s in [s for s in trip.imu_samples if s.timestamp_ns <= bo_start]:
    calib.update(s)
for g in [g for g in trip.gnss_samples if g.timestamp_ns <= bo_start]:
    calib.observe_gnss(g)
    ekf.update_gnss(g)

# Override yaw_axis_sign to +1.0
calib._alignment = MountAlignment(
    is_calibrated=True,
    R_phone_to_vehicle=calib._alignment.R_phone_to_vehicle,
    forward_axis_phone=calib._alignment.forward_axis_phone,
    lateral_axis_phone=calib._alignment.lateral_axis_phone,
    vertical_axis_phone=calib._alignment.vertical_axis_phone,
    yaw_axis_index=1,
    yaw_axis_sign=1.0,
    mount_yaw_offset_rad=0.0,
    pitch_deg=calib._alignment.pitch_deg,
    roll_deg=calib._alignment.roll_deg,
)

records = []
for s in trip.imu_samples:
    if s.timestamp_ns < bo_start:
        continue
    if s.timestamp_ns > bo_end:
        break
        
    c_s = calib.update(s)
    v_s = vel_est.estimate(c_s)
    st = ekf.predict(c_s, v_s)
    
    gt_e = float(np.interp(s.timestamp_ns, g_ts, enu_all[:, 0]))
    gt_n = float(np.interp(s.timestamp_ns, g_ts, enu_all[:, 1]))
    err = float(np.sqrt((st.position_enu_m[0] - gt_e)**2 + (st.position_enu_m[1] - gt_n)**2))
    records.append({
        "t_s": (s.timestamp_ns - t0)*1e-9,
        "est_e": st.position_enu_m[0], "est_n": st.position_enu_m[1],
        "gt_e": gt_e, "gt_n": gt_n,
        "err": err, "hdg": np.degrees(ekf._heading_rad) % 360.0
    })

df = pd.DataFrame(records)
dists = np.sqrt(np.diff(df["gt_e"])**2 + np.diff(df["gt_n"])**2)
tot_d = np.sum(dists)
final_err = df["err"].iloc[-1]
rmse_err = np.sqrt(np.mean(df["err"]**2))
drift_pct = (final_err / tot_d) * 100.0

print(f"\nScenario 2 (60s blackout at 300s) with yaw_axis_sign = +1.0:")
print(f"  Distance:    {tot_d:.1f} m")
print(f"  Final Error: {final_err:.2f} m")
print(f"  RMSE Error:  {rmse_err:.2f} m")
print(f"  Drift %:     {drift_pct:.2f}%")
print(f"  Start Heading: {df['hdg'].iloc[0]:.1f}° (GT: 242.0°)")
print(f"  End Heading:   {df['hdg'].iloc[-1]:.1f}° (GT: 323.5°)")

# Save plot
plt.figure(figsize=(8, 6))
plt.plot(df["gt_e"], df["gt_n"], "k-", label="Ground Truth GNSS", linewidth=2.5)
plt.plot(df["est_e"], df["est_n"], "r-", label="Estimated (Blackout INS)", linewidth=2.5)
plt.scatter([df["gt_e"].iloc[0]], [df["gt_n"].iloc[0]], color="black", s=100, label="Blackout Start")
plt.scatter([df["gt_e"].iloc[-1]], [df["gt_n"].iloc[-1]], color="blue", s=100, label="GT End")
plt.scatter([df["est_e"].iloc[-1]], [df["est_n"].iloc[-1]], color="red", s=100, label="Estimated End")
plt.title(f"Scenario 2 with Fixed Yaw Sign (Drift: {drift_pct:.2f}%)")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("artifacts/test_sc2_fixed.png", dpi=150)
plt.close()
