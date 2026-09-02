import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu

loader = GenericDataLoader()
trip = loader.load_file("data/raw/iovnbd_trips/S-S1.csv")

t0 = trip.imu_samples[0].timestamp_ns
bo_start = t0 + int(120 * 1e9)
bo_end = bo_start + int(30 * 1e9)

gnss_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
gnss_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
gnss_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
gnss_alts = np.array([g.altitude_m for g in trip.gnss_samples])
gnss_enu = geodetic_to_enu(gnss_lats, gnss_lons, gnss_alts, trip.reference_lat_deg, trip.reference_lon_deg, trip.reference_alt_m)

# Let's run a clean 2D/3D Kinematic AI-Dead-Reckoning filter:
# 1. Yaw tracked by integrating Gyro Z with calibrated gyro bias (estimated at stationary / pre-blackout).
# 2. Forward speed from AI velocity estimator (or pre-blackout velocity / accelerometer).
# 3. Heading initialized from GNSS course-over-ground.

# Pre-blackout gyro bias calibration:
pre_bo_imus = [s for s in trip.imu_samples if s.timestamp_ns < bo_start]
# Estimate gyro bias during stationary or straight driving
# Stationary check: accel norm close to 9.81 and low std
stationary_gyros = []
for i in range(len(pre_bo_imus) - 10):
    w = pre_bo_imus[i:i+10]
    acc_std = np.std([s.accel for s in w], axis=0)
    if np.mean(acc_std) < 0.08:
        stationary_gyros.append(pre_bo_imus[i].gyro)

bg_z = np.median([g[2] for g in stationary_gyros]) if stationary_gyros else 0.0
print(f"Estimated Gyro Z Bias: {bg_z:.6f} rad/s ({np.degrees(bg_z):.3f} deg/s)")

# Load AI model
import torch
from sih.models.tcn_attention import TCNAttentionVelocityModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location=device, weights_only=False)
model = TCNAttentionVelocityModel().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

# Simulate blackout
pos = None
heading_rad = None
last_t = None
records = []

# Buffer for AI velocity
buf = []

for imu in trip.imu_samples:
    t = imu.timestamp_ns
    vec = np.hstack([imu.accel, imu.gyro]).astype(np.float32)
    buf.append(vec)
    if len(buf) > 100:
        buf.pop(0)

    # GNSS updates before blackout
    if t < bo_start:
        # Find latest GNSS fix
        idx = np.searchsorted(gnss_ts, t)
        if idx < len(gnss_ts) and abs(gnss_ts[idx] - t) < 1e8:
            g = trip.gnss_samples[idx]
            pos = np.array([gnss_enu[idx, 0], gnss_enu[idx, 1]])
            if g.bearing_deg is not None:
                heading_rad = np.radians(g.bearing_deg)
        last_t = t
        continue

    if t > bo_end:
        break

    dt = (t - last_t) * 1e-9
    last_t = t

    # Propagate heading with Gyro Z:
    # Bearing (clockwise from North): d(heading)/dt = -gyro_z (for right-hand coords where Z points Up)
    gz = imu.gyro[2] - bg_z
    heading_rad = (heading_rad - gz * dt) % (2 * np.pi)

    # Get AI velocity:
    if len(buf) == 100:
        w_arr = np.array(buf, dtype=np.float32).T
        w_norm = (w_arr - norm_mean) / norm_std
        x_tensor = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)
        with torch.no_grad():
            speed_pred, _ = model(x_tensor)
            v = float(speed_pred[0, 0].item())
    else:
        v = 15.0

    # Propagate position:
    # East = v * sin(heading), North = v * cos(heading)
    pos[0] += v * np.sin(heading_rad) * dt
    pos[1] += v * np.cos(heading_rad) * dt

    # GT
    gt_e = np.interp(t, gnss_ts, gnss_enu[:, 0])
    gt_n = np.interp(t, gnss_ts, gnss_enu[:, 1])
    err = np.sqrt((pos[0] - gt_e)**2 + (pos[1] - gt_n)**2)
    records.append({
        "time_s": (t - t0) * 1e-9,
        "est_e": pos[0], "est_n": pos[1],
        "gt_e": gt_e, "gt_n": gt_n,
        "error_m": err,
        "heading_deg": np.degrees(heading_rad),
        "ai_speed": v
    })

df = pd.DataFrame(records)
gt_d = np.sum(np.sqrt(np.diff(df["gt_e"])**2 + np.diff(df["gt_n"])**2))
final_err = df["error_m"].iloc[-1]
drift_pct = (final_err / gt_d) * 100.0
rmse_err = np.sqrt(np.mean(df["error_m"]**2))

print(f"\n================ CLEAN AI DEAD RECKONING RESULTS ================")
print(f"  Distance Travelled:   {gt_d:.1f} m")
print(f"  Final Position Error: {final_err:.2f} m")
print(f"  Max Position Error:   {df['error_m'].max():.2f} m")
print(f"  RMSE Position Error:  {rmse_err:.2f} m")
print(f"  Drift Percentage:     {drift_pct:.2f}%")
print(f"  Passed (<10%):        {drift_pct < 10.0}")
