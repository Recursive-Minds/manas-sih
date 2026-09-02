import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu
from sih.models.tcn_attention import TCNAttentionVelocityModel

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

# Compute true COGs
cogs = np.zeros(len(gnss_ts))
for i in range(len(gnss_ts) - 1):
    de = gnss_enu[i+1, 0] - gnss_enu[i, 0]
    dn = gnss_enu[i+1, 1] - gnss_enu[i, 1]
    dist = np.sqrt(de**2 + dn**2)
    if dist > 3.0:
        cogs[i] = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
    else:
        cogs[i] = cogs[i-1] if i > 0 else 0.0
cogs[-1] = cogs[-2]

idx_before = np.searchsorted(gnss_ts, bo_start) - 1
start_pos = gnss_enu[idx_before, :2].copy()
start_heading_deg = cogs[idx_before]

# Estimate gravity vector and gyro bias pre-blackout
pre_bo = [s for s in trip.imu_samples if s.timestamp_ns < bo_start]
acc_mean = np.mean([s.accel for s in pre_bo[-500:]], axis=0)
g_unit = acc_mean / np.linalg.norm(acc_mean)

# Gyro bias projected onto gravity
bg_yaw = np.median([np.dot(s.gyro, g_unit) for s in pre_bo[-500:]])
print(f"Estimated Gravity unit vector: {g_unit}")
print(f"Estimated Yaw Gyro Bias: {bg_yaw:.6f} rad/s")

# Load AI model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location=device, weights_only=False)
model = TCNAttentionVelocityModel().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

# Evaluate Blackout Dead Reckoning
pos = start_pos.copy()
heading_rad = np.radians(start_heading_deg)
buf = []
records = []
last_t = None

for imu in trip.imu_samples:
    t = imu.timestamp_ns
    vec = np.hstack([imu.accel, imu.gyro]).astype(np.float32)
    buf.append(vec)
    if len(buf) > 100:
        buf.pop(0)

    if t < bo_start:
        last_t = t
        continue
    if t > bo_end:
        break

    dt = (t - last_t) * 1e-9
    last_t = t

    # 3D Gyro projected onto true vertical gravity axis
    w_vertical = np.dot(imu.gyro, g_unit) - bg_yaw
    heading_rad = (heading_rad - w_vertical * dt) % (2 * np.pi)

    # AI Speed
    if len(buf) == 100:
        w_arr = np.array(buf, dtype=np.float32).T
        w_norm = (w_arr - norm_mean) / norm_std
        x_tensor = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)
        with torch.no_grad():
            speed_pred, _ = model(x_tensor)
            v = float(speed_pred[0, 0].item())
    else:
        v = 15.0

    pos[0] += v * np.sin(heading_rad) * dt
    pos[1] += v * np.cos(heading_rad) * dt

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

print(f"\n================ 3D GRAVITY-PROJECTED YAW RESULTS ================")
print(f"  Distance Travelled:   {gt_d:.1f} m")
print(f"  Final Position Error: {final_err:.2f} m")
print(f"  Max Position Error:   {df['error_m'].max():.2f} m")
print(f"  RMSE Position Error:  {rmse_err:.2f} m")
print(f"  Drift Percentage:     {drift_pct:.2f}%")
