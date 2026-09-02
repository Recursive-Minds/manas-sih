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

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

t0_ns = trip.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(120.0 * 1e9)
bo_end_ns = t0_ns + int(150.0 * 1e9)

# Load AI model
ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda:0", weights_only=False)
in_channels = ckpt["norm_mean"].shape[0]
model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])

start_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns >= bo_start_ns][0]
end_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns <= bo_end_ns][-1]

# Initial GNSS fix right before blackout
pre_gnss = [g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns]
last_g = pre_gnss[-1]

pos = np.array([np.interp(bo_start_ns, g_ts, enu_all[:, 0]), np.interp(bo_start_ns, g_ts, enu_all[:, 1])])
curr_heading_rad = np.radians(last_g.bearing_deg)

import collections
buf = collections.deque(maxlen=100)
for s in trip.imu_samples[max(0, start_idx-100):start_idx]:
    acc = s.accel
    gyro = s.gyro
    norm_a = float(np.linalg.norm(acc))
    norm_w = float(np.linalg.norm(gyro))
    vec = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2], norm_a, norm_w], dtype=np.float32)
    buf.append(vec)

last_t = bo_start_ns
records = []

for s in trip.imu_samples[start_idx:end_idx+1]:
    t_curr = s.timestamp_ns
    dt = (t_curr - last_t) * 1e-9
    last_t = t_curr
    
    acc = s.accel
    gyro = s.gyro
    
    # In S-S1, the vehicle's yaw turn rate is on -gyro[1] (Pitch)!
    # Heading bearing (CW from North): right turn increases bearing
    # So d(bearing)/dt = -(-gyro[1]) = +gyro[1] or d(bearing)/dt = -gyro_yaw_veh
    # Let's test:
    curr_heading_rad += (-gyro[1]) * dt # right turn increases bearing
    
    norm_a = float(np.linalg.norm(acc))
    norm_w = float(np.linalg.norm(gyro))
    vec = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2], norm_a, norm_w], dtype=np.float32)
    buf.append(vec)
    
    w = np.array(buf, dtype=np.float32).T
    wn = (w - norm_mean) / norm_std
    xt = torch.from_numpy(wn).unsqueeze(0).float().cuda()
    with torch.inference_mode():
        sp, _ = model(xt)
        v_ai = float(sp[0, 0].item())
        
    pos[0] += v_ai * np.sin(curr_heading_rad) * dt
    pos[1] += v_ai * np.cos(curr_heading_rad) * dt
    
    gt_e = float(np.interp(t_curr, g_ts, enu_all[:, 0]))
    gt_n = float(np.interp(t_curr, g_ts, enu_all[:, 1]))
    err = float(np.sqrt((pos[0] - gt_e)**2 + (pos[1] - gt_n)**2))
    records.append({
        "t_s": (t_curr - t0_ns) * 1e-9,
        "est_e": pos[0], "est_n": pos[1],
        "gt_e": gt_e, "gt_n": gt_n,
        "err": err,
        "hdg": np.degrees(curr_heading_rad) % 360.0
    })

df = pd.DataFrame(records)
dists = np.sqrt(np.diff(df["gt_e"])**2 + np.diff(df["gt_n"])**2)
tot_d = np.sum(dists)
final_err = df["err"].iloc[-1]
rmse_err = np.sqrt(np.mean(df["err"]**2))
drift_pct = (final_err / tot_d) * 100.0

print(f"\nRESULTS WITH TRUE VEHICLE YAW AXIS:")
print(f"  Distance:    {tot_d:.1f} m")
print(f"  Start Error: {df['err'].iloc[0]:.2f} m")
print(f"  Final Error: {final_err:.2f} m")
print(f"  Max Error:   {df['err'].max():.2f} m")
print(f"  RMSE Error:  {rmse_err:.2f} m")
print(f"  Drift %:     {drift_pct:.2f}%")
print(f"  Final Estimated Heading: {df['hdg'].iloc[-1]:.1f}° (GT: 315.8°)")
