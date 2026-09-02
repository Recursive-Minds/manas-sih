import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu
from sih.models.tcn_attention import TCNAttentionVelocityModel
import torch

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
t0 = trip.imu_samples[0].timestamp_ns

ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda:0", weights_only=False)
model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

# Evaluate pre-blackout speed scale factor between AI speed and GNSS speed (t < 120s)
pre_imu = [s for s in trip.imu_samples if (s.timestamp_ns - t0)*1e-9 <= 120.0]
pre_gnss = [g for g in trip.gnss_samples if (g.timestamp_ns - t0)*1e-9 <= 120.0 and (g.speed_mps or 0) > 2.0]

g_ts = np.array([g.timestamp_ns for g in pre_gnss])
g_v = np.array([g.speed_mps for g in pre_gnss])

# Compute AI speed for pre-blackout
import collections
buf = collections.deque(maxlen=100)
ai_speeds = []
ai_ts = []

for s in pre_imu:
    norm_a = float(np.linalg.norm(s.accel))
    norm_w = float(np.linalg.norm(s.gyro))
    buf.append(np.array([s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2], norm_a, norm_w], dtype=np.float32))
    if len(buf) == 100:
        w = np.array(buf, dtype=np.float32).T
        wn = (w - norm_mean) / norm_std
        xt = torch.from_numpy(wn).unsqueeze(0).float().cuda()
        with torch.no_grad():
            sp, _ = model(xt)
            ai_speeds.append(float(sp[0, 0].item()))
            ai_ts.append(s.timestamp_ns)

ai_speeds = np.array(ai_speeds)
ai_ts = np.array(ai_ts)

# Interp GNSS speed at AI timestamps where speed > 2 m/s
interp_gv = np.interp(ai_ts, g_ts, g_v)
valid_mask = interp_gv > 3.0
scale_factor = np.mean(interp_gv[valid_mask]) / np.mean(ai_speeds[valid_mask])
print(f"Pre-blackout empirical AI speed scale factor (t < 120s): {scale_factor:.4f}")

# Now let's test Dead Reckoning during 30s blackout with this scale factor!
bo_start_ns = t0 + int(120.0 * 1e9)
bo_end_ns = t0 + int(150.0 * 1e9)

g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
all_g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])

start_g = [g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns][-1]
pos = np.array([np.interp(bo_start_ns, all_g_ts, enu_all[:, 0]), np.interp(bo_start_ns, all_g_ts, enu_all[:, 1])])
curr_hdg = np.radians(start_g.bearing_deg)

bo_imu = [s for s in trip.imu_samples if bo_start_ns <= s.timestamp_ns <= bo_end_ns]
last_t = bo_start_ns

records_unscaled = []
records_scaled = []

# Unscaled integration
pos_unscaled = pos.copy()
hdg_unscaled = curr_hdg
buf_u = collections.deque(list(buf), maxlen=100)

for s in bo_imu:
    dt = (s.timestamp_ns - last_t) * 1e-9
    last_t = s.timestamp_ns
    
    # Heading update using Y-gyro (Pitch/Yaw in mount)
    w_z = 1.0 * s.gyro[1]
    hdg_unscaled = (hdg_unscaled - w_z * dt) % (2.0 * np.pi)
    
    norm_a = float(np.linalg.norm(s.accel))
    norm_w = float(np.linalg.norm(s.gyro))
    buf_u.append(np.array([s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2], norm_a, norm_w], dtype=np.float32))
    
    w = np.array(buf_u, dtype=np.float32).T
    wn = (w - norm_mean) / norm_std
    xt = torch.from_numpy(wn).unsqueeze(0).float().cuda()
    with torch.no_grad():
        sp, _ = model(xt)
        v_raw = float(sp[0, 0].item())
        
    pos_unscaled[0] += v_raw * np.sin(hdg_unscaled) * dt
    pos_unscaled[1] += v_raw * np.cos(hdg_unscaled) * dt

# Scaled integration
pos_scaled = pos.copy()
hdg_scaled = curr_hdg
last_t = bo_start_ns
buf_s = collections.deque(list(buf), maxlen=100)

for s in bo_imu:
    dt = (s.timestamp_ns - last_t) * 1e-9
    last_t = s.timestamp_ns
    
    w_z = 1.0 * s.gyro[1]
    hdg_scaled = (hdg_scaled - w_z * dt) % (2.0 * np.pi)
    
    norm_a = float(np.linalg.norm(s.accel))
    norm_w = float(np.linalg.norm(s.gyro))
    buf_s.append(np.array([s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2], norm_a, norm_w], dtype=np.float32))
    
    w = np.array(buf_s, dtype=np.float32).T
    wn = (w - norm_mean) / norm_std
    xt = torch.from_numpy(wn).unsqueeze(0).float().cuda()
    with torch.no_grad():
        sp, _ = model(xt)
        v_scaled = float(sp[0, 0].item()) * scale_factor
        
    pos_scaled[0] += v_scaled * np.sin(hdg_scaled) * dt
    pos_scaled[1] += v_scaled * np.cos(hdg_scaled) * dt

gt_end_e = float(np.interp(bo_end_ns, all_g_ts, enu_all[:, 0]))
gt_end_n = float(np.interp(bo_end_ns, all_g_ts, enu_all[:, 1]))
tot_d = float(np.sqrt((gt_end_e - pos[0])**2 + (gt_end_n - pos[1])**2))

err_unscaled = float(np.sqrt((pos_unscaled[0] - gt_end_e)**2 + (pos_unscaled[1] - gt_end_n)**2))
err_scaled = float(np.sqrt((pos_scaled[0] - gt_end_e)**2 + (pos_scaled[1] - gt_end_n)**2))

print(f"\n30s Blackout Comparison:")
print(f"  Unscaled AI Velocity: Final Error = {err_unscaled:.2f} m ({err_unscaled/tot_d*100:.2f}% drift)")
print(f"  Scaled AI Velocity:   Final Error = {err_scaled:.2f} m ({err_scaled/tot_d*100:.2f}% drift)")
