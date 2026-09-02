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

ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda:0", weights_only=False)
in_channels = ckpt["norm_mean"].shape[0]
model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

def eval_projection(trip_name, bo_start_s, bo_dur_s):
    trip = GenericDataLoader().load_file(download_iovnbd_trip(trip_name))
    t0 = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0 + int(bo_start_s * 1e9)
    bo_end_ns = t0 + int((bo_start_s + bo_dur_s) * 1e9)
    
    # Pre-blackout IMU for gravity vector
    accels_pre = np.array([s.accel for s in trip.imu_samples if s.timestamp_ns <= bo_start_ns])
    g_vec = np.mean(accels_pre[:min(len(accels_pre), 500)], axis=0)
    u_z = g_vec / np.linalg.norm(g_vec)
    
    # Ground truth
    g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
    g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
    g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
    g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
    enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])
    
    # Last GNSS fix before blackout
    pre_g = [g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns]
    last_g = pre_g[-1]
    
    # Let's test the sign of projection with pre-blackout GNSS turns
    g_brg_unwrap = np.unwrap(np.radians(np.array([g.bearing_deg for g in pre_g])))
    g_ts_arr = np.array([g.timestamp_ns for g in pre_g])
    d_theta_gnss = np.diff(g_brg_unwrap)
    
    imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples if s.timestamp_ns <= bo_start_ns])
    gyros_all = np.array([s.gyro for s in trip.imu_samples if s.timestamp_ns <= bo_start_ns])
    
    # 3D projected yaw rate
    w_proj = gyros_all @ u_z
    
    d_theta_proj = []
    for i in range(len(d_theta_gnss)):
        t1 = g_ts_arr[i]
        t2 = g_ts_arr[i+1]
        mask = (imu_ts >= t1) & (imu_ts <= t2)
        if np.sum(mask) > 1:
            dt_imu = np.diff(imu_ts[mask]) * 1e-9
            d_theta_proj.append(np.sum(0.5 * (w_proj[mask][:-1] + w_proj[mask][1:]) * dt_imu))
        else:
            d_theta_proj.append(0.0)
            
    corr = np.corrcoef(d_theta_gnss, d_theta_proj)[0, 1]
    # In right-handed ENU (where d(heading)/dt = -w_z):
    # If corr < 0, turning right (d_theta > 0) has negative w_proj, so w_z = +w_proj (sign = +1)
    # If corr > 0, turning right (d_theta > 0) has positive w_proj, so w_z = -w_proj (sign = -1)
    proj_sign = 1.0 if corr < 0 else -1.0
    
    print(f"[{trip_name} @ {bo_start_s}s] Gravity u_z: [{u_z[0]:.3f}, {u_z[1]:.3f}, {u_z[2]:.3f}], Corr: {corr:.3f}, Sign: {proj_sign}")
    
    # Run Dead Reckoning
    pos = np.array([np.interp(bo_start_ns, g_ts, enu_all[:, 0]), np.interp(bo_start_ns, g_ts, enu_all[:, 1])])
    curr_heading_rad = np.radians(last_g.bearing_deg)
    
    start_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns >= bo_start_ns][0]
    end_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns <= bo_end_ns][-1]
    
    import collections
    buf = collections.deque(maxlen=100)
    for s in trip.imu_samples[max(0, start_idx-100):start_idx]:
        norm_a = float(np.linalg.norm(s.accel))
        norm_w = float(np.linalg.norm(s.gyro))
        buf.append(np.array([s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2], norm_a, norm_w], dtype=np.float32))
        
    last_t = bo_start_ns
    records = []
    
    for s in trip.imu_samples[start_idx:end_idx+1]:
        t_curr = s.timestamp_ns
        dt = (t_curr - last_t) * 1e-9
        last_t = t_curr
        
        # 3D projected yaw rate
        w_z = proj_sign * float(np.dot(u_z, s.gyro))
        # d(heading)/dt = -w_z
        curr_heading_rad = (curr_heading_rad - w_z * dt) % (2.0 * np.pi)
        
        norm_a = float(np.linalg.norm(s.accel))
        norm_w = float(np.linalg.norm(s.gyro))
        buf.append(np.array([s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2], norm_a, norm_w], dtype=np.float32))
        
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
            "t_s": (t_curr - t0)*1e-9, "est_e": pos[0], "est_n": pos[1],
            "gt_e": gt_e, "gt_n": gt_n, "err": err, "hdg": np.degrees(curr_heading_rad) % 360.0
        })
        
    df = pd.DataFrame(records)
    dists = np.sqrt(np.diff(df["gt_e"])**2 + np.diff(df["gt_n"])**2)
    tot_d = np.sum(dists)
    final_err = df["err"].iloc[-1]
    drift_pct = (final_err / tot_d) * 100.0
    print(f"  Distance: {tot_d:.1f} m | Final Error: {final_err:.2f} m | Drift: {drift_pct:.2f}% | Start Hdg: {df['hdg'].iloc[0]:.1f}° | End Hdg: {df['hdg'].iloc[-1]:.1f}°\n")

print("=== EVALUATION ON TRAINING TRIP S-S1 ===")
eval_projection("S-S1", 120.0, 30.0)
eval_projection("S-S1", 300.0, 60.0)

print("=== ZERO-LEAKAGE EVALUATION ON UNSEEN VALIDATION TRIP S-S2 ===")
eval_projection("S-S2", 120.0, 30.0)
eval_projection("S-S2", 300.0, 60.0)
