import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

def run_calibrated_dr(bo_start_s, bo_dur_s, name):
    bo_start_ns = t0 + int(bo_start_s * 1e9)
    bo_end_ns = t0 + int((bo_start_s + bo_dur_s) * 1e9)
    
    # 1. Pre-blackout online calibration of Speed Scale and Tilt Cosine factor (t <= bo_start_s)
    pre_imu = [s for s in trip.imu_samples if s.timestamp_ns <= bo_start_ns]
    pre_gnss = [g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns and (g.speed_mps or 0) > 3.0]
    
    g_ts = np.array([g.timestamp_ns for g in pre_gnss])
    g_brg = np.unwrap(np.radians(np.array([g.bearing_deg for g in pre_gnss])))
    d_theta_g = np.diff(g_brg)
    
    imu_ts_arr = np.array([s.timestamp_ns for s in pre_imu])
    gy_arr = np.array([s.gyro[1] for s in pre_imu])
    
    d_theta_gyro = []
    for i in range(len(d_theta_g)):
        t1 = g_ts[i]
        t2 = g_ts[i+1]
        mask = (imu_ts_arr >= t1) & (imu_ts_arr <= t2)
        if np.sum(mask) > 1:
            dt_imu = np.diff(imu_ts_arr[mask]) * 1e-9
            d_theta_gyro.append(np.sum(0.5 * (gy_arr[mask][:-1] + gy_arr[mask][1:]) * dt_imu))
        else:
            d_theta_gyro.append(0.0)
            
    d_theta_gyro = np.array(d_theta_gyro)
    turns = (np.abs(d_theta_g) > np.radians(3.0)) & (np.abs(d_theta_gyro) > np.radians(2.0))
    if np.sum(turns) >= 2:
        # Scale factor between GNSS angular turn and phone gyro integral
        turn_scale = float(np.median(np.abs(d_theta_g[turns]) / np.abs(d_theta_gyro[turns])))
    else:
        turn_scale = 1.35
        
    print(f"[{name}] Pre-blackout calibrated turn scale factor: {turn_scale:.4f}")
    
    # Speed scale
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
                
    g_v = np.array([g.speed_mps for g in pre_gnss])
    interp_gv = np.interp(ai_ts, g_ts, g_v)
    valid_mask = interp_gv > 3.0
    speed_scale = float(np.median(interp_gv[valid_mask] / np.maximum(ai_speeds, 0.1)[valid_mask]))
    print(f"[{name}] Pre-blackout calibrated speed scale factor: {speed_scale:.4f}")
    
    # 2. Integrate during blackout
    g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
    g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
    g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
    all_g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
    enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])
    
    start_g = [g for g in trip.gnss_samples if g.timestamp_ns <= bo_start_ns][-1]
    start_e = float(np.interp(bo_start_ns, all_g_ts, enu_all[:, 0]))
    start_n = float(np.interp(bo_start_ns, all_g_ts, enu_all[:, 1]))
    
    pos = np.array([start_e, start_n])
    hdg = np.radians(start_g.bearing_deg)
    
    bo_imu = [s for s in trip.imu_samples if bo_start_ns <= s.timestamp_ns <= bo_end_ns]
    last_t = bo_start_ns
    
    traj_e = [pos[0]]
    traj_n = [pos[1]]
    gt_e_list = [pos[0]]
    gt_n_list = [pos[1]]
    
    for s in bo_imu:
        dt = (s.timestamp_ns - last_t) * 1e-9
        last_t = s.timestamp_ns
        
        # Turn-scale corrected yaw rate (w_z = turn_scale * gy)
        w_z = turn_scale * float(s.gyro[1])
        hdg = (hdg - w_z * dt) % (2.0 * np.pi)
        
        norm_a = float(np.linalg.norm(s.accel))
        norm_w = float(np.linalg.norm(s.gyro))
        buf.append(np.array([s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2], norm_a, norm_w], dtype=np.float32))
        
        w = np.array(buf, dtype=np.float32).T
        wn = (w - norm_mean) / norm_std
        xt = torch.from_numpy(wn).unsqueeze(0).float().cuda()
        with torch.no_grad():
            sp, _ = model(xt)
            v = float(sp[0, 0].item()) * speed_scale
            
        pos[0] += v * np.sin(hdg) * dt
        pos[1] += v * np.cos(hdg) * dt
        
        gt_curr_e = float(np.interp(s.timestamp_ns, all_g_ts, enu_all[:, 0]))
        gt_curr_n = float(np.interp(s.timestamp_ns, all_g_ts, enu_all[:, 1]))
        
        traj_e.append(pos[0])
        traj_n.append(pos[1])
        gt_e_list.append(gt_curr_e)
        gt_n_list.append(gt_curr_n)
        
    gt_end_e = gt_e_list[-1]
    gt_end_n = gt_n_list[-1]
    
    dist_travelled = np.sum(np.sqrt(np.diff(gt_e_list)**2 + np.diff(gt_n_list)**2))
    final_err = float(np.sqrt((pos[0] - gt_end_e)**2 + (pos[1] - gt_end_n)**2))
    drift_pct = (final_err / dist_travelled) * 100.0
    
    print(f"  --> RESULT: Distance = {dist_travelled:.1f} m | Final Error = {final_err:.2f} m | Drift = {drift_pct:.2f}% | Target (<10%): {'PASSED' if drift_pct < 10.0 else 'FAILED'}\n")
    
    # Plot comparison
    plt.figure(figsize=(9, 7))
    plt.plot(gt_e_list, gt_n_list, 'b-', label='Ground Truth (Blue Path)', lw=3)
    plt.plot(traj_e, traj_n, 'r--', label='Estimated Path (Red Path)', lw=2.5)
    plt.plot(start_e, start_n, 'go', markersize=12, label='Blackout Start')
    plt.plot(gt_end_e, gt_end_n, 'bo', markersize=14, label=f'GT End (Blue Dot): ({gt_end_e:.1f}, {gt_end_n:.1f})')
    plt.plot(pos[0], pos[1], 'ro', markersize=14, label=f'Est End (Red Dot): ({pos[0]:.1f}, {pos[1]:.1f})')
    plt.title(f"{name}: True Kinematic Alignment (Final Error = {final_err:.1f} m, Drift = {drift_pct:.2f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.axis("equal")
    plt.savefig(f"artifacts/{name}_true_kinematic.png", dpi=150)
    plt.close()

run_calibrated_dr(120.0, 30.0, "30s_blackout_at_120s")
run_calibrated_dr(300.0, 60.0, "60s_blackout_at_300s")
