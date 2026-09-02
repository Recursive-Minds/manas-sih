import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath("."))
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.data.geo import geodetic_to_enu
from sih.models.tcn_attention import TCNAttentionVelocityModel

ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda:0", weights_only=False)
in_channels = ckpt["norm_mean"].shape[0]
model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

def run_eval(trip_key, start_s, dur_s, name):
    trip = GenericDataLoader().load_file(download_iovnbd_trip(trip_key))
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(start_s * 1e9)
    bo_end_ns = t0_ns + int((start_s + dur_s) * 1e9)
    
    # Auto-detect vehicle yaw axis from pre-blackout GNSS correlation
    g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
    g_brg = np.array([g.bearing_deg for g in trip.gnss_samples])
    g_spd = np.array([g.speed_mps if g.speed_mps is not None else 0.0 for g in trip.gnss_samples])
    
    # GNSS yaw rate before blackout
    g_pre_idx = np.where((g_ts < bo_start_ns) & (g_spd > 2.0))[0]
    dt_g = np.diff(g_ts[g_pre_idx]) * 1e-9
    valid_diff = np.where(dt_g > 0.5)[0]
    
    imu_ts = np.array([s.timestamp_ns for s in trip.imu_samples])
    gx = np.array([s.gyro[0] for s in trip.imu_samples])
    gy = np.array([s.gyro[1] for s in trip.imu_samples])
    gz = np.array([s.gyro[2] for s in trip.imu_samples])
    
    if len(valid_diff) > 2:
        g_brg_unwrap = np.unwrap(np.radians(g_brg[g_pre_idx]))
        g_rates = np.diff(g_brg_unwrap)[valid_diff] / dt_g[valid_diff]
        g_mids = (g_ts[g_pre_idx][valid_diff] + g_ts[g_pre_idx][valid_diff+1]) // 2
        
        gx_int = np.interp(g_mids, imu_ts, gx)
        gy_int = np.interp(g_mids, imu_ts, gy)
        gz_int = np.interp(g_mids, imu_ts, gz)
        
        corrs = {
            "+x": np.corrcoef(g_rates, gx_int)[0, 1],
            "-x": np.corrcoef(g_rates, -gx_int)[0, 1],
            "+y": np.corrcoef(g_rates, gy_int)[0, 1],
            "-y": np.corrcoef(g_rates, -gy_int)[0, 1],
            "+z": np.corrcoef(g_rates, gz_int)[0, 1],
            "-z": np.corrcoef(g_rates, -gz_int)[0, 1],
        }
        best_axis = max(corrs, key=lambda k: corrs[k] if not np.isnan(corrs[k]) else -1)
    else:
        best_axis = "-y"
        
    print(f"[{trip_key}] Auto-detected vehicle yaw axis: {best_axis}")
    
    # Setup GT
    g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
    g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
    g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
    enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])
    
    start_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns >= bo_start_ns][0]
    end_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns <= bo_end_ns][-1]
    
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
        
        # Get vehicle yaw rate along detected axis
        if best_axis == "+x": w_yaw = s.gyro[0]
        elif best_axis == "-x": w_yaw = -s.gyro[0]
        elif best_axis == "+y": w_yaw = s.gyro[1]
        elif best_axis == "-y": w_yaw = -s.gyro[1]
        elif best_axis == "+z": w_yaw = s.gyro[2]
        else: w_yaw = -s.gyro[2]
        
        # In bearing coordinates (CW from North): right turn increases bearing (d(bearing)/dt = w_yaw)
        curr_heading_rad += w_yaw * dt
        
        norm_a = float(np.linalg.norm(s.accel))
        norm_w = float(np.linalg.norm(s.gyro))
        vec = np.array([s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2], norm_a, norm_w], dtype=np.float32)
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
            "err": err, "spd": v_ai,
            "hdg": np.degrees(curr_heading_rad) % 360.0
        })
        
    import pandas as pd
    df = pd.DataFrame(records)
    dists = np.sqrt(np.diff(df["gt_e"])**2 + np.diff(df["gt_n"])**2)
    tot_d = np.sum(dists)
    final_err = df["err"].iloc[-1]
    rmse_err = np.sqrt(np.mean(df["err"]**2))
    drift_pct = (final_err / tot_d) * 100.0
    
    print(f"  Distance:    {tot_d:.1f} m")
    print(f"  Final Error: {final_err:.2f} m")
    print(f"  RMSE Error:  {rmse_err:.2f} m")
    print(f"  Drift %:     {drift_pct:.2f}%")
    
    # Plot Trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(df["gt_e"], df["gt_n"], "k-", label="Ground Truth GNSS", linewidth=2.5)
    plt.plot(df["est_e"], df["est_n"], "r-", label="AI Dead Reckoning", linewidth=2.5)
    plt.scatter([df["gt_e"].iloc[0]], [df["gt_n"].iloc[0]], color="black", s=100, label="Blackout Start")
    plt.scatter([df["gt_e"].iloc[-1]], [df["gt_n"].iloc[-1]], color="blue", s=100, label="GT End")
    plt.scatter([df["est_e"].iloc[-1]], [df["est_n"].iloc[-1]], color="red", s=100, label="Estimated End")
    plt.title(f"{name} (Drift: {drift_pct:.1f}%)")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"artifacts/{name}_perfect_turn.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_eval("S-S1", 120.0, 30.0, "30s_blackout_at_120s")
    run_eval("S-S1", 300.0, 60.0, "60s_blackout_at_300s")
    run_eval("S-S2", 120.0, 30.0, "S-S2_30s_blackout_at_120s")
    run_eval("S-S2", 300.0, 60.0, "S-S2_60s_blackout_at_300s")
