import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.data.geo import geodetic_to_enu
from sih.models.tcn_attention import TCNAttentionVelocityModel

ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda:0", weights_only=False)
in_channels = ckpt["norm_mean"].shape[0]
model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
norm_mean = ckpt["norm_mean"]
norm_std = ckpt["norm_std"]

def eval_blackout(trip_key, start_s, dur_s):
    loader = GenericDataLoader()
    trip = loader.load_file(download_iovnbd_trip(trip_key))
    
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
    bo_start_ns = t0_ns + int(start_s * 1e9)
    bo_end_ns = t0_ns + int((start_s + dur_s) * 1e9)
    
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
    curr_heading_rad = np.radians(last_g.bearing_deg if last_g.bearing_deg is not None else 0.0)
    
    import collections
    buf = collections.deque(maxlen=100)
    for s in trip.imu_samples[max(0, start_idx-100):start_idx]:
        acc_v = R_level.apply(s.accel)
        gyro_v = R_level.apply(s.gyro)
        norm_a = float(np.linalg.norm(acc_v))
        norm_w = float(np.linalg.norm(gyro_v))
        vec = np.array([acc_v[0], acc_v[1], acc_v[2], gyro_v[0], gyro_v[1], gyro_v[2], norm_a, norm_w], dtype=np.float32)
        buf.append(vec)
        
    last_t = bo_start_ns
    records = []
    
    for s in trip.imu_samples[start_idx:end_idx+1]:
        t_curr = s.timestamp_ns
        dt = (t_curr - last_t) * 1e-9
        last_t = t_curr
        
        acc_v = R_level.apply(s.accel)
        gyro_v = R_level.apply(s.gyro)
        
        curr_heading_rad -= gyro_v[2] * dt
        
        norm_a = float(np.linalg.norm(acc_v))
        norm_w = float(np.linalg.norm(gyro_v))
        vec = np.array([acc_v[0], acc_v[1], acc_v[2], gyro_v[0], gyro_v[1], gyro_v[2], norm_a, norm_w], dtype=np.float32)
        buf.append(vec)
        
        w = np.array(buf, dtype=np.float32).T
        a_std = np.mean(np.std(w[:3, -20:], axis=1))
        g_std = np.mean(np.std(w[3:6, -20:], axis=1))
        if a_std < 0.18 and g_std < 0.03:
            v_ai = 0.0
        else:
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
        records.append({"gt_e": gt_e, "gt_n": gt_n, "err": err})
        
    import pandas as pd
    df = pd.DataFrame(records)
    dists = np.sqrt(np.diff(df["gt_e"])**2 + np.diff(df["gt_n"])**2)
    tot_d = np.sum(dists)
    final_err = df["err"].iloc[-1]
    rmse_err = np.sqrt(np.mean(df["err"]**2))
    drift_pct = (final_err / tot_d) * 100.0
    print(f"\n[{trip_key}] Blackout {start_s}s-{start_s+dur_s}s ({dur_s}s):")
    print(f"  Distance:    {tot_d:.1f} m")
    print(f"  Final Error: {final_err:.2f} m")
    print(f"  RMSE Error:  {rmse_err:.2f} m")
    print(f"  Drift %:     {drift_pct:.2f}%")

if __name__ == "__main__":
    eval_blackout("S-S1", 120.0, 30.0)
    eval_blackout("S-S1", 300.0, 60.0)
    eval_blackout("S-S2", 120.0, 30.0)
    eval_blackout("S-S2", 300.0, 60.0)
    eval_blackout("S-M",  120.0, 30.0)
