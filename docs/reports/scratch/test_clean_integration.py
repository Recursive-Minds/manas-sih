import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

def test_heading_propagation():
    # Precompute leveling
    accels_all = np.array([s.accel for s in trip.imu_samples[:500]])
    g_body = np.mean(accels_all, axis=0)
    g_unit = g_body / np.linalg.norm(g_body)
    up = np.array([0.0, 0.0, 1.0])
    cross = np.cross(g_unit, up)
    dot = float(np.dot(g_unit, up))
    cn = np.linalg.norm(cross)
    R_level = R.from_rotvec((cross/cn)*np.arctan2(cn, dot)) if cn > 1e-6 else R.identity()
    
    # Track heading by integrating leveled gyro_z
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(120.0 * 1e9)
    bo_end_ns = t0_ns + int(150.0 * 1e9)
    
    # Initial GNSS heading at t=120s
    gnss_120 = [g for g in trip.gnss_samples if abs((g.timestamp_ns - bo_start_ns)*1e-9) < 1.0][0]
    print(f"GNSS @ 120s: speed={gnss_120.speed_mps} m/s, COG bearing={gnss_120.bearing_deg:.1f}°")
    
    # Let's integrate forward velocity with integrated heading from 120s to 150s
    import torch
    from sih.models.tcn_attention import TCNAttentionVelocityModel
    ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda:0", weights_only=False)
    in_channels = ckpt["norm_mean"].shape[0]
    model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]
    
    from sih.data.geo import geodetic_to_enu
    g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
    g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
    g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
    g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
    enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])
    
    start_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns >= bo_start_ns][0]
    end_idx = [i for i, s in enumerate(trip.imu_samples) if s.timestamp_ns <= bo_end_ns][-1]
    
    pos = np.array([np.interp(bo_start_ns, g_ts, enu_all[:, 0]), np.interp(bo_start_ns, g_ts, enu_all[:, 1])])
    curr_heading_rad = np.radians(gnss_120.bearing_deg)
    
    last_t = bo_start_ns
    import collections
    buf = collections.deque(maxlen=100)
    
    # Pre-fill buffer before 120s
    for s in trip.imu_samples[start_idx-100:start_idx]:
        acc_v = R_level.apply(s.accel)
        gyro_v = R_level.apply(s.gyro)
        norm_a = float(np.linalg.norm(acc_v))
        norm_w = float(np.linalg.norm(gyro_v))
        vec = np.array([acc_v[0], acc_v[1], acc_v[2], gyro_v[0], gyro_v[1], gyro_v[2], norm_a, norm_w], dtype=np.float32)
        buf.append(vec)
        
    records = []
    for s in trip.imu_samples[start_idx:end_idx+1]:
        t_curr = s.timestamp_ns
        dt = (t_curr - last_t) * 1e-9
        last_t = t_curr
        
        acc_v = R_level.apply(s.accel)
        gyro_v = R_level.apply(s.gyro)
        
        # Heading integration: in ENU (bearing clockwise from North): d(bearing)/dt = -gyro_z
        # (gyro_z > 0 means CCW turn / left turn, so bearing decreases)
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
                
        # Move along heading
        pos[0] += v_ai * np.sin(curr_heading_rad) * dt # East
        pos[1] += v_ai * np.cos(curr_heading_rad) * dt # North
        
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
    drift_pct = (final_err / tot_d) * 100.0
    print(f"\nRESULTS FOR CLEAN INTEGRATION:")
    print(f"  Distance:    {tot_d:.1f} m")
    print(f"  Start Error: {df['err'].iloc[0]:.2f} m")
    print(f"  Final Error: {final_err:.2f} m")
    print(f"  RMSE Error:  {np.sqrt(np.mean(df['err']**2)):.2f} m")
    print(f"  Drift %:     {drift_pct:.2f}%")

if __name__ == "__main__":
    test_heading_propagation()
