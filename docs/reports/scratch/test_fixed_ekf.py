import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from sih.data.loader import GenericDataLoader
from sih.data.downloader import download_iovnbd_trip
from sih.fusion.es_ekf import ErrorStateEKF, exact_alignment
from sih.core.contracts import CalibratedSample, VelocityEstimate

trip = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))

def test_fixed_ekf():
    # Let's run from t=0 to t=150s (30s blackout at 120s)
    ekf = ErrorStateEKF(
        accel_noise_std=0.2,
        gyro_noise_std=0.015,
        gnss_pos_std=1.0,
        gnss_vel_std=0.1,
        gnss_heading_std=0.02,
        enable_nhc=True,
    )
    ekf.init_from_gnss(trip.gnss_samples[0])
    
    # Load AI model
    import torch
    from sih.models.tcn_attention import TCNAttentionVelocityModel
    ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda:0", weights_only=False)
    in_channels = ckpt["norm_mean"].shape[0]
    model = TCNAttentionVelocityModel(in_channels=in_channels, base_channels=32, num_attention_heads=4).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]
    
    # Gravity leveling rotation
    accels_all = np.array([s.accel for s in trip.imu_samples[:500]])
    g_body = np.mean(accels_all, axis=0)
    g_unit = g_body / np.linalg.norm(g_body)
    from scipy.spatial.transform import Rotation as R
    up = np.array([0.0, 0.0, 1.0])
    cross = np.cross(g_unit, up)
    dot = float(np.dot(g_unit, up))
    cn = np.linalg.norm(cross)
    R_level = R.from_rotvec((cross/cn)*np.arctan2(cn, dot)) if cn > 1e-6 else R.identity()
    
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(120.0 * 1e9)
    bo_end_ns = t0_ns + int(150.0 * 1e9)
    
    # Precompute GT
    from sih.data.geo import geodetic_to_enu
    g_lats = np.array([g.latitude_deg for g in trip.gnss_samples])
    g_lons = np.array([g.longitude_deg for g in trip.gnss_samples])
    g_alts = np.array([g.altitude_m for g in trip.gnss_samples])
    g_ts = np.array([g.timestamp_ns for g in trip.gnss_samples])
    enu_all = geodetic_to_enu(g_lats, g_lons, g_alts, g_lats[0], g_lons[0], g_alts[0])
    
    gnss_idx = 0
    import collections
    buf = collections.deque(maxlen=100)
    
    records = []
    
    for imu in trip.imu_samples:
        t_curr = imu.timestamp_ns
        if t_curr > bo_end_ns + int(2e9):
            break
            
        # Process GNSS
        while gnss_idx < len(trip.gnss_samples) and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
            if not in_bo:
                # Direct measurement update with appropriate process noise growth
                dt_gap = (g.timestamp_ns - (ekf._last_gnss_ts or g.timestamp_ns)) * 1e-9
                if dt_gap > 0.5:
                    ekf._P[0:3, 0:3] += np.eye(3) * (dt_gap * 5.0)**2
                    ekf._P[3:6, 3:6] += np.eye(3) * (dt_gap * 1.0)**2
                    ekf._P[6:9, 6:9] += np.eye(3) * (0.1)**2
                ekf.update_gnss(g)
            gnss_idx += 1
            
        # Calibrate IMU
        acc_v = R_level.apply(imu.accel)
        gyro_v = R_level.apply(imu.gyro)
        calib = CalibratedSample(
            timestamp_ns=t_curr,
            accel_vehicle=acc_v,
            gyro_vehicle=gyro_v,
            rotation_body_to_vehicle=R_level.as_matrix(),
            gravity_vehicle=np.array([0.0, 0.0, 9.80665]),
            is_calibrated=True,
        )
        
        # AI Velocity
        norm_a = float(np.linalg.norm(acc_v))
        norm_w = float(np.linalg.norm(gyro_v))
        vec = np.array([acc_v[0], acc_v[1], acc_v[2], gyro_v[0], gyro_v[1], gyro_v[2], norm_a, norm_w], dtype=np.float32)
        buf.append(vec)
        
        if len(buf) == 100:
            w = np.array(buf, dtype=np.float32).T
            # Check stationary
            a_std = np.mean(np.std(w[:3, -20:], axis=1))
            g_std = np.mean(np.std(w[3:6, -20:], axis=1))
            if a_std < 0.18 and g_std < 0.03:
                vel_est = VelocityEstimate(t_curr, 0.0, 0.01, "STATIONARY")
            else:
                wn = (w - norm_mean) / norm_std
                xt = torch.from_numpy(wn).unsqueeze(0).float().cuda()
                with torch.inference_mode():
                    sp, lv = model(xt)
                    vel_est = VelocityEstimate(t_curr, float(sp[0, 0].item()), float(torch.exp(lv[0, 0]).item()), "DRIVING")
        else:
            vel_est = VelocityEstimate(t_curr, 0.0, 5.0, "INITIALIZING")
            
        st = ekf.predict(calib, vel_est)
        
        gt_e = np.interp(t_curr, g_ts, enu_all[:, 0])
        gt_n = np.interp(t_curr, g_ts, enu_all[:, 1])
        err = np.sqrt((st.position_enu_m[0] - gt_e)**2 + (st.position_enu_m[1] - gt_n)**2)
        in_blackout = (bo_start_ns <= t_curr <= bo_end_ns)
        records.append({
            "t_s": (t_curr - t0_ns) * 1e-9,
            "est_e": st.position_enu_m[0],
            "est_n": st.position_enu_m[1],
            "gt_e": gt_e,
            "gt_n": gt_n,
            "err": err,
            "spd": vel_est.forward_speed_mps,
            "hdg": np.degrees(st.heading_rad),
            "in_bo": in_blackout
        })

    import pandas as pd
    df = pd.DataFrame(records)
    df_bo = df[df["in_bo"]]
    
    dists = np.sqrt(np.diff(df_bo["gt_e"])**2 + np.diff(df_bo["gt_n"])**2)
    tot_d = np.sum(dists)
    final_err = df_bo["err"].iloc[-1]
    drift_pct = (final_err / tot_d) * 100.0
    print(f"\nRESULTS FOR 30s BLACKOUT AT 120s:")
    print(f"  Distance: {tot_d:.1f} m")
    print(f"  Start Error: {df_bo['err'].iloc[0]:.2f} m")
    print(f"  Final Error: {final_err:.2f} m")
    print(f"  Max Error:   {df_bo['err'].max():.2f} m")
    print(f"  RMSE Error:  {np.sqrt(np.mean(df_bo['err']**2)):.2f} m")
    print(f"  Drift %:     {drift_pct:.2f}%")

if __name__ == "__main__":
    test_fixed_ekf()
