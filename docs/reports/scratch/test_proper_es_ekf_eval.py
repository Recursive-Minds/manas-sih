import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, r"C:\Users\carpe\.gemini\antigravity-ide\brain\10a4684a-0cfa-486c-a598-6a3b7a170b92")
from sih.data.downloader import download_iovnbd_trip
from sih.data.loader import GenericDataLoader
from sih.core.contracts import CalibratedSample, VelocityEstimate
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.calibration.mount import MountCalibrator
from sih.data.geo import geodetic_to_enu
from scratch.test_proper_es_ekf import FullErrorStateEKF
import torch

trip_s1 = GenericDataLoader().load_file(download_iovnbd_trip("S-S1"))
trip_s2 = GenericDataLoader().load_file(download_iovnbd_trip("S-S2"))

def get_trip_inferences(trip):
    ckpt = torch.load("models/checkpoints/best_velocity_model.pt", map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    model = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std = ckpt.get("norm_std", np.ones((8, 1), dtype=np.float32))
    
    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples:
        calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
    
    acc = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr = np.array([s.gyro_vehicle for s in calib_samples], dtype=np.float32)
    feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
    
    N = len(feats)
    window_size = 100
    windows = []
    for i in range(N):
        if i < window_size:
            pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
            w = np.vstack([pad, feats[:i+1]]).T
        else:
            w = feats[i - window_size + 1 : i + 1].T
        windows.append(w)
        
    windows_norm = (np.array(windows, dtype=np.float32) - norm_mean) / norm_std
    preds = []
    with torch.no_grad():
        for b in range(0, N, 1024):
            x = torch.from_numpy(windows_norm[b : b + 1024]).to(device)
            p, _ = model(x)
            preds.extend(p.cpu().numpy().flatten())
    return calib_samples, np.array(preds, dtype=np.float32)

print("Precomputing inferences...")
calib_s1, v_s1 = get_trip_inferences(trip_s1)
calib_s2, v_s2 = get_trip_inferences(trip_s2)

def run_single_eval(trip, calib_samples, v_preds, bo_start_s, bo_dur_s):
    ekf = FullErrorStateEKF(enable_nhc=True)
    ekf.reset(trip.gnss_samples[0])
    
    t0_ns = trip.imu_samples[0].timestamp_ns
    bo_start_ns = t0_ns + int(bo_start_s * 1e9)
    bo_end_ns = bo_start_ns + int(bo_dur_s * 1e9)
    
    gnss_idx = 0
    n_gnss = len(trip.gnss_samples)
    
    est_pts = []
    gt_pts = []
    ts_list = []
    nhc_dthetas = []
    bg_z_history = []
    
    for i, imu in enumerate(trip.imu_samples):
        t_curr = imu.timestamp_ns
        if t_curr > bo_end_ns + int(2e9):
            break
            
        while gnss_idx < n_gnss and trip.gnss_samples[gnss_idx].timestamp_ns <= t_curr:
            g = trip.gnss_samples[gnss_idx]
            in_bo = (bo_start_ns <= g.timestamp_ns <= bo_end_ns)
            if not in_bo:
                ekf.update_gnss(g)
            gnss_idx += 1
            
        calib = calib_samples[i]
        fwd_v = float(v_preds[i])
        m_state = "STATIONARY" if fwd_v < 0.2 else "DRIVING"
        vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=fwd_v, speed_variance=1.0, motion_state=m_state)
        fused = ekf.predict(calib, vel)
        
        bg_z_history.append((float((t_curr - t0_ns)*1e-9), float(np.degrees(ekf._bg[2]))))
        
        if bo_start_ns <= t_curr <= bo_end_ns:
            est_pts.append(fused.position_enu_m[:2])
            g_enu = geodetic_to_enu(
                trip.gnss_samples[min(gnss_idx, n_gnss-1)].latitude_deg,
                trip.gnss_samples[min(gnss_idx, n_gnss-1)].longitude_deg,
                0.0,
                trip.reference_lat_deg,
                trip.reference_lon_deg,
                0.0
            )
            gt_pts.append(g_enu[:2])
            ts_list.append(t_curr)
            nhc_dthetas.append(ekf.last_nhc_dtheta)
            
    est_pts = np.array(est_pts)
    gt_pts = np.array(gt_pts)
    
    # Distance in blackout
    g_start = geodetic_to_enu(trip.gnss_samples[0].latitude_deg, trip.gnss_samples[0].longitude_deg, 0, trip.reference_lat_deg, trip.reference_lon_deg, 0)[:2]
    # compute cumulative gt distance
    dists = np.sqrt(np.sum(np.diff(gt_pts, axis=0)**2, axis=1))
    tot_dist = max(float(np.sum(dists)), 1.0)
    
    final_err = float(np.linalg.norm(est_pts[-1] - gt_pts[-1]))
    drift_pct = (final_err / tot_dist) * 100.0
    
    return {
        "final_error_m": final_err,
        "drift_pct": drift_pct,
        "dist_m": tot_dist,
        "nhc_dthetas": nhc_dthetas,
        "bg_z_history": bg_z_history,
        "pre_bo_bg_z": bg_z_history[-len(nhc_dthetas)][1]
    }

print("\nEvaluating Scenario A (30s @ 120s on S-S1)...")
res_sc1 = run_single_eval(trip_s1, calib_s1, v_s1, 120.0, 30.0)
print(f"Scenario A with proper Kalman NHC & Gyro Observability:")
print(f"  Final Position Error: {res_sc1['final_error_m']:.2f} m (Old was 119.08 m)")
print(f"  Drift Percentage:     {res_sc1['drift_pct']:.2f} % (Old was 23.34 %)")
print(f"  Pre-Blackout bg_z:    {res_sc1['pre_bo_bg_z']:.4f} deg/s (Old was 0.0000 deg/s)")
print(f"  NHC dtheta (mean):    {np.mean(res_sc1['nhc_dthetas']):.4f} deg/step (Old was 0.0000 deg/step)")
print(f"  NHC dtheta (max):     {np.max(res_sc1['nhc_dthetas']):.4f} deg/step")

print("\nEvaluating Scenario B (60s @ 300s on S-S1)...")
res_sc2 = run_single_eval(trip_s1, calib_s1, v_s1, 300.0, 60.0)
print(f"Scenario B with proper Kalman NHC & Gyro Observability:")
print(f"  Final Position Error: {res_sc2['final_error_m']:.2f} m (Old was 254.07 m)")
print(f"  Drift Percentage:     {res_sc2['drift_pct']:.2f} % (Old was 31.21 %)")
print(f"  Pre-Blackout bg_z:    {res_sc2['pre_bo_bg_z']:.4f} deg/s (Old was 0.0000 deg/s)")
print(f"  NHC dtheta (mean):    {np.mean(res_sc2['nhc_dthetas']):.4f} deg/step")
print(f"  NHC dtheta (max):     {np.max(res_sc2['nhc_dthetas']):.4f} deg/step")
