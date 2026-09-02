import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu

loader = GenericDataLoader()
trip = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load(r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt", map_location=device, weights_only=False)
model  = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

calibrator = MountCalibrator(window_size=100)
for g in trip.gnss_samples: calibrator.observe_gnss(g)
calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]
acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)
feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
N = len(feats)
window_size = 100
windows = []
norm_mean = ckpt.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
norm_std  = ckpt.get("norm_std",  np.ones((8, 1),  dtype=np.float32))
for i in range(N):
    if i < window_size:
        pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
        w   = np.vstack([pad, feats[:i+1]]).T
    else:
        w   = feats[i - window_size + 1 : i + 1].T
    windows.append(w)
windows_norm = (np.array(windows, dtype=np.float32) - norm_mean) / norm_std
preds = []
with torch.no_grad():
    for b in range(0, N, 2048):
        x = torch.from_numpy(windows_norm[b : b + 2048]).to(device)
        p, _ = model(x)
        preds.extend(p.cpu().numpy().flatten())
v_preds = np.array(preds, dtype=np.float32)

def test_k_nhc(r_val):
    ekf = ErrorStateEKF(nhc_lateral_std=r_val, nhc_vertical_std=r_val)
    ekf.init_from_gnss(trip.gnss_samples[0])
    
    k_norms = []
    y_norms = []
    
    for j in range(2000, 3000): # Real driving segment
        imu = trip.imu_samples[j]
        cal = calib_samples[j]
        v_fwd = float(v_preds[j])
        vel = VelocityEstimate(timestamp_ns=imu.timestamp_ns, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state="DRIVING")
        
        # Calculate K_nhc manually
        C_b_n = ekf._q.as_matrix()
        C_n_b = C_b_n.T
        v_b = C_n_b @ ekf._v
        
        def skew(w):
            return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float64)
            
        H_nhc = np.zeros((2, 15), dtype=np.float64)
        H_nhc[0, 3:6] = C_n_b[1, :]
        H_nhc[0, 6:9] = -(C_n_b @ skew(ekf._v))[1, :]
        H_nhc[1, 3:6] = C_n_b[2, :]
        H_nhc[1, 6:9] = -(C_n_b @ skew(ekf._v))[2, :]
        
        y_nhc = np.array([0.0 - v_b[1], 0.0 - v_b[2]], dtype=np.float64)
        R_nhc = np.diag([r_val**2, r_val**2]).astype(np.float64)
        
        S_nhc = H_nhc @ ekf._P @ H_nhc.T + R_nhc
        K_nhc = ekf._P @ H_nhc.T @ np.linalg.inv(S_nhc)
        
        k_norms.append(np.linalg.norm(K_nhc))
        y_norms.append(np.linalg.norm(y_nhc))
        
        ekf.predict(cal, vel)
        
    return np.mean(k_norms), np.max(k_norms), np.mean(y_norms), np.max(y_norms)

k_mean_05, k_max_05, y_mean_05, y_max_05 = test_k_nhc(0.05)
k_mean_1,  k_max_1,  y_mean_1,  y_max_1  = test_k_nhc(1.00)

print(f"R_NHC = 0.05 -> K_NHC Mean Norm: {k_mean_05:.6f}, Max Norm: {k_max_05:.6f}")
print(f"R_NHC = 1.00 -> K_NHC Mean Norm: {k_mean_1:.6f}, Max Norm: {k_max_1:.6f}")
print(f"Innovation y_NHC Mean Norm: {y_mean_05:.8f}, Max Norm: {y_max_05:.8f}")
