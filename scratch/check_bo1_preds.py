import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.data.geo import geodetic_to_enu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = GenericDataLoader()
trip_s1 = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-S1.csv")

ckpt   = torch.load(r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt", map_location=device, weights_only=False)
model  = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

calibrator = MountCalibrator(window_size=100)
for g in trip_s1.gnss_samples: calibrator.observe_gnss(g)
calib_samples = [calibrator.update(imu) for imu in trip_s1.imu_samples]
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

t0_ns = trip_s1.imu_samples[0].timestamp_ns
bo_start_ns = t0_ns + int(1956.0 * 1e9)
bo_end_ns   = bo_start_ns + int(75.0 * 1e9)
ts = np.array([s.timestamp_ns for s in trip_s1.imu_samples])

idx_bo = np.where((ts >= bo_start_ns) & (ts <= bo_end_ns))[0]
print(f"Blackout 1 Index Range: {idx_bo[0]} to {idx_bo[-1]} (Count: {len(idx_bo)})")
print(f"AI Velocity Mean in Blackout: {np.mean(v_preds[idx_bo]):.2f} m/s ({np.mean(v_preds[idx_bo])*3.6:.1f} km/h)")
print(f"AI Integrated Distance in Blackout: {np.sum(v_preds[idx_bo]) * 0.01:.2f} meters")

# Ground truth GNSS distance in blackout
gt_start = min(trip_s1.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_start_ns))
gt_end   = min(trip_s1.gnss_samples, key=lambda g: abs(g.timestamp_ns - bo_end_ns))
p_start  = geodetic_to_enu(gt_start.latitude_deg, gt_start.longitude_deg, 0, trip_s1.reference_lat_deg, trip_s1.reference_lon_deg, 0)[:2]
p_end    = geodetic_to_enu(gt_end.latitude_deg, gt_end.longitude_deg, 0, trip_s1.reference_lat_deg, trip_s1.reference_lon_deg, 0)[:2]
gt_dist  = np.linalg.norm(p_end - p_start)
print(f"Ground Truth GNSS Distance in Blackout: {gt_dist:.2f} meters")
print(f"GNSS Start Speed: {gt_start.speed_mps} m/s, End Speed: {gt_end.speed_mps} m/s")
